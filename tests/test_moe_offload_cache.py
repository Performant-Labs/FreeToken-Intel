"""Tests for the MoE LRU expert slot cache (issue #7, ADR 0002).

The cache is the CPU mirror of the upstream FreeToken ``OffloadMoeCache``:
per-layer host expert banks feed a small pool of device slots via a
timestamp LRU (prefill materializes a whole layer, decode streams only the
missed experts). These tests exercise the LRU bookkeeping and the
host->slot copy on a CPU device (machine-independent, no XPU / no network),
so they run in the CPU venv. The XPU-specific piece (streaming the misses
over PCIe / oneAPI) is the scope of ``moe-offload`` and is not exercised
here.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from freetoken.moe.offload_cache import OffloadMoeCache

DEVICE = torch.device("cpu")
L, E, S = 2, 4, 8  # 2 layers, 4 experts, 8 slots (== 2E, the double-buffer floor)
H, I = 16, 8  # hidden, moe intermediate (small so the banks are tiny)

# Distinguishable host bank contents: bank value at (layer, expert, ...) is
# 100*(layer+1) + 10*(expert+1) + <0 for gate rows, +1 for up rows (gate_up),
# and 100*(layer+1) + 10*(expert+1) + 2 for down rows. The constants are small
# enough to be exact in float32 and distinct per (layer, expert, projection),
# so a test can tell which (layer, expert, half) a slot currently holds.
def _bank_values(layer: int, expert: int) -> tuple[torch.Tensor, torch.Tensor]:
    base = 100 * (layer + 1) + 10 * (expert + 1)
    gate = torch.full((I, H), base + 0.0, dtype=torch.float32)
    up = torch.full((I, H), base + 1.0, dtype=torch.float32)
    down = torch.full((H, I), base + 2.0, dtype=torch.float32)
    return torch.cat([gate, up], dim=0), down  # gate_up row [2I, H], down row [H, I]


@pytest.fixture
def cache():
    # The default pool is the double buffer plus TWO decode slots (S == 2E + 2),
    # so a decode miss can always be placed without evicting a double-buffer
    # slot -- this keeps the hit/miss bookkeeping tests simple. The LRU eviction
    # tests below use a smaller pool (2E + 1) to force a real evict.
    c = OffloadMoeCache(L, E, S, DEVICE, prefill_overlap=False)
    c.set_bank_sources(
        {
            "gate_up": [torch.stack([_bank_values(l, e)[0] for e in range(E)]) for l in range(L)],
            "down": [torch.stack([_bank_values(l, e)[1] for e in range(E)]) for l in range(L)],
        }
    )
    return c


def _bank_sources():
    return {
        "gate_up": [torch.stack([_bank_values(l, e)[0] for e in range(E)]) for l in range(L)],
        "down": [torch.stack([_bank_values(l, e)[1] for e in range(E)]) for l in range(L)],
    }


def _slot_gate_up_row(cache: OffloadMoeCache, slot: int) -> torch.Tensor:
    return cache.bank_caches["gate_up"][slot]


def test_prefill_materializes_whole_layer_into_slots_0_to_e(cache):
    # materialize_layer only STAGES the copy plan (evict_slots/src_indices/
    # num_indices) and updates the slot maps; the actual host->slot byte copy
    # happens in copy_missing(). Stage, then copy, then assert on the bytes.
    cache.materialize_layer(0)
    assert cache.num_indices.item() == E  # the whole layer was staged
    for e in range(E):
        assert cache.slot_for_id[0, e].item() == e  # identity slot map
    cache.copy_missing()  # stream the host rows into the slots
    gu, dn = cache.bank_views(n=E)  # first E slots = the materialized layer
    # Slot k == expert k for layer 0: each slot now holds expert k's bytes.
    for e in range(E):
        exp_gu, exp_dn = _bank_values(0, e)
        assert torch.equal(_slot_gate_up_row(cache, e), exp_gu), (e, "gate_up")
        assert torch.equal(cache.bank_caches["down"][e], exp_dn), (e, "down")


def test_second_layer_evicts_first_layer(cache):
    cache.materialize_layer(0)
    cache.copy_missing()
    cache.materialize_layer(1)
    # Layer 1 now owns slots [0, E): the last materialized layer is resident.
    for e in range(E):
        assert cache.slot_for_id[1, e].item() == e
        assert cache.slot_for_id[0, e].item() == -1, "layer 0 must be evicted"
    cache.copy_missing()
    # The slot cache now holds layer 1's bytes, not layer 0's.
    assert torch.equal(_slot_gate_up_row(cache, 0), _bank_values(1, 0)[0])


def test_decode_hit_bumps_usage_no_copy():
    # Use the default pool (2E + 2): after materializing layer 1, all of layer 1
    # is resident in the double buffer, so routing any of its experts is a pure
    # hit (no decode-slot copy, no evict).
    c = OffloadMoeCache(L, E, S, DEVICE, prefill_overlap=False)
    c.set_bank_sources(_bank_sources())
    c.materialize_layer(0)
    c.materialize_layer(1)  # evicts layer 0; resident = layer 1 (double buffer)
    ids = torch.tensor([1], dtype=torch.int64)
    before_missing = c.stat_missing.item()
    c.ensure_experts(1, ids)
    # Expert 1 is already resident -> a hit: no new miss, no scheduled copy.
    assert c.stat_missing.item() == before_missing
    assert c.num_indices.item() == 0
    # The routed position was rewritten to the expert's slot (expert 1 -> slot 1).
    assert ids[0].item() == 1
    # Its slot's usage was bumped to the newest step (the LRU key moved).
    assert c.usage[1].item() == c.step.item()


def test_decode_miss_evicts_least_recently_used():
    # A pool that can hold the double buffer plus exactly THREE decode experts
    # (cache_size == 2E + 3: slots [0, E) are the non-evictable double buffer;
    # slots 2E, 2E+1, 2E+2 are the evictable decode slots). We fill all three
    # decode slots with layer-0 experts at three distinct, known steps, then
    # route a 4th expert -- a MISS -- and assert the victim is the
    # LEAST-recently-used decode slot (min (usage, slot)), never a double-buffer
    # slot and never a more-recently-used decode slot.
    c = OffloadMoeCache(L, E, 2 * E + 3, DEVICE, prefill_overlap=False)
    c.set_bank_sources(_bank_sources())
    c.materialize_layer(0)  # step 1: layer 0 experts in the double buffer [0, E)
    # Fill the three decode slots with layer-0 experts at distinct steps, with
    # slot 2E the OLDEST (LRU) and slot 2E+2 the NEWEST (MRU). We seed the maps
    # directly (a real run would reach this via decode steps) and stamp distinct
    # usage values so the LRU ordering is unambiguous: 2E < 2E+1 < 2E+2.
    #   slot 2E   <- (layer 0, expert 0), usage 1  (LRU)
    #   slot 2E+1 <- (layer 0, expert 1), usage 2
    #   slot 2E+2 <- (layer 0, expert 2), usage 3  (MRU)
    # The double buffer is left holding experts 3 (slot 3) and is NOT evictable.
    c.step.fill_(3)
    # Experts 0,1,2 live in the decode slots; expert 3's double-buffer residency
    # is consumed by the decode handoff (cleared) so it is a genuine miss.
    c.slot_for_id[0, 0] = 2 * E
    c.slot_for_id[0, 1] = 2 * E + 1
    c.slot_for_id[0, 2] = 2 * E + 2
    c.slot_for_id[0, 3] = -1  # double-buffer slot 3 freed -> expert 3 is a miss
    c.id_of_slot[0] = -1
    c.id_of_slot[1] = -1
    c.id_of_slot[2] = -1
    c.id_of_slot[3] = -1
    c.id_of_slot[2 * E] = 0
    c.id_of_slot[2 * E + 1] = 1
    c.id_of_slot[2 * E + 2] = 2
    c.usage[2 * E] = 1
    c.usage[2 * E + 1] = 2
    c.usage[2 * E + 2] = 3
    c.usage[0] = 0
    c.usage[1] = 0
    c.usage[2] = 0
    c.usage[3] = 0
    # Route (layer 0, expert 3) -> a MISS. It must evict the LRU decode slot
    # (slot 2E, expert 0, usage 1) and take its place.
    c.ensure_experts(0, torch.tensor([3], dtype=torch.int64))
    assert c.slot_for_id[0, 3].item() == 2 * E, "miss expert 3 must take the LRU slot 2E"
    assert c.slot_for_id[0, 0].item() == -1, "LRU expert 0 (slot 2E) must be evicted"
    # The more-recently-used decode slots are untouched.
    assert c.slot_for_id[0, 1].item() == 2 * E + 1
    assert c.slot_for_id[0, 2].item() == 2 * E + 2
    c.copy_missing()
    # Slot 2E now holds expert 3's exact host bytes (the evicted slot was reused).
    assert torch.equal(c.bank_caches["gate_up"][2 * E], _bank_values(0, 3)[0])
    assert torch.equal(c.bank_caches["down"][2 * E], _bank_values(0, 3)[1])
    assert c.usage[2 * E].item() == c.step.item()  # re-stamped to the newest step


def test_copy_missing_streams_correct_host_bytes():
    # A decode miss must stream the *missed* expert's exact host bytes (gate/up
    # and down) into the evicted slot -- the bytes the forward will read.
    c = OffloadMoeCache(L, E, 2 * E + 1, DEVICE, prefill_overlap=False)
    c.set_bank_sources(
        {
            "gate_up": [torch.stack([_bank_values(l, e)[0] for e in range(E)]) for l in range(L)],
            "down": [torch.stack([_bank_values(l, e)[1] for e in range(E)]) for l in range(L)],
        }
    )
    c.materialize_layer(1)  # layer 1 resident in the double buffer slots [0, E)
    # Simulate the prefill->decode handoff: the decode slot 2E now holds
    # (layer 1, expert 0) (an old usage stamp) and the double-buffer slot 0 is
    # freed; (layer 1, expert 0) is marked NOT resident, so the next decode step
    # that routes expert 0 is a genuine MISS that must evict slot 2E and stream
    # expert 0's layer-1 host row back into it.
    c.slot_for_id[1, 0] = -1  # not resident -> the next routing of expert 0 is a miss
    c.id_of_slot[0] = -1  # free the double-buffer slot
    c.id_of_slot[2 * E] = 4  # flat id of (layer 1, expert 0)
    c.usage[2 * E] = 1
    c.ensure_experts(1, torch.tensor([0], dtype=torch.int64))
    assert c.slot_for_id[1, 0].item() == 2 * E  # the miss re-acquired the decode slot
    assert c.num_indices.item() == 1  # a miss was scheduled (evict slot 2E, copy row 0)
    # Stream it: the slot must now hold expert 0's layer-1 host bytes exactly.
    c.copy_missing()
    exp_gu, exp_dn = _bank_values(1, 0)
    assert torch.equal(c.bank_caches["gate_up"][2 * E], exp_gu)
    assert torch.equal(c.bank_caches["down"][2 * E], exp_dn)


def test_hit_miss_counters(cache):
    cache.materialize_layer(0)
    cache.materialize_layer(1)
    base_missing = cache.stat_missing.item()
    base_calls = cache.stat_calls.item()
    # A hit (expert 0 already resident after materialize(1)).
    cache.ensure_experts(1, torch.tensor([0], dtype=torch.int64))
    assert cache.stat_calls.item() == base_calls + 1
    assert cache.stat_missing.item() == base_missing  # no new miss on a hit
    # reset_stats clears them.
    cache.reset_stats()
    assert cache.stat_missing.item() == 0
    assert cache.stat_calls.item() == 0


def test_bank_views_full_and_prefill(cache):
    full = cache.bank_views()
    assert len(full) == 2  # gate_up + down
    assert full[0].shape == (S, 2 * I, H)
    assert full[1].shape == (S, H, I)
    head = cache.bank_views(n=E)
    assert head[0].shape == (E, 2 * I, H)
