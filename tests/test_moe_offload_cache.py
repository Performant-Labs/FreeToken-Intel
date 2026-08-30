"""Tests for the MoE LRU expert slot cache (issue #7, ADR 0002).

The cache is the CPU mirror of the upstream FreeToken ``OffloadMoeCache``:
per-layer host expert banks feed a single global pool of device slots via a
timestamp LRU shared by *all* layers (the only place 61 GB of experts fits).
Prefill materializes a whole layer into the pool (evicting the LRU-resident
experts of other layers); decode streams only the *missed* routed experts,
each into an evicted slot. These tests exercise that global-LRU bookkeeping
and the host->slot copy on a CPU device (machine-independent, no XPU / no
network). The XPU-specific piece (streaming the misses over PCIe / oneAPI)
is the scope of ``moe-offload`` and is not exercised here.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from freetoken.moe.offload_cache import OffloadMoeCache

DEVICE = torch.device("cpu")
L, E, S = 2, 4, 8  # 2 layers, 4 experts, 8 slots (S > E: the pool is not full)
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
    # The default pool S=8 > E=4, so materializing BOTH layers keeps all of each
    # layer resident (layer 0 in the 4 slots it first took, layer 1 in the 4 free
    # slots) -- this keeps the hit bookkeeping tests simple (a routed expert is a
    # hit, not a miss). The LRU-eviction tests below use a smaller pool (S == E)
    # to force a real cross-layer evict.
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


def test_second_layer_evicts_first_layer():
    # With a pool that holds exactly ONE layer (S == E), materializing layer 0
    # then layer 1 FORCES a real cross-layer evict: the pool is full after layer 0,
    # so layer 1's experts evict layer 0's (the global LRU). This is the ADR 0002
    # core: the 61 GB of experts only fits one (or a few) layers at a time.
    c = OffloadMoeCache(L, E, E, DEVICE, prefill_overlap=False)
    c.set_bank_sources(_bank_sources())
    c.materialize_layer(0)
    c.copy_missing()
    c.materialize_layer(1)
    # Global LRU: layer 1 is the MRU layer, so all of it stays resident; layer 0
    # is the LRU layer and is evicted (its slots are reused by layer 1).
    for e in range(E):
        assert c.slot_for_id[1, e].item() != -1, "layer 1 (MRU) must be resident"
        assert c.slot_for_id[0, e].item() == -1, "layer 0 (LRU) must be evicted"
    c.copy_missing()
    # Exactly layer 1's E experts are resident (the slot pool holds only layer 1).
    assert c.resident_slots(1) == list(range(E))
    assert c.resident_slots(0) == []
    # The slot cache now holds layer 1's bytes, not layer 0's.
    assert torch.equal(_slot_gate_up_row(c, 0), _bank_values(1, 0)[0])


def test_decode_hit_bumps_usage_no_copy():
    # With S > E (the default 8 > 4), after materializing layer 0 then layer 1
    # the pool is NOT full: layer 0 keeps 4 resident slots and layer 1's 4 experts
    # land in the 4 free slots. Routing a layer-1 expert is therefore a pure HIT
    # (already resident -> no new miss, no scheduled copy, slot id rewritten in
    # place) -- the hit bookkeeping path under the global-LRU scheme.
    c = OffloadMoeCache(L, E, S, DEVICE, prefill_overlap=False)
    c.set_bank_sources(_bank_sources())
    c.materialize_layer(0)
    c.materialize_layer(1)  # layer 0 resident (4 slots), layer 1 in the 4 free slots
    assert c.resident_slots(0) == [0, 1, 2, 3], "layer 0 stays resident when the pool is not full"
    assert c.resident_slots(1) == [4, 5, 6, 7], "layer 1 takes the free slots (hits on decode)"
    ids = torch.tensor([1], dtype=torch.int64)
    before_missing = c.stat_missing.item()
    c.ensure_experts(1, ids)
    # Expert 1 is already resident -> a hit: no new miss, no scheduled copy.
    assert c.stat_missing.item() == before_missing
    assert c.num_indices.item() == 0
    # The routed position was rewritten to the expert's (hit) slot.
    assert ids[0].item() == c.slot_for_id[1, 1].item()
    # Its slot's usage was bumped to the newest step (the LRU key moved).
    assert c.usage[c.slot_for_id[1, 1].item()].item() == c.step.item()


def test_decode_miss_evicts_least_recently_used():
    # The pool is a single GLOBAL LRU shared by all layers (ADR 0002): a miss
    # evicts the least-recently-used slot in the *entire* slot space (including
    # slots other layers hold -- the whole point of offload is that experts are
    # re-fetched on demand, never all resident at once). We seed a FULL pool whose
    # slots have distinct, ordered usage stamps, then route a NON-resident expert
    # -- a miss -- and assert the victim is the global LRU slot (min (usage, slot)).
    c = OffloadMoeCache(L, E, E, DEVICE, prefill_overlap=False)  # pool holds exactly one layer
    c.set_bank_sources(_bank_sources())
    c.materialize_layer(0)  # step 1: layer 0 fills slots [0, E) (usage 1)
    c.step.fill_(3)
    # Give each slot a distinct LRU age (unambiguous ordering): slot s -> usage s+1.
    c.usage[:] = torch.arange(E, dtype=torch.int64, device=DEVICE) + 1
    # The pool is now full. Make (layer 0, expert 0) -- the LRU slot 0 --
    # non-resident (as if layer 1 had evicted it during a later prefill) so that
    # routing it is a genuine miss.
    c.slot_for_id[0, 0] = -1
    # Route (layer 0, expert 0) -> a MISS in a full pool. It must evict the
    # global LRU slot. The other (more-recently-used) slots are owned by layer 0
    # experts 1..3, so the only LRU candidate is slot 0 (usage 1, now free) --
    # the miss re-acquires it. Assert the victim is the LRU, not a more-recent slot.
    c.ensure_experts(0, torch.tensor([0], dtype=torch.int64))
    assert c.slot_for_id[0, 0].item() == 0, "miss must take the global LRU (free) slot 0"
    # The more-recently-used slots (experts 1..3) are untouched.
    for e in range(1, E):
        assert c.slot_for_id[0, e].item() == e
    c.copy_missing()
    # Slot 0 now holds expert 0's exact host bytes (the evicted slot was reused).
    assert torch.equal(c.bank_caches["gate_up"][0], _bank_values(0, 0)[0])
    assert torch.equal(c.bank_caches["down"][0], _bank_values(0, 0)[1])
    assert c.usage[0].item() == c.step.item()  # re-stamped to the newest step


def test_decode_miss_evicts_full_pool_lru_victim():
    # The pool is FULL (every slot owned by layer 0) and we route a LAYER-1
    # expert: a miss, and the victim is the global LRU slot (min (usage, slot))
    # -- which is owned by a *different* layer. This cross-layer eviction is the
    # whole point of offload when the pool is smaller than the model's total
    # experts (it forces the LRU layer to be evicted so the new layer can fit).
    c = OffloadMoeCache(L, E, E, DEVICE, prefill_overlap=False)
    c.set_bank_sources(_bank_sources())
    # Seed a full pool: layer 0 owns all E slots, with slot 0 the LRU (usage 1)
    # and slot E-1 the MRU (usage E).
    for e in range(E):
        c.id_of_slot[e] = e  # flat id (layer 0, expert e)
        c.slot_for_id[0, e] = e
    c.step.fill_(5)
    c.usage[:] = torch.arange(E, dtype=torch.int64, device=DEVICE) + 1  # slot e -> usage e+1
    # Route (layer 1, expert 0): a miss (no layer-1 expert resident). Evict the
    # global LRU slot (slot 0, usage 1, currently (layer 0, expert 0)).
    c.ensure_experts(1, torch.tensor([0], dtype=torch.int64))
    # (layer 1, expert 0) took slot 0; (layer 0, expert 0) was evicted.
    assert c.slot_for_id[1, 0].item() == 0, "the layer-1 miss must take the global LRU slot 0"
    assert c.slot_for_id[0, 0].item() == -1, "the evicted (layer 0, expert 0) is no longer resident"
    # The MRU slots (other layer-0 experts) are untouched.
    assert c.slot_for_id[0, 1].item() == 1
    assert c.slot_for_id[0, E - 1].item() == E - 1
    c.copy_missing()
    assert torch.equal(c.bank_caches["gate_up"][0], _bank_values(1, 0)[0])


def test_copy_missing_streams_correct_host_bytes():
    # A decode miss must stream the *missed* expert's exact host bytes (gate/up
    # and down) into the evicted slot -- the bytes the forward will read. We seed
    # a FULL pool (layer 0 owns all E slots, slot 0 the LRU) and route a LAYER-1
    # expert: the miss evicts the global LRU slot 0 and must copy (layer 1,
    # expert 0)'s host row into it.
    c = OffloadMoeCache(L, E, E, DEVICE, prefill_overlap=False)
    c.set_bank_sources(_bank_sources())
    c.materialize_layer(0)  # layer 0 fills slots [0, E)
    c.step.fill_(3)
    c.usage[:] = torch.arange(E, dtype=torch.int64, device=DEVICE) + 1  # slot e -> usage e+1
    # Route (layer 1, expert 0): a miss (nothing of layer 1 resident). It evicts
    # the global LRU slot 0 (held by (layer 0, expert 0), usage 1) and stages
    # (slot 0, layer-1 host row 0) for the copy.
    c.ensure_experts(1, torch.tensor([0], dtype=torch.int64))
    assert c.slot_for_id[1, 0].item() == 0  # the miss re-acquired the LRU slot 0
    assert c.num_indices.item() == 1  # a miss was scheduled (evict slot 0, copy row 0)
    # Stream it: the slot must now hold expert 0's layer-1 host bytes exactly.
    c.copy_missing()
    exp_gu, exp_dn = _bank_values(1, 0)
    assert torch.equal(c.bank_caches["gate_up"][0], exp_gu)
    assert torch.equal(c.bank_caches["down"][0], exp_dn)


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


def _assert_maps_consistent(c: OffloadMoeCache) -> None:
    """The forward's per-layer ``slot_for_id`` rows must agree with the cache's
    global ``id_of_slot`` map. The forward indexes the slot pool by
    ``slot_for_id[layer][expert]`` (qwen3_5_moe), so a row entry pointing at a
    slot that ``id_of_slot`` attributes to the *other* layer makes the forward
    read the wrong expert's bytes -- a silent divergence from the in-VRAM path.
    This is the regression for issue #18 T2 (the stale cross-layer ``slot_for_id``
    entries that survive a cross-layer LRU eviction)."""
    for l in range(L):
        for e in range(E):
            s = int(c.slot_for_id[l, e].item())
            if s >= 0:
                owner = int(c.id_of_slot[s].item())
                assert owner == l * E + e, (
                    f"slot {s}: slot_for_id[{l},{e}] points at a slot owned by "
                    f"flat id {owner} (layer {owner // E}, expert {owner % E}), "
                    f"but the forward will read it as (layer {l}, expert {e})"
                )


def test_cross_layer_eviction_keeps_slot_for_id_consistent():
    # A pool smaller than the model's total experts (S=6 < 2*E=8) forces a
    # cross-layer LRU eviction: materializing layer 1 evicts layer 0's
    # least-recently-used slots. The eviction reassigns those slots (id_of_slot)
    # to layer 1, but the old bookkeeping only cleared the *evicting* layer's
    # slot_for_id row, leaving layer 0's row holding stale positive slot ids that
    # now point at layer 1's slots. The forward would then index the pool by
    # those stale ids and read the wrong expert (issue #18 T2: offload tokens
    # diverged from in-VRAM starting at token 7). The cache must keep
    # slot_for_id derived from id_of_slot so the two never disagree.
    c = OffloadMoeCache(L, E, 6, DEVICE, prefill_overlap=False)  # pool < both layers
    c.set_bank_sources(_bank_sources())

    # Prefill: materialize both layers; layer 1 evicts layer 0's LRU experts.
    c.materialize_layer(0)
    c.copy_missing()
    c.materialize_layer(1)
    c.copy_missing()
    # The evicted (layer 0, experts 0/1) rows must read -1, not stale slot ids.
    _assert_maps_consistent(c)
    assert int(c.slot_for_id[0, 0].item()) == -1, "evicted layer-0 expert must not stay resident"
    assert int(c.slot_for_id[0, 1].item()) == -1, "evicted layer-0 expert must not stay resident"

    # Decode: alternate layers, routing each layer's experts one at a time so the
    # global LRU keeps evicting across layers. The per-layer rows must track the
    # reassignments every step (the divergence appeared mid-decode, not at prefill).
    for l in range(L):
        for e in range(E):
            c.ensure_experts(l, torch.tensor([e], dtype=torch.int64))
            c.copy_missing()
            _assert_maps_consistent(c)
