"""Tests for :meth:`OffloadMoeCache.rebuild` (issue #16, elastic-memory).

``rebuild`` is the runtime path that lets the engine re-plan the MoE cache / KV
split (off the device's free VRAM) and resize the MoE slot pool **without
reloading any weights**: the host source banks stay put and only the device slot
pool + LRU bookkeeping are re-allocated.

These tests build the cache on the CPU (machine-independent, no XPU / no
network) so the invariant is covered in the per-PR / CPU nightly. The
``xpu``-marked ``test_moe_cache_budget_xpu.py`` separately confirms the *engine*
calls ``rebuild`` at the re-planned size on a real B70 (that one needs the
shared GPU and so runs only where the XPU is quiescent).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from freetoken.moe.offload_cache import OffloadMoeCache

DEVICE = torch.device("cpu")
# E=8 so the constructor's prefill double-buffer (cache_size >= 2*num_experts)
# fits a pool as small as 2*E=16; the banks stay tiny (H, I below).
L, E = 2, 8
H, I = 16, 8  # hidden, moe intermediate


def _bank_sources():
    # Distinguishable per (layer, expert) host rows: gate [I, H], up [I, H],
    # down [H, I]; the value tag is 100*(l+1)+10*(e+1) (+0/+1/+2 per bank).
    def rows(layer, expert):
        base = 100 * (layer + 1) + 10 * (expert + 1)
        gate_up = torch.full((2 * I, H), base, dtype=torch.float32)
        gate_up[:I] += 0.0  # gate rows
        gate_up[I:] += 1.0  # up rows
        down = torch.full((H, I), base + 2.0, dtype=torch.float32)
        return gate_up, down

    return {
        "gate_up": [torch.stack([rows(l, e)[0] for e in range(E)]) for l in range(L)],
        "down": [torch.stack([rows(l, e)[1] for e in range(E)]) for l in range(L)],
    }


def _source_ids(bank_sources):
    # The host bank objects are the allocation rebuild must NOT touch.
    return [id(t) for lst in bank_sources.values() for t in lst]


def test_rebuild_grows_pool_and_resets_lru():
    c = OffloadMoeCache(L, E, E, DEVICE, prefill_overlap=False)  # pool == one layer (full)
    c.set_bank_sources(_bank_sources())
    src_ids_before = _source_ids(c.bank_sources)

    # Seed a full pool with an LRU history that rebuild must discard.
    c.materialize_layer(0)
    c.copy_missing()
    c.materialize_layer(1)
    c.copy_missing()  # layer 1 evicted layer 0 (cross-layer LRU); layer 1 resident
    c.step.fill_(9)
    c.usage[:] = torch.arange(E, dtype=torch.int64, device=DEVICE) + 1
    assert int(c.slot_for_id[1, 0].item()) != -1, "layer 1 is resident pre-rebuild"
    assert int(c.slot_for_id[0, 0].item()) == -1, "layer 0 is evicted pre-rebuild"

    new_size = E + 4  # grow the pool (4 extra slots)
    c.rebuild(new_size)

    # The size-dependent tensors are all re-allocated at the new size.
    assert c.cache_size == new_size
    assert c.id_of_slot.shape == (new_size,)
    assert c.usage.shape == (new_size,)
    assert c.evict_slots.shape == (new_size,)
    assert c.src_indices.shape == (new_size,)
    # The per-bank slot caches grew but kept their row shape / dtype.
    assert c.bank_caches["gate_up"].shape == (new_size, 2 * I, H)
    assert c.bank_caches["down"].shape == (new_size, H, I)

    # The host source banks are the SAME objects (no re-load of weights).
    assert _source_ids(c.bank_sources) == src_ids_before
    # The (sources, cache) pairs were rebuilt against the new caches.
    for _, cache in c.banks:
        assert cache.shape[0] == new_size

    # The LRU bookkeeping was cleared: both the forward (slot_for_id, [L, E]) and
    # inverse (id_of_slot, [cache_size]) maps are re-filled with -1, so their sums
    # are -L*E and -new_size respectively (NOT 0); the counter tensors are zeroed.
    assert int(c.slot_for_id.sum().item()) == -(L * E)
    assert int(c.id_of_slot.sum().item()) == -new_size
    assert int(c.usage.sum().item()) == 0
    assert int(c.step.item()) == 0
    assert int(c.active_mask.sum().item()) == 0
    assert c._pending_src_layer is None
    assert c._pending_whole_layer is False


def test_rebuild_can_shrink_pool_to_num_experts():
    c = OffloadMoeCache(L, E, E * 2, DEVICE, prefill_overlap=False)  # pool = 2 layers
    c.set_bank_sources(_bank_sources())
    c.materialize_layer(0)
    c.copy_missing()
    c.materialize_layer(1)
    c.copy_missing()
    assert c.cache_size == E * 2

    # Shrink back down to the minimum (one full layer's worth of slots).
    c.rebuild(E)
    assert c.cache_size == E
    assert c.id_of_slot.shape == (E,)
    assert c.bank_caches["gate_up"].shape == (E, 2 * I, H)
    assert int(c.slot_for_id.sum().item()) == -(L * E)
    assert int(c.step.item()) == 0


def test_rebuild_rejects_size_below_num_experts():
    c = OffloadMoeCache(L, E, E, DEVICE, prefill_overlap=False)
    c.set_bank_sources(_bank_sources())
    with pytest.raises(ValueError, match="must be >= num_experts"):
        c.rebuild(E - 1)
    # The cache is untouched by the rejected rebuild.
    assert c.cache_size == E


def test_rebuild_without_banks_is_a_safe_noop_on_banks():
    # The engine only ever calls rebuild() on a live cache whose banks were
    # registered at load, so a bank-less rebuild is an unreachable edge -- but it
    # must not crash (a KeyError would take down the engine on a malformed state).
    # With no banks, the guard skips the per-bank reallocation; the LRU maps are
    # still re-allocated at the new size and cleared.
    c = OffloadMoeCache(L, E, E, DEVICE, prefill_overlap=False)
    c.rebuild(E + 2)
    assert c.cache_size == E + 2
    assert c.id_of_slot.shape == (E + 2,)
    assert int(c.slot_for_id.sum().item()) == -(L * E)
    assert int(c.step.item()) == 0
    # Nothing was registered, so the bank dicts stay empty (no crash, no ghosts).
    assert c.bank_sources == {}
    assert c.bank_caches == {}
    assert c.banks == []
