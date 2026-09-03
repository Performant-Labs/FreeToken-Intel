"""HybridRadixCache: paired KV + GDN-snapshot radix tree (issue
`semantic-cache-hybrid-tree`, #169, part of the `semantic-cache` epic,
#32).

CPU-safe (dual-venv contract), mirrors test_kvcache_radix.py's own style
-- a SEPARATE tree class from RadixPrefixCache, sharing only RadixTreeNode/
the walk/split logic, so these tests never touch the plain KV radix at all.
``page_size`` is fixed (not read from global ctx) since HybridRadixCache
takes it directly in its constructor.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from freetoken.kvcache.hybrid_radix_cache import HybridRadixCache


def _cache(page_size: int = 2) -> HybridRadixCache:
    return HybridRadixCache(torch.device("cpu"), page_size=page_size)


def _ids(*vals: int) -> torch.Tensor:
    return torch.tensor(vals, dtype=torch.int32)


def _slots(n: int, start: int = 1) -> torch.Tensor:
    return torch.arange(start, start + n, dtype=torch.int32)


# --- insert + re-match --------------------------------------------------


def test_insert_donates_a_mamba_snapshot_at_the_end_boundary():
    cache = _cache()
    prefix_len, mamba_exist = cache.insert(_ids(10, 11, 12, 13), _slots(4), mamba_value=7)
    assert prefix_len == 0  # brand new -- nothing was already cached
    assert mamba_exist is False  # donated slot 7 was actually attached

    m = cache.match_prefix(_ids(10, 11, 12, 13))
    assert m.cached_len == 4
    assert m.mamba_value == 7
    assert m.kv_indices.tolist() == [1, 2, 3, 4]


def test_match_truncates_to_the_deepest_live_snapshot_boundary():
    """A continuation can only resume the GDN recurrence from a
    checkpointed boundary -- if the DEEPEST matched node has no live
    snapshot (tombstoned by a prior eviction, or simply never got one),
    match_prefix must walk back UP to the shallower node that still has
    one, not report the full (unrestorable) depth as cached."""
    cache = _cache()
    cache.insert(_ids(10, 11, 12, 13), _slots(4), mamba_value=7)
    # A longer prompt sharing the first 4 tokens as a prefix: node1 (len 4)
    # becomes an INTERNAL node once this child (node2, len 2 more) is
    # attached, with its OWN snapshot (9) at the full 6-token depth.
    cache.insert(_ids(10, 11, 12, 13, 14, 15), _slots(6), mamba_value=9)

    # Directly tombstone node2's snapshot (simulating it having been
    # independently evicted, or never having been snapshotted at all --
    # evict_mamba's own dedicated tests below cover the real eviction path
    # itself; this test is isolating match_prefix's own truncation logic).
    node2 = cache.match_prefix(_ids(10, 11, 12, 13, 14, 15)).node
    assert node2.mamba_value == 9
    node2.mamba_value = None
    cache.mamba_evictable -= 1

    m = cache.match_prefix(_ids(10, 11, 12, 13, 14, 15))
    # The full 6-token KV span still matches, but node2 no longer has a
    # live snapshot -- truncate to node1's boundary (cached_len=4,
    # mamba_value=7), the deepest point actually resumable.
    assert m.cached_len == 4
    assert m.mamba_value == 7


def test_insert_dedups_when_the_boundary_node_already_has_a_live_snapshot():
    cache = _cache()
    cache.insert(_ids(10, 11, 12, 13), _slots(4), mamba_value=7)
    prefix_len, mamba_exist = cache.insert(_ids(10, 11, 12, 13), _slots(4), mamba_value=9)
    assert mamba_exist is True  # caller must free its own donated slot 9

    m = cache.match_prefix(_ids(10, 11, 12, 13))
    assert m.mamba_value == 7  # the ORIGINAL snapshot wins, not the dup


def test_insert_into_root_reports_exist_since_root_cannot_hold_a_snapshot():
    cache = _cache()
    prefix_len, mamba_exist = cache.insert(_ids(), _slots(0), mamba_value=1)
    assert mamba_exist is True


# --- dual locking ---------------------------------------------------------


def test_inc_lock_protects_both_currencies():
    cache = _cache()
    cache.insert(_ids(10, 11, 12, 13), _slots(4), mamba_value=7)
    m = cache.match_prefix(_ids(10, 11, 12, 13))
    assert cache.full_evictable_size == 4
    assert cache.mamba_evictable_size == 1

    cache.inc_lock(m.node)
    assert cache.full_evictable_size == 0
    assert cache.mamba_evictable_size == 0

    cache.dec_lock(m.node)
    assert cache.full_evictable_size == 4
    assert cache.mamba_evictable_size == 1


def test_locked_snapshot_survives_eviction_pressure():
    cache = _cache()
    cache.insert(_ids(10, 11, 12, 13), _slots(4), mamba_value=7)
    m = cache.match_prefix(_ids(10, 11, 12, 13))
    cache.inc_lock(m.node)

    # Nothing evictable while locked.
    result = cache.evict_mamba(1)
    assert result.mamba_slots == []
    assert cache.mamba_evictable_size == 0


# --- dual eviction ----------------------------------------------------------


def test_evict_full_frees_kv_and_the_leaf_snapshot_together():
    cache = _cache()
    cache.insert(_ids(10, 11, 12, 13), _slots(4), mamba_value=7)
    assert cache.full_evictable_size == 4
    assert cache.mamba_evictable_size == 1

    result = cache.evict_full(4)
    assert set(result.kv_indices.tolist()) == {1, 2, 3, 4}
    assert result.mamba_slots == [7]
    assert cache.full_evictable_size == 0
    assert cache.mamba_evictable_size == 0


def test_evict_mamba_on_an_internal_node_tombstones_without_freeing_kv():
    """Internal node -> TOMBSTONE (free the slot, keep KV + children):
    evict_mamba alone must never delete a node that still has live
    descendants relying on its KV as a prefix."""
    cache = _cache()
    cache.insert(_ids(10, 11, 12, 13), _slots(4), mamba_value=7)
    # A second, longer prompt shares the first 4 tokens as a prefix and
    # extends it -- the shared node becomes an INTERNAL node (has a child).
    cache.insert(_ids(10, 11, 12, 13, 14, 15), _slots(6), mamba_value=9)

    assert cache.mamba_evictable_size == 2  # both boundary nodes hold a snapshot
    result = cache.evict_mamba(1)
    assert result.mamba_slots == [7]  # the older (internal) node's snapshot
    assert result.kv_indices.numel() == 0  # its KV was NOT freed (still a prefix dependency)
    assert cache.mamba_evictable_size == 1

    # The KV is still there: the longer prompt still matches in full.
    m = cache.match_prefix(_ids(10, 11, 12, 13, 14, 15))
    assert m.kv_indices.numel() == 6


def test_evict_mamba_on_a_leaf_frees_kv_too_and_cascades():
    cache = _cache()
    cache.insert(_ids(10, 11, 12, 13), _slots(4), mamba_value=7)

    result = cache.evict_mamba(1)
    assert result.mamba_slots == [7]
    assert set(result.kv_indices.tolist()) == {1, 2, 3, 4}
    assert cache.full_evictable_size == 0
    assert cache.mamba_evictable_size == 0


def test_check_integrity_passes_after_a_sequence_of_operations():
    cache = _cache()
    cache.insert(_ids(10, 11, 12, 13), _slots(4), mamba_value=7)
    cache.insert(_ids(10, 11, 12, 13, 14, 15), _slots(6), mamba_value=9)
    m = cache.match_prefix(_ids(10, 11, 12, 13, 14, 15))
    cache.inc_lock(m.node)
    cache.dec_lock(m.node)
    cache.evict_mamba(1)
    cache.check_integrity()  # must not raise
