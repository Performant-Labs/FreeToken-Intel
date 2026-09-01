"""Radix prefix cache: repeated-prompt match/insert, LRU evict, lock protection.

CPU-safe (dual-venv contract: torch lives in ``.venv-xpu``; the CPU venv
deselects via ``importorskip``). These tests lock in the second half of issue
``kvcache`` acceptance -- the radix tree half:

  * ``insert_prefix`` adds a page-aligned (token-key -> pool-slot) span, splitting
    an existing node at the divergence point.
  * ``match_prefix`` recovers the length of an incoming prompt already resident in
    the pool, plus the slot indices of that prefix (via the handle).
  * ``lock_handle`` / ``evict`` -- locked (in-flight) spans are protected from the
    LRU eviction that frees the least-recently-used, unref'd leaves.

``page_size`` is read from the global context, so each test installs a
``Context(page_size=...)`` via ``set_global_ctx`` and resets it in a fixture.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from freetoken.core import Context, reset_global_ctx, set_global_ctx  # noqa: E402
from freetoken.kvcache import RadixPrefixCache  # noqa: E402


@pytest.fixture(autouse=True)
def _ctx(page_size: int = 2):
    set_global_ctx(Context(page_size=page_size))
    yield
    reset_global_ctx()


def _cache(page_size: int = 2) -> RadixPrefixCache:
    return RadixPrefixCache(torch.device("cpu"), page_size=page_size)


def _ids(*vals: int) -> torch.Tensor:
    return torch.tensor(vals, dtype=torch.int32)


def _slots(n: int, start: int = 1) -> torch.Tensor:
    return torch.arange(start, start + n, dtype=torch.int32)


# --- insert + re-match of a repeated prompt ---------------------------------

def test_insert_then_match_full_prefix():
    cache = _cache()
    ins = cache.insert_prefix(_ids(10, 11, 12, 13, 14, 15), _slots(6))
    assert ins.cached_len == 0  # brand new prompt -> nothing was cached

    m = cache.match_prefix(_ids(10, 11, 12, 13, 14, 15))
    assert m.cached_len == 6
    # The handle walks the (possibly split) nodes back to root and returns the
    # pool slot indices the prefix is resident under.
    assert m.handle.get_matched_indices().tolist() == [1, 2, 3, 4, 5, 6]


def test_repeated_insert_is_a_full_hit():
    cache = _cache()
    ids = _ids(10, 11, 12, 13)
    cache.insert_prefix(ids, _slots(4))
    # A second insert of the same prompt is entirely already-cached.
    ins = cache.insert_prefix(ids, _slots(4, start=100))
    assert ins.cached_len == 4
    # The already-resident slots are what the second insert reports, NOT the new
    # indices it was handed (the new indices are discarded as a full hit).
    assert ins.handle.get_matched_indices().tolist() == [1, 2, 3, 4]


# --- partial overlap -> node split ------------------------------------------

def test_partial_overlap_splits_node_and_reports_prefix():
    cache = _cache()
    cache.insert_prefix(_ids(10, 11, 12, 13, 14, 15), _slots(6))  # 6-token node
    # Shares the first 4 tokens, diverges after. page_size=2 -> prefix aligns to 4.
    m = cache.match_prefix(_ids(10, 11, 12, 13, 99, 100))
    assert m.cached_len == 4
    assert m.handle.get_matched_indices().tolist() == [1, 2, 3, 4]
    cache.check_integrity()


def test_divergent_prompt_is_a_zero_hit():
    cache = _cache()
    cache.insert_prefix(_ids(10, 11, 12, 13), _slots(4))
    m = cache.match_prefix(_ids(20, 21, 22, 23))
    assert m.cached_len == 0
    assert m.handle.get_matched_indices().numel() == 0


def test_empty_prompt_matches_zero():
    cache = _cache()
    cache.insert_prefix(_ids(10, 11), _slots(2))
    assert cache.match_prefix(_ids()).cached_len == 0


# --- page alignment ----------------------------------------------------------

def test_insert_truncates_to_page_boundary():
    cache = _cache(page_size=2)  # 5 tokens -> aligned down to 4
    ins = cache.insert_prefix(_ids(1, 2, 3, 4, 5), _slots(5))
    assert ins.cached_len == 0
    assert cache.match_prefix(_ids(1, 2, 3, 4, 5)).cached_len == 4


def test_page_size_one_is_token_granular():
    cache = _cache(page_size=1)
    cache.insert_prefix(_ids(7, 8, 9), _slots(3))
    # page_size=1: the full 3-token prompt is page-aligned (no truncation).
    assert cache.match_prefix(_ids(7, 8, 9)).cached_len == 3
    # Diverge at the last token -> 2-token prefix.
    assert cache.match_prefix(_ids(7, 8, 99)).cached_len == 2


# --- lock / evict ------------------------------------------------------------

def test_lock_protects_from_evict_then_unlock():
    cache = _cache()
    cache.insert_prefix(_ids(10, 11, 12, 13, 14, 15), _slots(6))
    assert cache.size_info.evictable_size == 6

    h = cache.match_prefix(_ids(10, 11, 12, 13, 14, 15))
    cache.lock_handle(h.handle)
    si = cache.size_info
    assert si.protected_size == 6 and si.evictable_size == 0

    # Nothing is evictable while the in-flight span is locked.
    with pytest.raises(AssertionError):
        cache.evict(6)

    cache.lock_handle(h.handle, unlock=True)
    si = cache.size_info
    assert si.protected_size == 0 and si.evictable_size == 6
    cache.check_integrity()


def test_evict_frees_unreferenced_leaf():
    cache = _cache()
    cache.insert_prefix(_ids(10, 11, 12, 13, 14, 15), _slots(6))
    evicted = cache.evict(6)
    assert evicted.tolist() == [1, 2, 3, 4, 5, 6]
    si = cache.size_info
    assert si.evictable_size == 0 and si.protected_size == 0
    # After eviction the prompt is no longer resident.
    assert cache.match_prefix(_ids(10, 11, 12, 13, 14, 15)).cached_len == 0
    cache.check_integrity()


def test_evict_rejects_over_eviction():
    cache = _cache()
    cache.insert_prefix(_ids(10, 11, 12, 13), _slots(4))
    with pytest.raises(AssertionError):
        cache.evict(8)  # only 4 is evictable


def test_evict_zero_is_noop():
    cache = _cache()
    cache.insert_prefix(_ids(10, 11), _slots(2))
    assert cache.evict(0).numel() == 0
    assert cache.size_info.evictable_size == 2  # untouched


def test_evict_cascades_through_unrefed_parent():
    cache = _cache()
    # Root has two leaves (A: 2 tokens, B: 4 tokens) -> 6 evictable total.
    cache.insert_prefix(_ids(1, 2), _slots(2))
    cache.insert_prefix(_ids(3, 4, 5, 6), _slots(4, start=3))
    assert cache.size_info.evictable_size == 6
    # Evicting all 6 pops the LRU leaf, whose parent (root stays; the *other*
    # leaf is the second pop) -- the cascade frees a node whose only children
    # were already evicted and is itself unref'd.
    evicted = cache.evict(6)
    assert evicted.numel() == 6
    assert cache.size_info.evictable_size == 0
    cache.check_integrity()  # no dangling child pointers remain


def test_size_info_tracks_lock_unlock_balance():
    cache = _cache()
    cache.insert_prefix(_ids(10, 11, 12, 13), _slots(4))
    h = cache.match_prefix(_ids(10, 11, 12, 13))
    cache.lock_handle(h.handle)
    cache.lock_handle(h.handle)  # double lock -> ref_count 2
    assert cache.size_info.protected_size == 4  # counted once, not twice
    cache.lock_handle(h.handle, unlock=True)
    assert cache.size_info.protected_size == 4  # still held by the outer lock
    cache.lock_handle(h.handle, unlock=True)
    assert cache.size_info.protected_size == 0
    assert cache.size_info.evictable_size == 4


def test_check_integrity_passes_on_full_tree():
    cache = _cache()
    cache.insert_prefix(_ids(1, 2, 3, 4), _slots(4))
    cache.insert_prefix(_ids(1, 2, 5, 6), _slots(4, start=5))  # shares 1,2
    cache.check_integrity()
