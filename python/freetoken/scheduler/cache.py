"""Scheduler cache manager: radix prefix reuse for the live engine.

Upstream NVIDIA path: python/freetoken/scheduler/cache.py
Fill in: GitHub issue `kvcache` (see docs/architecture.md).

Thin wrapper over :class:`~freetoken.kvcache.radix_cache.RadixPrefixCache`
(the standalone, already-tested tree) that gives the engine the four
operations a real prefix-cache-driven request lifecycle needs: ``match`` a
new prompt against what's cached, ``lock``/``unlock`` a matched span so it
survives eviction while a request is actively using it, ``commit`` a
finished request's full token sequence into the tree for future reuse, and
``evict`` to reclaim pool pressure. Torch is only touched inside methods
(not at import time), matching this package's dual-venv contract.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from freetoken.kvcache.base import BaseCacheHandle, InsertResult, MatchResult


class CacheManager:
    """Owns one :class:`RadixPrefixCache` for the engine's live request
    lifecycle (issue `kvcache`, #12)."""

    def __init__(self, device: "torch.device", page_size: int) -> None:
        from freetoken.kvcache.radix_cache import RadixPrefixCache

        self.prefix_cache = RadixPrefixCache(device, page_size)

    def match(self, input_ids: "torch.Tensor") -> "MatchResult":
        """How much of ``input_ids`` is already cached, and a handle to it.

        The returned ``cached_len`` is page-aligned (never a partial page)
        -- see :class:`RadixPrefixCache`'s own ``_tree_walk``. Callers must
        :meth:`lock` the handle before consuming its matched indices (an
        unlocked match can be evicted out from under a caller that hasn't
        locked it yet), and :meth:`unlock` it once the request that used it
        finishes (or is committed -- see :meth:`commit`).
        """
        return self.prefix_cache.match_prefix(input_ids)

    def lock(self, handle: "BaseCacheHandle") -> None:
        self.prefix_cache.lock_handle(handle)

    def unlock(self, handle: "BaseCacheHandle") -> None:
        self.prefix_cache.lock_handle(handle, unlock=True)

    def commit(self, input_ids: "torch.Tensor", indices: "torch.Tensor") -> "InsertResult":
        """Insert a finished request's full token sequence + KV pool slot
        indices into the tree, for a future request to reuse.

        Slots that were already in the tree (the ``cached_len`` prefix this
        request itself matched at admission, via :meth:`match`) are NOT
        re-inserted (:meth:`RadixPrefixCache.insert_prefix` walks past them
        automatically); only the newly-extended suffix becomes a new node.
        Ownership of every slot index in ``indices[:insert_len]`` (rounded
        down to the page boundary) transfers to the tree from this point on
        -- the caller must NOT return them to the KV pool's per-request
        free-list; only :meth:`evict` ever does that, when the tree needs
        the space back.
        """
        return self.prefix_cache.insert_prefix(input_ids, indices)

    def evict(self, size: int) -> "torch.Tensor":
        """Reclaim at least ``size`` tokens' worth of KV pool slots from the
        least-recently-used, unlocked part of the tree. Returns the freed
        slot indices (the caller returns them to the KV pool, e.g.
        :meth:`freetoken.kvcache.mha_pool.MHAKVCache.free_slots`)."""
        return self.prefix_cache.evict(size)

    @property
    def evictable_size(self) -> int:
        return self.prefix_cache.size_info.evictable_size
