"""Per-request naive (non-sharing) KV cache.

Upstream NVIDIA path: python/freetoken/kvcache/naive_cache.py
Fill in: GitHub issue `kvcache` (see docs/architecture.md).

``NaiveKVCache`` is the no-sharing prefix cache: every request gets a private
page-table row, and nothing is reused across requests (no radix tree, no
prefix match). It is the baseline the radix cache is measured against, and it
is what the Intel engine loop effectively uses today (the flat pool + identity
page table, one row per request). It implements the ``BasePrefixCache``
interface so a caller can swap it for ``RadixPrefixCache`` without changing the
call site: ``match_prefix`` always reports zero (nothing is shared), and
``insert_prefix`` / ``evict`` are no-ops on the shared-cache accounting (each
request's pages are owned by its own page-table row, freed when the request
finishes -- not by this object).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from freetoken.core import get_global_ctx

from .base import (
    BaseCacheHandle,
    BasePrefixCache,
    InsertResult,
    MatchResult,
    SizeInfo,
)


@dataclass(frozen=True)
class NaiveCacheHandle(BaseCacheHandle):
    # cached_len is declared here (not inherited) so the frozen-dataclass
    # __init__ is generated cleanly. The naive handle has no back-reference.
    cached_len: int

    def get_matched_indices(self) -> torch.Tensor:
        return torch.empty(0, dtype=torch.int32)


class NaiveKVCache(BasePrefixCache):
    """A per-request KV prefix cache that shares nothing across requests.

    It exists to (a) give the radix cache a common ``BasePrefixCache`` call
    surface to be dropped-in compatible with, and (b) document that the
    engine's current flat-pool + identity-page-table behavior is exactly a
    naive (non-sharing) cache. ``match_prefix`` always returns ``cached_len=0``
    because no cross-request prefix is ever reused.
    """

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device if device is not None else get_global_ctx().device
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=self.device)
        # The naive cache shares nothing, so both accounting fields stay 0.
        self.evictable_size = 0
        self.protected_size = 0

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        # Nothing is shared, so there is nothing to protect or release.
        return None

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        return MatchResult(0, NaiveCacheHandle(0))

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        return InsertResult(0, NaiveCacheHandle(0))

    def evict(self, size: int) -> torch.Tensor:
        if size != 0:
            raise RuntimeError("NaiveKVCache owns no shared pages to evict")
        return self.empty_tensor

    def reset(self) -> None:
        self.evictable_size = 0
        self.protected_size = 0

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(evictable_size=0, protected_size=0)

    def check_integrity(self) -> None:
        return None


__all__ = ["NaiveKVCache", "NaiveCacheHandle"]
