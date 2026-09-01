"""Paged MHA/GQA KV pool with per-request page allocation.

Upstream NVIDIA path: python/freetoken/kvcache/mha_pool.py
Fill in: GitHub issue `kvcache` (see docs/architecture.md).

``MHAKVCache`` extends the Intel reference flat pool (``BaseKVCachePool``) with
real per-request page bookkeeping: ``allocate(req_id)`` hands a request a run
of ``page_size``-grained pool slots off a free-list, and ``free(req_id)`` gives
them back. This is the "allocate/free pages for a decode batch" half of the
issue's acceptance.

The buffer layout is *unchanged* from ``BaseKVCachePool``: ``k_buffer`` /
``v_buffer`` are each ``[num_layers, num_slots, num_kv_heads, head_dim]``. The
attention backends (SYCL in particular) index ``pool.k_buffer`` / ``pool.v_buffer``
directly against the engine's slot page table, so this class must keep that
exact shape and the identity ``slot == token position`` convention it relies on;
it only adds the free-list allocator on top.
"""
from __future__ import annotations

from typing import Dict

import torch

from .base import BaseKVCachePool


class MHAKVCache(BaseKVCachePool):
    """A paged MHA/GQA KV pool that allocates and frees ``page_size``-grained
    slot runs per request id, on top of the flat ``[L, S, H, D]`` buffer the
    Intel attention backends read.

    Page bookkeeping is a simple free-list over the pool's slot space. Slot 0 is
    reserved (it is the dummy / padding slot the engine's page table uses), so
    allocatable slots are ``[1, num_slots)``. ``allocate`` pops a contiguous run
    of ``num_pages * page_size`` slots; ``free`` returns them. Because the
    attention backends read through the engine's page table (position -> slot),
    the pool only owns the *allocation* of slots; the engine still installs the
    per-request page-table rows via ``attach_page_table``.
    """

    def __init__(self, model_config, page_size: int, num_pages: int, device, dtype):
        super().__init__(model_config, page_size, num_pages, device, dtype)
        # Free-list of allocatable slots. Slot 0 is reserved (dummy/padding), so
        # the pool hands out slots 1 .. num_slots-1. ``num_pages`` here is the
        # *page* count; each page holds ``page_size`` slots, so the pool has
        # ``num_slots = num_pages * page_size`` slots total, of which slot 0 is
        # reserved and the rest are allocatable.
        self._free_slots = list(range(1, self.num_slots))
        # Per-request allocation: req_id -> list of slot indices (in order).
        self._alloc: Dict[int, list] = {}

    @property
    def num_allocatable_slots(self) -> int:
        return len(self._free_slots)

    def allocate(self, req_id: int, num_pages: int | None = None) -> torch.Tensor:
        """Allocate ``num_pages`` pages (``num_pages * page_size`` slots) for
        request ``req_id`` and return them as a 1-D int64 slot-index tensor.

        ``num_pages`` defaults to the pool's full page count (a single-request
        pool); pass a smaller value for a batched pool where each request takes
        a slice. Raises ``RuntimeError`` if the free list cannot satisfy the
        request (the pool is full).
        """
        if num_pages is None:
            num_pages = self.num_pages
        need = num_pages * self.page_size
        if req_id in self._alloc:
            raise ValueError(f"request {req_id} already allocated; free it first")
        if need > len(self._free_slots):
            raise RuntimeError(
                f"KV pool full: cannot allocate {need} slots "
                f"({num_pages} pages x {self.page_size}); only {len(self._free_slots)} free"
            )
        # Pop a contiguous run from the front of the free list (FIFO; the engine
        # page table is identity-mapped, so contiguous-from-low is the natural
        # layout a single request occupies).
        slots = self._free_slots[:need]
        del self._free_slots[:need]
        self._alloc[req_id] = slots
        return torch.tensor(slots, dtype=torch.int64, device=self.device)

    def free(self, req_id: int) -> None:
        """Return request ``req_id``'s slots to the free list."""
        slots = self._alloc.pop(req_id, None)
        if slots is None:
            return
        self._free_slots.extend(slots)
        self._free_slots.sort()

    def is_allocated(self, req_id: int) -> bool:
        return req_id in self._alloc

    def allocated_slots(self, req_id: int) -> list:
        return list(self._alloc.get(req_id, ()))


__all__ = ["MHAKVCache"]
