"""KV pool / handle interfaces.

Upstream NVIDIA path: python/freetoken/kvcache/base.py
Fill in: GitHub issue `kvcache` (see docs/architecture.md).

This is the minimal paged KV pool the Intel engine loop drives. It is a
deliberately simple, pure-torch implementation: a flat pool of ``num_pages``
pages (each holding ``page_size`` token slots) plus a per-request page table
that maps a token position to a pool slot. Read/write are index-based
(``index_select``), which runs identically on CPU and XPU. The production
radix/hybrid pools are separate issues.
"""
from __future__ import annotations

import torch


class BaseCacheHandle:
    def __init__(self, *args, **kwargs) -> None:
        pass


class BaseKVCachePool:
    """A flat paged K/V store.

    Layout: ``k_buffer`` / ``v_buffer`` are each ``[num_pages * page_size,
    num_kv_heads, head_dim]`` -- one row per token slot. ``page_table`` is
    ``[num_requests, max_seq_len]`` of int32 slot indices, filled in by the
    engine (an identity map suffices: slot ``pos`` holds token at position
    ``pos``). Attention reads the keys/values for positions ``[0, len)`` via
    ``page_table[req, pos]``.
    """

    def __init__(self, model_config, page_size: int, num_pages: int, device, dtype):
        # ``model_config`` is the *parsed* ModelConfig (NOT the EngineConfig) so
        # the pool never forces an HF-config lookup. ``page_size`` is the engine's
        # paging granularity (1 == token-granular slots).
        self.model_config = model_config
        self.device = device
        self.dtype = dtype
        self.page_size: int = page_size
        self.num_pages = num_pages
        self.num_slots = num_pages * page_size
        self.num_kv_heads = model_config.num_key_value_heads
        # Some models (Qwen3.5/3.6) set an explicit head_dim that differs from
        # hidden // num_heads; honor it when present, else derive.
        self.head_dim = getattr(model_config, "head_dim", None) or (
            model_config.hidden_size // model_config.num_attention_heads
        )
        self.k_buffer = torch.empty(
            (self.num_slots, self.num_kv_heads, self.head_dim), device=device, dtype=dtype
        )
        self.v_buffer = torch.empty(
            (self.num_slots, self.num_kv_heads, self.head_dim), device=device, dtype=dtype
        )

    def attach_page_table(self, page_table: torch.Tensor) -> None:
        """Bind the engine's page table (slot indices) to this pool."""
        self.page_table = page_table

    def write_kv(self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor) -> None:
        """Write per-token keys/values into the pool at slots ``out_loc``.

        ``out_loc`` is ``[num_tokens]`` int64 slot indices. The attention block
        hands ``k`` / ``v`` in **head-major** ``[num_kv_heads, num_tokens,
        head_dim]`` (the order its own projections use), so we transpose them
        back to token-major ``[num_tokens, num_kv_heads, head_dim]`` before the
        slot scatter (out_loc is ordered by token).
        """
        k_tok = k.transpose(0, 1).contiguous()
        v_tok = v.transpose(0, 1).contiguous()
        self.k_buffer[out_loc.long()] = k_tok.reshape(out_loc.numel(), *k_tok.shape[1:])
        self.v_buffer[out_loc.long()] = v_tok.reshape(out_loc.numel(), *v_tok.shape[1:])

    def read_kv(self, table_idx: int, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Read keys/values for request ``table_idx`` at token positions ``pos``
        (``[num_tokens]``). The page table row maps position -> pool slot, so
        this is a per-request gather. Returns ``[num_tokens, num_kv_heads,
        head_dim]`` for K and V.
        """
        slots = self.page_table[table_idx, pos.long()]
        return self.k_buffer[slots], self.v_buffer[slots]

    def allocate(self, *args, **kwargs):
        # Page bookkeeping for the minimal pool is the identity map the engine
        # installs in ``attach_page_table``; nothing to allocate lazily.
        return 0

    def free(self, *args, **kwargs) -> None:
        pass


def create_kv_pool(model_config, page_size: int, num_pages: int, device, dtype) -> BaseKVCachePool:
    """Build the engine's KV pool from a parsed ModelConfig.

    (The production pool family -- radix / hybrid-SWA -- is a separate issue;
    the Intel engine loop uses this flat paged pool.)
    """
    return BaseKVCachePool(model_config, page_size, num_pages, device, dtype)


__all__ = ["BaseKVCachePool", "BaseCacheHandle", "create_kv_pool"]
