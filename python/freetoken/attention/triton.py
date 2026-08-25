"""Attention backend for the Intel Arc Pro B70 (XPU).

Upstream NVIDIA path: python/freetoken/attention/triton.py
Fill in: GitHub issue `attn-triton` (see docs/architecture.md).

This is the *reference* attention the Intel engine loop runs: a correct,
dependency-free (pure torch) GQA/flash-style attention that executes on the
XPU (and CPU). It implements the ``BaseAttnBackend`` contract so the model's
``forward`` is backend-agnostic. A hand-tuned Triton-Intel / SYCL kernel is a
follow-up; this one is exact and is what makes ``ft serve`` produce tokens.
"""
from __future__ import annotations

import torch

from .base import AttentionSpec, AttnType, BaseAttnBackend, BaseAttnMetadata


class TritonMetadata(BaseAttnMetadata):
    """Per-layer gather indices for the decode phase (one new token per request)."""

    def __init__(self, seq_lens: torch.Tensor, qo_ind: torch.Tensor, kv_lens: torch.Tensor) -> None:
        self.seq_lens = seq_lens
        self.qo_ind = qo_ind
        self.kv_lens = kv_lens

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.qo_ind[:bs]


class TritonAttentionBackend(BaseAttnBackend):
    """Pure-torch grouped-query attention.

    ``forward(q, k, v, layer_id, batch, attn_spec)`` takes ``q`` as
    ``[num_tokens, num_heads, head_dim]`` and ``k`` / ``v`` as
    ``[num_tokens, num_kv_heads, head_dim]`` (the *new* tokens this step), and
    returns ``[num_tokens, num_heads, head_dim]``. K/V for the whole sequence
    are read from the KV pool via the page table, so the buffer holds the full
    history and each step only appends the new tokens.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.device = torch.device("xpu" if _xpu_available() else "cpu")
        self.capture = None
        self.capture_bs = []
        self.max_graph_bs = 0

    # -- BaseAttnBackend interface -------------------------------------------

    def prepare_metadata(self, batch) -> None:
        # For the reference backend the per-layer gather indices are derived
        # directly from the batch inside forward(); nothing to precompute.
        return None

    def init_capture_graph(self, max_seq_len: int, bs_list) -> None:
        # No CUDA/XPU graph capture in the reference backend.
        self.max_graph_bs = max(bs_list) if bs_list else 0

    def prepare_for_capture(self, batch) -> None:
        return None

    def prepare_for_replay(self, batch) -> None:
        return None

    def forward(self, q, k, v, layer_id: int, batch, attn_spec: AttentionSpec | None = None) -> torch.Tensor:
        # q / k / v: [num_tokens, heads, head_dim] (token-ordered across the
        # batch). For decode, one token per request, so num_tokens == bs.
        # The KV pool holds each request's *written* history (prompt during
        # prefill, full history during decode); the new tokens' K/V were just
        # written into it, so reading ``written`` rows covers the full context.
        #
        # The model lays the per-token K/V out **head-major**: q / k / v are
        # [heads, tokens, head_dim]. The pool is token-major [tokens, kv, D],
        # so to attend we transpose the read to head-major [kv, tokens, D] and
        # expand the KV head dim to the full query head count (GQA) by
        # repeating along dim 0. That makes
        #   scores = qh [H, qlen, D] @ k_all.t [D, T]  -> [H, qlen, T]
        # (heads on dim 0, query tokens dim 1, key tokens dim 2), so the
        # causal mask is [1, qlen, T] -- broadcast over the head dim.
        ctx = _get_ctx()
        kv_cache = ctx.kv_cache
        # q / k / v are head-major [heads, tokens, head_dim], so the head count
        # is dim 0 (dim 1 is the token dim -- do NOT read heads from there).
        num_heads = q.shape[0]
        num_kv = k.shape[0]
        repeat = num_heads // num_kv
        scale = 1.0 / (q.shape[-1] ** 0.5)

        out = torch.empty_like(q)
        # Walk the batch in token order (matches the model's per-request slicing
        # and the flattened out_loc / positions tensors).
        token_idx = 0
        for i, req in enumerate(batch.reqs):
            if batch.is_decode:
                ext = 1
                written = req.device_len  # full history (new token already in pool)
            else:
                ext = req.extend_len
                written = ext  # only this request's prompt tokens are in the pool
            # Read the request's KV history from the pool (token-major
            # [written, kv, D]), then transpose to head-major [kv, written, D].
            k_tok, v_tok = kv_cache.read_kv(req.table_idx, torch.arange(written, device=q.device))
            k_all = k_tok.transpose(0, 1).contiguous()  # [kv, written, D]
            v_all = v_tok.transpose(0, 1).contiguous()  # [kv, written, D]
            if repeat != 1:
                # GQA: repeat each KV head so there is one key/value per query
                # head. repeat along dim 0 -> [num_heads, written, D].
                k_all = k_all.repeat(repeat, 1, 1)
                v_all = v_all.repeat(repeat, 1, 1)
            # This request's new queries (token slice). q is head-major
            # [heads, tokens, D], so slice the token (middle) dim.
            qh = q[:, token_idx : token_idx + ext, :]  # [heads, qlen, D]
            # scores = qh @ k_all^T -> [heads, qlen, written]: heads on dim 0,
            # query-token on dim 1, key-token on dim 2. The causal mask must
            # therefore be [1, qlen, written] (broadcast over the head dim),
            # comparing query positions (dim 1) against key positions (dim 2).
            q_pos = batch.positions[token_idx : token_idx + ext]
            key_pos = torch.arange(written, device=q.device)
            allowed = q_pos[None, :, None] >= key_pos[None, None, :]  # [1, qlen, written]
            scores = torch.matmul(qh, k_all.transpose(-1, -2)) * scale
            scores = torch.where(allowed, scores, torch.full_like(scores, float("-inf")))
            o = torch.matmul(torch.softmax(scores, dim=-1), v_all)  # [heads, qlen, D]
            out[:, token_idx : token_idx + ext, :] = o
            token_idx += ext
        return out


def _xpu_available() -> bool:
    try:
        import torch as _t

        return bool(getattr(_t, "xpu", None) and _t.xpu.is_available())
    except Exception:
        return False


def _get_ctx():
    from freetoken.core import get_global_ctx

    return get_global_ctx()


def _expand_kv(x: torch.Tensor, repeat: int) -> torch.Tensor:
    if repeat == 1:
        return x
    # [L, kv, D] -> [L, kv*repeat, D] (each KV head repeated to the Q head count).
    return x.repeat_interleave(repeat, dim=1)


__all__ = ["TritonAttentionBackend"]
