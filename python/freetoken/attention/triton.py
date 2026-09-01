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

from .base import AttentionSpec, BaseAttnBackend, BaseAttnMetadata


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

    ``forward(q, k, v, layer_id, batch, attn_spec)`` takes ``q`` / ``k`` / ``v``
    head-major ``[heads, tokens, head_dim]`` (the *new* tokens this step) and
    returns ``[heads, tokens, head_dim]``. K/V for the whole sequence are read
    from the KV pool via the page table, so the buffer holds the full history
    and each step only appends the new tokens.

    Supports both ``AttnType.FULL`` (plain causal) and ``AttnType.SWA``: when
    ``attn_spec.sliding_window > 0`` a query attends only the most recent
    ``sliding_window`` keys (matching the SYCL kernel's branch-free mask).
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

    def forward(
        self,
        q,
        k,
        v,
        layer_id: int,
        batch,
        attn_spec: AttentionSpec | None = None,
        table_idx: int | None = None,
    ) -> torch.Tensor:
        # q / k / v: [heads, tokens, head_dim] (head-major), holding the *new*
        # tokens for the request(s) this step processes. The model runs each
        # request's layers on its own hidden slice, so when ``table_idx`` is
        # given (the per-request call pattern the model uses), q/k/v hold ONLY
        # that one request's tokens -- attend them against that request's KV
        # history from the pool. When ``table_idx`` is None (a caller that hands
        # the whole batch's tokens in one call), fall back to the original
        # walk-the-batch behavior.
        # q / k / v are head-major [heads, tokens, head_dim], so the head count
        # is dim 0 (dim 1 is the token dim -- do NOT read heads from there).
        num_heads = q.shape[0]
        num_kv = k.shape[0]
        repeat = num_heads // num_kv
        scale = 1.0 / (q.shape[-1] ** 0.5)

        out = torch.empty_like(q)

        # Locate the request this call is for. With ``table_idx`` (the model's
        # per-request call pattern) q/k/v hold ONLY that request's tokens, so we
        # attend the whole q against that request's history in one shot. Without
        # it (a caller passing the whole batch's tokens) we walk the batch by
        # global token offset -- the original behavior.
        # SWA: the layer's sliding-window size (0 / None => plain full causal).
        # Read once here and passed into _attend_one, so the mask in both call paths
        # (per-request and whole-batch) is identical.
        window = int(attn_spec.sliding_window) if (attn_spec is not None and attn_spec.sliding_window) else 0

        if table_idx is not None:
            req = next((r for r in batch.reqs if r.table_idx == table_idx), batch.reqs[0])
            ext = q.shape[1]
            # Phase comes from the scheduler's per-step flag ``batch.phase``
            # ("prefill" for any step that extends the prompt -- first chunk or a
            # chunked continuation -- "decode" otherwise), which is uniform within
            # a step: the scheduler emits a prefill step or a decode step, never a
            # mix, so one flag sizes every request's slice this step. This matches
            # the engine, which appends/samples and keys its token bookkeeping off
            # batch.phase. (NOT req.can_decode -- that is a per-request token-BUDGET
            # flag (remain_len > 0): it is False for a finished decode req, and it
            # is the chunked-prefill "do not sample this chunk" signal, NOT the
            # attention phase. The old shape checks (device_len != / > len(input_ids))
            # misfire under chunked prefill, where a continuation has device_len >
            # len(input_ids) and a full prefill has device_len == len(input_ids).)
            is_decode = batch.phase == "decode"
            # Decode steps read the full KV history (prompt + every generated
            # token). A prefill step reads the prompt prefix already resident
            # in the pool plus the tokens this chunk carries: for the first
            # chunk (cached_len 0) that is the whole prompt; for a
            # continuation the pool holds [0, device_len) and ext == the
            # chunk, so cached_len + ext == device_len == the full attended
            # prefix. The old "written = device_len" under-counted a
            # continuation (device_len was the pre-bump), so the kernel read pool slots that
            # had not been written yet (torch.empty garbage, which differs per
            # process -- the original nondeterminism).
            written = req.device_len if is_decode else req.cached_len + ext
            q_pos = self._request_positions(batch, table_idx, ext)
            out[:] = self._attend_one(req, q, q_pos, written, repeat, scale, window, layer_id)
            return out

        # Whole-batch call (table_idx is None): q/k/v span all requests, so walk
        # the batch in token order (matches the flattened out_loc / positions).
        # Each request's phase is decided per-request (a batch can mix phases).
        token_idx = 0
        for req in batch.reqs:
            # Uniform phase from the scheduler's per-step flag (see the
            # per-request call above): one flag sizes every request's slice.
            is_decode = batch.phase == "decode"
            # A decode step always contributes exactly one new token per request,
            # independent of req.extend_len (which is device_len - cached_len and
            # can be larger when a request was framed/overridden with a stale
            # cached_len -- e.g. a test that overrides device_len but leaves
            # cached_len at 0). Pref pattern of upstream: decode -> 1 new token,
            # prefill -> the whole chunk (req.extend_len).
            ext = 1 if is_decode else req.extend_len
            # Decode reads the full history; prefill reads the resident prompt
            # prefix + this chunk (see the per-request path above).
            written = req.device_len if is_decode else req.cached_len + ext
            qh = q[:, token_idx : token_idx + ext, :]
            q_pos = batch.positions[token_idx : token_idx + ext]
            out[:, token_idx : token_idx + ext, :] = self._attend_one(req, qh, q_pos, written, repeat, scale, window, layer_id)
            token_idx += ext
        return out

    def _attend_one(self, req, qh, q_pos, written, repeat, scale, window: int = 0, layer_id: int = 0) -> torch.Tensor:
        """Attend a block of query rows against one request's KV history."""
        ctx = _get_ctx()
        kv_cache = ctx.kv_cache
        dev = qh.device
        k_tok, v_tok = kv_cache.read_kv(req.table_idx, torch.arange(written, device=dev), layer_id)
        k_all = k_tok.transpose(0, 1).contiguous()  # [kv, written, D]
        v_all = v_tok.transpose(0, 1).contiguous()  # [kv, written, D]
        if repeat != 1:
            # GQA: query head h attends KV head h // repeat. That is an
            # *interleave* of the KV heads ([0,0,1,1] for 2 kv heads, repeat 2),
            # NOT a tile ([0,1,0,1]) -- `.repeat(repeat, ...)` tiles the whole
            # axis and would map query heads to the wrong KV head whenever there
            # are multiple KV heads (head 1 would read KV head 1 instead of 0).
            # `repeat_interleave` gives the correct h // repeat mapping.
            k_all = k_all.repeat_interleave(repeat, dim=0)  # [num_heads, written, D]
            v_all = v_all.repeat_interleave(repeat, dim=0)
        # scores = qh @ k_all^T -> [heads, qlen, written]. The mask is
        # [1, qlen, written] (broadcast over the head dim), comparing query
        # positions (dim 1) against key positions (dim 2). Causal: a query at
        # position q attends keys at position <= q. SWA (window > 0) additionally
        # restricts to the most recent `window` keys: q - keypos < window. This
        # matches the SYCL kernel's branch-free mask exactly.
        key_pos = torch.arange(written, device=dev)
        allowed = q_pos[None, :, None] >= key_pos[None, None, :]
        if window > 0:
            allowed = allowed & ((q_pos[None, :, None] - key_pos[None, None, :]) < window)
        scores = torch.matmul(qh, k_all.transpose(-1, -2)) * scale
        scores = torch.where(allowed, scores, torch.full_like(scores, float("-inf")))
        return torch.matmul(torch.softmax(scores, dim=-1), v_all)  # [heads, qlen, D]

    def _request_positions(self, batch, table_idx: int, ext: int) -> torch.Tensor:
        """This request's new-token positions (for the causal mask).

        ``batch.positions`` is the whole-batch flatten in request order; the
        request identified by ``table_idx`` occupies a contiguous run of ``ext``
        positions starting at its global token offset (cumulative extend lengths
        of the preceding requests).
        """
        offset = 0
        for r in batch.reqs:
            if r.table_idx == table_idx:
                break
            # A decoding request contributes one new token this step; a
            # prefilling one (first chunk or chunked continuation) contributes
            # its extend length. The phase is the scheduler's uniform per-step
            # flag (not a per-request device_len shape test, which misfires for
            # a full prefill / chunked continuation).
            offset += 1 if batch.phase == "decode" else r.extend_len
        return batch.positions[offset : offset + ext]


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
