"""Qwen3.8-Flash-Next (``qwen4_exp``) -- Intel Arc Pro B70 port. Epic #198.

Upstream NVIDIA path: python/freetoken/models/qwen4_exp/
Fill in: GitHub issues `models-qwen4-hc` (#206), `models-qwen4-ple` (#207),
`models-qwen4-qsa` (#208), `models-qwen4-e2e` (#209) -- one package, built
up incrementally across all four (this port's established one-``__init__.py``
-per-model convention, unlike upstream's multi-file-per-model layout).

This file currently ships #206 only: the hyper-connection (gated residual)
primitive below. ``parse_config``/``iter_weights``/``Qwen4ExpForCausalLM``
stay stubs (``unimplemented``) until #209 wires the full model -- the
primitive is unit-tested standalone in the meantime (see
``tests/test_models_qwen4_hc.py``).

## Hyper-connections (``GatedResidual`` / ``GroupedPlusOneRMSNorm``, #206)

Upstream NVIDIA path: python/freetoken/models/qwen4_exp/hc.py

Every hyper-connection decoder layer reads and writes ``hc_count`` PARALLEL
residual streams packed as ``R [..., hc_count*hidden]`` (stream outer,
hidden inner), instead of the single residual stream every other decoder
layer in this port assumes:

    x, s = hc.mix(R)          # R [..., hc_count*hidden] -> x [..., hidden], s [..., hc_count] or None
    y    = block(x)            # attention / GDN / MoE, plain [..., hidden] -> [..., hidden]
    R    = hc.combine(R, y, s)

Formulas (upstream ``hc.py``'s own docstring, HF
``Qwen4ExpTextGatedResidual``)::

    Rn      = groupRMSNorm(R) * (1 + hc_norm.weight)        # per hidden-size stream, fp32 stats
    lora, s = input_mix_weight_down_block_inject(Rn)         # merged GEMM: [lowrank | hc_count | pad]
    gate    = input_mix_weight_up(silu(lora / hc_count))
    x       = mean_i(sigmoid(gate_i) * Rn_i)
    R'_i    = R_i + 2*sigmoid(s_i / hc_count) * y

``s`` is the RAW inject logit slice (pre ``2*sigmoid``) -- ``combine``
applies the activation. ``use_combine=False`` is the top-level mixer: it
owns the unmerged ``input_mix_weight_down``, returns ``s = None`` and has
no ``combine``.

This port ships the pure-torch reference path only (this session's
established "reference correctness first" discipline for every new
mechanism -- GDN, MLA, DSA before it): upstream's vendored Triton/CUDA
kernels (``kernel/triton/hc.py``) are NOT ported. Unlike upstream's
``BaseOP``/``LinearReplicated`` class hierarchy, this port follows the
established per-model convention (see ``glm_moe_dsa``/``deepseek_v4``):
plain ``nn.Module`` + ``nn.Linear``/``nn.Parameter``.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def grouped_plus_one_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, num_groups: int
) -> torch.Tensor:
    """RMSNorm each of ``num_groups`` equal slices of the last dim on its own fp32 statistic, then scale by (1+w)."""
    xf = x.float().unflatten(-1, (num_groups, -1))
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf.flatten(-2) * (1.0 + weight.float())).to(x.dtype)


class GroupedPlusOneRMSNorm(nn.Module):
    """Per-stream RMSNorm of an ``[..., num_groups*group]`` tensor with one weight element per feature.

    HF ``Qwen4ExpTextRMSNorm(dim, group_size)``. The checkpoint weight is
    zero-centered and loaded RAW: ``(1+w)`` is applied at runtime in fp32,
    never folded into the stored weight.
    """

    def __init__(self, size: int, eps: float, num_groups: int, *, dtype=None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(size, dtype=dtype))
        self.eps = eps
        self.num_groups = num_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return grouped_plus_one_rms_norm(x, self.weight, self.eps, self.num_groups)


class GatedResidual(nn.Module):
    """One hyper-connection block: ``mix`` reads the residual streams, ``combine`` writes a block output back.

    Weight keys (checkpoint names, prefix stripped): ``hc_norm.weight``,
    ``input_mix_weight_down_block_inject.weight`` (loader: concat of
    ``input_mix_weight_down`` ``[lowrank, hc*hidden]``, ``block_inject_weight``
    ``[hc_count, hc*hidden]`` and ``pad`` zero rows), ``input_mix_weight_up.weight``.
    """

    def __init__(
        self,
        hidden_size: int,
        hc_count: int,
        lowrank: int,
        eps: float,
        *,
        use_combine: bool = True,
        dtype=None,
    ) -> None:
        super().__init__()
        self.hc_count = hc_count
        self.hidden_size = hidden_size
        self.lowrank = lowrank
        self.use_combine = use_combine
        width = hc_count * hidden_size
        self.hc_norm = GroupedPlusOneRMSNorm(width, eps, hc_count, dtype=dtype)
        if use_combine:
            # 16-row alignment for the merged skinny GEMM (vLLM hyperconnection.py:98)
            self.pad_size = (-(lowrank + hc_count)) % 16
            self.input_mix_weight_down_block_inject = nn.Linear(
                width, lowrank + hc_count + self.pad_size, bias=False, dtype=dtype
            )
        else:
            self.pad_size = 0
            self.input_mix_weight_down = nn.Linear(width, lowrank, bias=False, dtype=dtype)
        self.input_mix_weight_up = nn.Linear(lowrank, width, bias=False, dtype=dtype)

    def _down(self, rn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Run the down GEMM and split off the raw inject logits; the pad columns are dropped."""
        if not self.use_combine:
            return self.input_mix_weight_down(rn), None
        down = self.input_mix_weight_down_block_inject(rn)
        return down[..., : self.lowrank], down[..., self.lowrank : self.lowrank + self.hc_count]

    def mix(self, R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Return the block input ``x [..., hidden]`` and the inject logits ``s [..., hc_count]`` (None if no combine)."""
        rn = grouped_plus_one_rms_norm(R, self.hc_norm.weight, self.hc_norm.eps, self.hc_count)
        lora, s = self._down(rn)
        lora = F.silu(lora.float() / self.hc_count)
        gate = self.input_mix_weight_up(lora.to(R.dtype))
        mixed = torch.sigmoid(gate.float()).unflatten(-1, (self.hc_count, self.hidden_size))
        mixed = mixed * rn.float().unflatten(-1, (self.hc_count, self.hidden_size))
        return mixed.mean(-2).to(R.dtype), s

    def combine(self, R: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Inject the block output ``y [..., hidden]`` back into every stream of ``R``."""
        inject = 2.0 * torch.sigmoid(s.float() / self.hc_count)
        out = R.float().unflatten(-1, (self.hc_count, self.hidden_size))
        out = out + y.float().unsqueeze(-2) * inject.unsqueeze(-1)
        return out.flatten(-2).to(R.dtype)


# --------------------------------------------------------------------------- #
# QSA: block-sparse indexer attention (#208, 12 of 48 layers)
# --------------------------------------------------------------------------- #
#
# Confirmed a genuinely different mechanism from DSA (#191/#22), grounded
# directly against upstream's real `models/qwen4_exp/attention.py` and
# `kernel/triton/qsa/*.py`, not assumed to be "DSA with different names":
#
# * DSA (glm_moe_dsa/deepseek_v4) scores and top-k-selects individual
#   PAST TOKENS directly, one score per historical key.
# * QSA compresses the key history into non-overlapping BLOCKS of
#   `index_ratio` consecutive raw keys first (fp32 mean pool -- a block is
#   only formed once it has `index_ratio` members, so the trailing partial
#   block is never a top-k candidate), scores and top-k-selects BLOCKS, then
#   expands the winning blocks back to token positions for the real
#   attention. Per HF `Qwen4ExpTextQSAIndexer` (real docstring, upstream
#   `attention.py`):
#
#       q_h    = rope64(rmsnorm(q_h) * (1 + q_norm_weight), pos = query position)
#       kbar_b = rope64(rmsnorm(mean_fp32(k[4b:4b+4])) * (1 + k_norm_weight), pos = 4b)
#       s_b    = sum_h relu(<q_h, kbar_b>) / sqrt(index_head_dim)
#
#   (`rmsnorm(x) * (1+w)` is this module's own `grouped_plus_one_rms_norm`
#   with `num_groups=1` -- per-vector, not per-hyper-connection-stream --
#   the same zero-centered-weight convention, confirmed a real shared HF
#   idiom rather than coincidence.)
# * The trailing incomplete block (< `index_ratio` tokens, not yet
#   compressible) is always attended directly, never top-k'd -- upstream's
#   own `TorchDenseQSAReference` docstring is the confirmation: "QSA is
#   exactly dense while a request sees at most `index_budget + index_ratio
#   - 1` tokens", i.e. `index_topk_blocks * index_ratio` selected-block
#   tokens plus up to `index_ratio - 1` trailing raw tokens. That equivalence
#   is this module's own test oracle for the short-sequence case below.
#
# This port ships the pure-torch reference mechanism only (compress / score
# / top-k / expand / masked-attend as plain functions), not the vendored
# Triton kernels (`kernel/triton/qsa/{compress,score,topk,expand,attend}.py`)
# -- matching this port's established "reference correctness first"
# discipline. Full engine wiring (checkpoint weights, KV-cache-backed
# incremental compression across decode steps, the `Qwen4ExpAttention`
# gated-GQA wrapper) is #209's job; the functions here operate on a full,
# already-materialized sequence (prefill-shaped), proven against a
# brute-force reference.


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int) -> torch.Tensor:
    """Half-split (NeoX) RoPE over the first ``rotary_dim`` of the last axis; the rest passes through.
    ``cos``/``sin`` broadcast against ``x``'s ``[..., rotary_dim]`` leading slice."""
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x_rot = (x_rot.float() * cos + _rotate_half(x_rot.float()) * sin).to(x.dtype)
    return torch.cat([x_rot, x_pass], dim=-1)


def qsa_rope_cos_sin(positions: torch.Tensor, rotary_dim: int, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """``cos``/``sin [T, rotary_dim]`` (``cat(freqs, freqs)``, matching ``apply_partial_rope``'s rotate_half split)."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=positions.device) / rotary_dim)
    )
    freqs = torch.outer(positions.to(torch.float32), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def qsa_compress_keys(raw_keys: torch.Tensor, index_ratio: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool disjoint groups of ``index_ratio`` consecutive positions of ``raw_keys
    [T, kv_heads, D]`` (pre-norm, pre-rope) in fp32. A group is only formed once it has
    ``index_ratio`` members -- the trailing partial group is dropped. Returns
    ``(pooled [num_blocks, kv_heads, D], block_start_positions [num_blocks])``."""
    T = raw_keys.shape[0]
    num_blocks = T // index_ratio
    if num_blocks == 0:
        return raw_keys.new_zeros((0,) + raw_keys.shape[1:]), torch.zeros(0, dtype=torch.long, device=raw_keys.device)
    trimmed = raw_keys[: num_blocks * index_ratio].float()
    pooled = trimmed.unflatten(0, (num_blocks, index_ratio)).mean(dim=1)
    block_start = torch.arange(num_blocks, device=raw_keys.device) * index_ratio
    return pooled, block_start


def qsa_score(
    q: torch.Tensor,
    kbar: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    eps: float,
    query_positions: torch.Tensor,
    block_start_positions: torch.Tensor,
    rotary_dim: int,
    theta: float,
) -> torch.Tensor:
    """``q [T, n_heads, D]`` and ``kbar [num_blocks, kv_heads, D]`` raw (pre-norm, pre-rope).
    Returns block scores ``[T, num_blocks]`` -- ``sum_h relu(<q_h, kbar_b>) / sqrt(D)``."""
    n_heads, kv_heads, D = q.shape[1], kbar.shape[1], q.shape[-1]
    rep = n_heads // kv_heads
    q_n = grouped_plus_one_rms_norm(q, q_norm_weight, eps, 1)
    k_n = grouped_plus_one_rms_norm(kbar, k_norm_weight, eps, 1)
    cos_q, sin_q = qsa_rope_cos_sin(query_positions, rotary_dim, theta)
    cos_k, sin_k = qsa_rope_cos_sin(block_start_positions, rotary_dim, theta)
    q_n = apply_partial_rope(q_n, cos_q[:, None, :], sin_q[:, None, :], rotary_dim)
    k_n = apply_partial_rope(k_n, cos_k[:, None, :], sin_k[:, None, :], rotary_dim)
    k_n = k_n.repeat_interleave(rep, dim=1)  # [num_blocks, n_heads, D] -- GQA expand to match q
    scores = torch.einsum("thd,bhd->thb", q_n.float(), k_n.float())
    scores = F.relu(scores).sum(dim=1) / (D**0.5)
    return scores


def qsa_topk_blocks(
    scores: torch.Tensor, block_last_positions: torch.Tensor, query_positions: torch.Tensor, budget_blocks: int
) -> torch.Tensor:
    """Per-query top ``budget_blocks`` CAUSALLY VISIBLE blocks (a block is visible once its last
    token position is <= the query's own position). Returns ``[T, budget_blocks]`` int64 block
    indices, ``-1`` padded past however many blocks are actually visible."""
    T, num_blocks = scores.shape
    device = scores.device
    if num_blocks == 0 or budget_blocks == 0:
        return torch.full((T, budget_blocks), -1, dtype=torch.long, device=device)
    causal = block_last_positions[None, :] <= query_positions[:, None]
    masked = scores.masked_fill(~causal, float("-inf"))
    k_eff = min(budget_blocks, num_blocks)
    topk_vals, topk_idx = torch.topk(masked, k_eff, dim=-1)
    topk_idx = topk_idx.masked_fill(topk_vals == float("-inf"), -1)
    if k_eff < budget_blocks:
        pad = torch.full((T, budget_blocks - k_eff), -1, dtype=torch.long, device=device)
        topk_idx = torch.cat([topk_idx, pad], dim=-1)
    return topk_idx


def qsa_expand_block_mask(topk_idx: torch.Tensor, index_ratio: int, seq_len: int) -> torch.Tensor:
    """Expand ``topk_idx [T, budget_blocks]`` block selection (``-1`` = pad) to a boolean token
    mask ``[T, seq_len]``: token ``j`` is set iff it belongs to a selected block."""
    T, budget = topk_idx.shape
    device = topk_idx.device
    if seq_len == 0 or budget == 0:
        return torch.zeros(T, seq_len, dtype=torch.bool, device=device)
    valid = topk_idx >= 0
    starts = topk_idx.clamp(min=0) * index_ratio
    offsets = torch.arange(index_ratio, device=device)
    pos = starts.unsqueeze(-1) + offsets  # [T, budget, index_ratio]
    in_range = valid.unsqueeze(-1) & (pos < seq_len)
    pos = pos.clamp(max=max(seq_len - 1, 0)).reshape(T, -1)
    in_range = in_range.reshape(T, -1)
    grid = torch.arange(seq_len, device=device)
    hit = (pos.unsqueeze(-1) == grid.view(1, 1, -1)) & in_range.unsqueeze(-1)
    return hit.any(dim=1)


def qsa_sparse_attend(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attend_mask: torch.Tensor, sm_scale: float
) -> torch.Tensor:
    """``q [T, num_q_heads, D]``, ``k``/``v [S, num_kv_heads, D]`` (post main norm+rope, model
    dtype), ``attend_mask [T, S]`` bool (sparse selection already AND'd with causality). Returns
    ``[T, num_q_heads, D]``."""
    num_q, num_kv = q.shape[1], k.shape[1]
    rep = num_q // num_kv
    k_full = k.repeat_interleave(rep, dim=1)
    v_full = v.repeat_interleave(rep, dim=1)
    scores = torch.einsum("thd,shd->hts", q.float(), k_full.float()) * sm_scale
    scores = scores.masked_fill(~attend_mask[None, :, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hts,shd->thd", probs, v_full.float())
    return out.to(q.dtype)


def qsa_attend_mask(
    topk_idx: torch.Tensor, index_ratio: int, query_positions: torch.Tensor, num_complete_blocks: int, seq_len: int
) -> torch.Tensor:
    """The full attend mask for one QSA layer: selected-block tokens UNION the trailing
    incomplete-block tokens (always attended raw, see module note above), AND'd with causality."""
    device = topk_idx.device
    block_mask = qsa_expand_block_mask(topk_idx, index_ratio, seq_len)
    tail_start = num_complete_blocks * index_ratio
    positions = torch.arange(seq_len, device=device)
    tail_mask = positions >= tail_start
    causal = positions[None, :] <= query_positions[:, None]
    return (block_mask | tail_mask[None, :]) & causal


# --------------------------------------------------------------------------- #
# Not yet implemented: PLE (#207), full model wiring (#209).
# --------------------------------------------------------------------------- #

from freetoken._stub import unimplemented


def parse_config(*args, **kwargs):
    unimplemented("parse_config", "models-qwen4-e2e")


def iter_weights(*args, **kwargs):
    unimplemented("iter_weights", "models-qwen4-e2e")


class Qwen4ExpForCausalLM:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(self, *args, **kwargs):
        unimplemented("Qwen4ExpForCausalLM.forward", "models-qwen4-e2e")


__all__ = [
    "GatedResidual",
    "GroupedPlusOneRMSNorm",
    "grouped_plus_one_rms_norm",
    "apply_partial_rope",
    "qsa_rope_cos_sin",
    "qsa_compress_keys",
    "qsa_score",
    "qsa_topk_blocks",
    "qsa_expand_block_mask",
    "qsa_sparse_attend",
    "qsa_attend_mask",
    "parse_config",
    "iter_weights",
    "Qwen4ExpForCausalLM",
]
