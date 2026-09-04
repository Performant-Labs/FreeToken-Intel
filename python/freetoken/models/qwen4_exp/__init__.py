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
    topk_idx: torch.Tensor, index_ratio: int, query_positions: torch.Tensor, seq_len: int
) -> torch.Tensor:
    """The full attend mask for one QSA layer: selected-block tokens UNION each query's own
    OPEN GROUP (upstream ``qsa_sparse.py`` module docstring, step 5: "expand them to token
    indices plus the causal tail of the open group") -- the raw tokens of the still-forming
    block a query's own position falls in are always attended, since that block has no
    compressed representative yet to be top-k-selected. This is PER-QUERY (each row's open
    group is ``[(pos // index_ratio) * index_ratio, pos]``), not a single global trailing
    region for the whole sequence -- a query early in a long, block-aligned sequence has an
    open group near its own position, not near the sequence's end."""
    device = topk_idx.device
    block_mask = qsa_expand_block_mask(topk_idx, index_ratio, seq_len)
    positions = torch.arange(seq_len, device=device)
    group_start = (query_positions // index_ratio) * index_ratio
    open_mask = (positions[None, :] >= group_start[:, None]) & (positions[None, :] <= query_positions[:, None])
    causal = positions[None, :] <= query_positions[:, None]
    return (block_mask | open_mask) & causal


# --------------------------------------------------------------------------- #
# PLE: hashed n-gram Per-Layer Embedding (#207, active on `ple_layer_ids`)
# --------------------------------------------------------------------------- #
#
# Per token, grounded against upstream's real `models/qwen4_exp/ple.py`
# module docstring (HF `Qwen4ExpTextNGramEmbedding`/`Qwen4ExpTextPLELayer`):
#
#     E = table[hash(ngram)]                        # heads_per_ngram * (ngram_size-1) heads -> ple_embed_dim
#     K = norm_key(key_proj(E)).view(hc, hidden)     # V = value_proj(E) [hidden]
#     Q = norm_query(R).view(hc, hidden)
#     u = <K_i, Q_i> / sqrt(hidden)                  # per stream
#     U = sigmoid(sign(u) * sqrt(max(|u|, 1e-6))) * V
#     D = U + silu(conv1d(norm_conv(U)))             # depthwise, kernel size, dilation ngram_size
#     R += D                                         # before the attention hyper-connection mix
#
# N-gram hashing (`ple_ngram_row_ids`): for each ngram order 2..ngram_size, XOR
# together `layer_multipliers[i] * token[t-i]` for i in 0..order-1 (splitmix64-derived
# per-layer multipliers), mod each head's own prime vocab size, offset into that head's
# global table range. The hash window resets at `ngram_boundary_token_id` (eos): a shifted
# token more than `in_segment` positions from the last boundary falls back to the boundary
# id itself, matching upstream's `_shift_ignore_eos` -- confirmed NOT a silent no-op by the
# regression test below (a boundary token mid-sequence measurably changes later rows' hashes
# versus the same token stream with no boundary).
#
# Table backend: upstream's real table is a 47.7 GiB FP8 store (`PinnedUVATable`, kept
# resident in pinned HOST RAM, gathered over UVA). This box has 29 GiB total RAM -- full
# residency is categorically impossible here, so `PleDiskTable` below is a REAL requirement
# of this port, not an optional upstream optimization: it reads exactly the requested rows
# out of a safetensors shard file via `safe_open`'s lazy slicing (mmap'd, never materializes
# the whole tensor), mirroring upstream's own `ple_disk.py` role (`--ple-backend disk`) minus
# its C++ io_uring/pinned-staging machinery -- reference correctness first, same discipline
# as HC/QSA above. `PleInMemoryTable` is the small-table oracle `PleDiskTable` is diffed
# against, matching upstream's own `GpuResidentTable` role.
#
# This port ships the pure-torch/plain-Python reference path only; the vendored Triton
# gather/conv kernels are not ported. Full engine wiring (checkpoint loading, the
# `linear_state_pool` conv/n-gram-context slot states, ragged multi-request batching,
# decode-step incremental hashing) is #209's job -- the functions/classes here operate on
# one full, already-materialized token sequence (prefill-shaped), proven against
# hand-computed references.


def derive_ngram_hash_constants(
    *,
    vocab_size: int,
    ngram_size: int,
    num_ngram_heads: int,
    ngram_vocab_size_base: int,
    ple_layer_index: int,
    seed: int = 1234,
) -> Tuple[list, list, list]:
    """Recompute (multipliers, per-head vocab sizes, per-head offsets) the way HF derives
    them at init (checkpoint ships them as int64 tensors; this is the dummy-weight path and
    the oracle a loader test can check real checkpoint values against). Verbatim port of
    upstream's own derivation (splitmix64 mix + nth-prime-after search)."""
    mask64 = (1 << 64) - 1
    gamma = 0x9E3779B97F4A7C15
    m1 = 0xBF58476D1CE4E5B9
    m2 = 0x94D049BB133111EB
    layer_prime = 10007

    def splitmix64(value: int) -> int:
        value = (value + gamma) & mask64
        value = ((value ^ (value >> 30)) * m1) & mask64
        value = ((value ^ (value >> 27)) * m2) & mask64
        return (value ^ (value >> 31)) & mask64

    def is_prime(value: int) -> bool:
        if value < 2:
            return False
        if value % 2 == 0:
            return value == 2
        for divisor in range(3, int(value**0.5) + 1, 2):
            if value % divisor == 0:
                return False
        return True

    def nth_prime_after(start: int, count: int) -> int:
        prime = start
        for _ in range(count):
            prime += 1
            while not is_prime(prime):
                prime += 1
        return prime

    half_bound = max(1, ((1 << 63) - 1) // max(vocab_size, 1) // 2)
    base_seed = seed + layer_prime * ple_layer_index
    multipliers = [
        2 * (splitmix64((base_seed + gamma * (i + 1)) & mask64) % half_bound) + 1 for i in range(ngram_size)
    ]
    sizes: list = []
    offsets: list = []
    total = 0
    for head in range(num_ngram_heads):
        global_head = ple_layer_index * num_ngram_heads + head
        size = nth_prime_after(ngram_vocab_size_base - 1, global_head + 1)
        sizes.append(size)
        offsets.append(total)
        total += size
    return multipliers, sizes, offsets


def ple_ngram_row_ids(
    token_ids: torch.Tensor,
    boundary_token_id: int,
    ngram_size: int,
    heads_per_ngram: int,
    multipliers: torch.Tensor,
    head_vocab_sizes: torch.Tensor,
    head_offsets: torch.Tensor,
) -> torch.Tensor:
    """Global table row per (token, hash head): ``token_ids [T] -> [T, num_ngram_heads]`` int64.

    ``token_ids`` is the flat sequence to hash over (a caller wanting left-context from a
    prior forward prepends it and slices the result). The hash window at position ``t`` never
    crosses a ``boundary_token_id`` occurrence at or before ``t``: a shifted tap that would
    reach past the most recent boundary reads the boundary id itself instead (matches
    upstream ``_shift_ignore_eos``, verified by the regression test below)."""
    device = token_ids.device
    T = token_ids.shape[0]
    pos = torch.arange(T, device=device)
    eos_pos = torch.where(token_ids == boundary_token_id, pos, torch.full_like(pos, -1))
    prev_eos = torch.cummax(eos_pos, dim=0).values
    prev_eos = torch.cat([eos_pos.new_full((1,), -1), prev_eos[:-1]])
    in_segment = pos - prev_eos - 1

    shifted = [token_ids]
    for shift in range(1, ngram_size):
        src = (pos - shift).clamp(min=0)
        gathered = token_ids[src]
        valid = (pos - shift >= 0) & (in_segment >= shift)
        shifted.append(torch.where(valid, gathered, torch.full_like(token_ids, boundary_token_id)))

    blocks = []
    for ngram in range(2, ngram_size + 1):
        start = (ngram - 2) * heads_per_ngram
        end = start + heads_per_ngram
        mixed = shifted[0].to(torch.int64) * multipliers[0]
        for position in range(1, ngram):
            mixed = torch.bitwise_xor(mixed, shifted[position].to(torch.int64) * multipliers[position])
        head_ids = torch.remainder(mixed.unsqueeze(-1), head_vocab_sizes[start:end])
        blocks.append(head_ids + head_offsets[start:end])
    return torch.cat(blocks, dim=-1)


class PleInMemoryTable:
    """Row store held whole in device/host memory; the oracle ``PleDiskTable`` is diffed
    against (matches upstream ``GpuResidentTable``'s role). ``weight [num_rows, head_dim]``."""

    def __init__(self, weight: torch.Tensor, scale: float = 1.0) -> None:
        self.weight = weight
        self.scale = float(scale)
        self.num_rows, self.head_dim = weight.shape

    def lookup(self, row_ids: torch.Tensor) -> torch.Tensor:
        rows = self.weight.index_select(0, row_ids.reshape(-1)).float()
        if self.scale != 1.0:
            rows = rows * self.scale
        return rows.view(*row_ids.shape[:-1], -1).to(self.weight.dtype)


class PleDiskTable:
    """Row store kept ENTIRELY on disk: a safetensors shard file, opened once via
    ``safe_open`` (mmap'd) and read through ``get_slice(...)``'s lazy row indexing, which
    reads only the bytes of the requested rows -- the table itself is never materialized in
    RAM. This is the RAM-safe path the real 47.7 GiB n-gram store requires on a box that
    cannot hold it whole (this port's own established constraint, not upstream's -- see
    module note above)."""

    def __init__(self, path: str, tensor_name: str, scale: float = 1.0, dtype: torch.dtype = torch.float32) -> None:
        import safetensors

        self._file = safetensors.safe_open(path, framework="pt")
        self._slice = self._file.get_slice(tensor_name)
        self.scale = float(scale)
        self.dtype = dtype
        shape = self._slice.get_shape()
        self.num_rows, self.head_dim = shape[0], shape[1]

    def lookup(self, row_ids: torch.Tensor) -> torch.Tensor:
        flat = row_ids.reshape(-1)
        rows = torch.stack([self._slice[int(i) : int(i) + 1][0] for i in flat]).float()
        if self.scale != 1.0:
            rows = rows * self.scale
        return rows.view(*row_ids.shape[:-1], -1).to(self.dtype)


def ple_short_conv(x: torch.Tensor, state: torch.Tensor, weight: torch.Tensor, dilation: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """``silu(depthwise_conv1d([state | x]))`` over one sequence, advancing ``state``.

    ``x [T, width]``, ``state [width, state_len]`` (oldest first), ``weight [width, 1, kernel]``.
    Returns ``(out [T, width], new_state [width, state_len])``."""
    state_len = state.shape[-1]
    history = torch.cat([state.to(x.dtype), x.transpose(0, 1)], dim=-1).unsqueeze(0)
    out = F.conv1d(history, weight, groups=weight.shape[0], dilation=dilation).squeeze(0)
    new_state = history[0, :, -state_len:] if state_len > 0 else state
    return F.silu(out.transpose(0, 1)), new_state


class PLELayer(nn.Module):
    """One PLE block: hashed n-gram value gated by the residual streams, then a dilated
    depthwise conv. ``forward(R, token_ids, table, conv_state) -> (D, new_conv_state)``; the
    caller adds ``D`` to ``R`` before the attention hyper-connection mix (matches upstream
    ``PLELayer.forward``'s contract, minus the ragged multi-request batching and slot-state
    pool wiring #209 owns).

    Weight keys (checkpoint names, prefix stripped): ``key_proj.weight``
    ``[hc*hidden, ple_embed_dim]``, ``value_proj.weight`` ``[hidden, ple_embed_dim]``,
    ``norm_key/norm_query/norm_conv.weight`` ``[hc*hidden]`` (zero-centered, loaded RAW),
    ``conv1d.weight`` ``[hc*hidden, 1, kernel]``, plus the ``ple_embedding`` hash buffers
    (``layer_multipliers``, ``ngram_heads_vocab_sizes``, ``ngram_heads_offsets``).
    """

    def __init__(
        self,
        hidden_size: int,
        hc_count: int,
        ple_embed_dim: int,
        conv_kernel_size: int,
        dilation: int,
        eps: float,
        *,
        dtype=None,
    ) -> None:
        super().__init__()
        self.hc_count = hc_count
        self.hidden_size = hidden_size
        self.dilation = dilation
        width = hc_count * hidden_size
        self.key_proj = nn.Linear(ple_embed_dim, width, bias=False, dtype=dtype)
        self.value_proj = nn.Linear(ple_embed_dim, hidden_size, bias=False, dtype=dtype)
        self.norm_key = GroupedPlusOneRMSNorm(width, eps, hc_count, dtype=dtype)
        self.norm_query = GroupedPlusOneRMSNorm(width, eps, hc_count, dtype=dtype)
        self.norm_conv = GroupedPlusOneRMSNorm(width, eps, hc_count, dtype=dtype)
        self.conv1d = nn.Parameter(torch.zeros(width, 1, conv_kernel_size, dtype=dtype))

    def forward(
        self,
        R: torch.Tensor,
        embeddings: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``R [T, hc*hidden]``, ``embeddings [T, ple_embed_dim]`` (already gathered table
        rows), ``conv_state [hc*hidden, state_len]``. Returns ``(D [T, hc*hidden], new_conv_state)``."""
        key = self.norm_key(self.key_proj(embeddings))
        value = self.value_proj(embeddings)
        query = self.norm_query(R)
        shape = (-1, self.hc_count, self.hidden_size)
        gate = (key.view(shape).float() * query.view(shape).float()).sum(-1, keepdim=True) / (self.hidden_size**0.5)
        gate = torch.sigmoid(gate.sign() * gate.abs().clamp_min(1e-6).sqrt())
        gated = (gate * value.unsqueeze(-2).float()).flatten(-2).to(R.dtype)
        x = self.norm_conv(gated)
        conv_out, new_state = ple_short_conv(x, conv_state, self.conv1d, self.dilation)
        return gated + conv_out, new_state


# --------------------------------------------------------------------------- #
# Full model wiring (#209): config parsing, the decoder layer combining
# hyper-connections + GDN/QSA attention + PLE (on its own layer subset) +
# MoE, and the causal-LM wrapper.
#
# Router shape confirmed from upstream's own moe.py: `Qwen4ExpMoE` subclasses
# `Qwen3_5MoE` (this port's own qwen3_5_moe MoE block) -- a plain softmax-topk
# router + an always-on sigmoid-gated shared expert, NOT the DeepSeek-V3-style
# grouped/sigmoid/bias-corrected router deepseek_v4/glm_moe_dsa/glm4_moe use.
# This module ships its own small self-contained MoE block with that same
# router math (in-VRAM only, mirroring _GlmMoeDsaMoE's simple style) rather
# than reusing qwen3_5_moe's own `_Qwen35MoE` directly -- that class carries
# offload/cpu/hybrid backend branches this model does not implement, and
# duplicating just the needed math avoids importing that complexity.
#
# GDN (linear-attention layers) reuses this port's own PROVEN GDN block
# (`_GatedDeltaNet` from `qwen3_5_moe`, #170/#172) directly -- same math,
# just a different config source for the head-count/kernel-size fields.
#
# The decoder-layer forward CONTRACT this port's engine uses elsewhere
# (`layer(hidden_states, positions, table_idx, ctx, batch, ...)`, one
# request's token slice at a time -- see glm_moe_dsa/qwen3_5_moe) is
# preserved; only the mid-layer residual shape changes (hyper-connections'
# ``R [T, hc_count*hidden]`` instead of a single ``[T, hidden]`` stream),
# entirely internal to this package -- no other model's decoder-layer loop
# is touched.
# --------------------------------------------------------------------------- #

# qwen3_5_moe's forward-side classes are declared torch-free (plain, baseless)
# at module scope and only rebound to real nn.Module subclasses inside its own
# `_ensure_torch()` (normally triggered by building a Qwen3_5MoEForCausalLM,
# which this module never does) -- call it ourselves before grabbing
# `_GatedDeltaNet`/`_LinearStatePool`, or they stay the pre-rebind plain
# classes whose bodies reference an unbound `nn`/`torch`/`F` module global.
import freetoken.models.qwen3_5_moe as _qwen35_mod

_qwen35_mod._ensure_torch()
_GatedDeltaNet = _qwen35_mod._GatedDeltaNet
_LinearStatePool = _qwen35_mod._LinearStatePool
from freetoken.models.weight import iter_safetensors
from freetoken.utils import cached_load_hf_config


def _raw_checkpoint_json(model_path: str) -> dict:
    import json
    import os

    path = os.path.join(model_path, "config.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _layer_types(num_layers: int, explicit, interval: int) -> list:
    if explicit is not None:
        return list(explicit)
    return ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(num_layers)]


def parse_config(hf_config, model_path: str | None = None, **_kwargs):
    """Build a :class:`ModelConfig` for Qwen3.8-Flash-Next.

    No real public checkpoint exists for this architecture (confirmed: not
    in upstream's own docs/scripts, and `transformers` 5.15.1 has no
    registered ``qwen4_exp``/``qwen4next`` config class the way it
    surprisingly did for DeepSeek-V4/GLM-5.2's config classes) -- this
    parses the checkpoint's own config.json using upstream's real field
    names (``config.py``/``args.py``), the same ``file_raw`` defensive
    pattern established for deepseek_v4/glm_moe_dsa (this session found
    real HF config classes leaking non-None defaults through both
    ``getattr`` and ``to_dict()``; reading the checkpoint's own file avoids
    that class of bug even though there is no installed HF class here to
    leak from).
    """
    from freetoken.models.config import ModelConfig

    raw = hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(hf_config)
    file_raw = _raw_checkpoint_json(model_path) if model_path else raw

    def field(name, default=None):
        val = getattr(hf_config, name, None)
        return val if val is not None else raw.get(name, default)

    num_layers = int(field("num_hidden_layers"))
    hidden_size = int(field("hidden_size"))
    num_heads = int(field("num_attention_heads"))
    num_kv_heads = int(field("num_key_value_heads", num_heads))
    head_dim = int(field("head_dim", hidden_size // num_heads))
    rope_theta = float(field("rope_theta", 10000.0))

    layer_types = _layer_types(
        num_layers, file_raw.get("layer_types"), int(field("full_attention_interval", 4))
    )

    cfg = ModelConfig(
        architectures=["Qwen4ExpForCausalLM"],
        hidden_size=hidden_size,
        vocab_size=int(field("vocab_size")),
        num_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        num_experts=int(field("num_experts", 0) or 0),
        num_experts_per_tok=int(field("num_experts_per_tok", 0) or 0),
        moe_intermediate_size=int(field("moe_intermediate_size", 0) or 0),
        is_moe=bool(field("num_experts", 0)),
        tie_word_embeddings=bool(field("tie_word_embeddings", False)),
        rope_theta=rope_theta,
        dtype=field("torch_dtype") or field("dtype"),
    )
    cfg.attrs["rms_norm_eps"] = float(field("rms_norm_eps", 1e-5))
    cfg.attrs["rotary_dim"] = int(field("rotary_dim", head_dim))
    cfg.attrs["layer_types"] = layer_types
    cfg.attrs["shared_expert_intermediate_size"] = int(
        field("shared_expert_intermediate_size", cfg.moe_intermediate_size)
    )

    cfg.attrs["hc_count"] = int(field("hc_count"))
    cfg.attrs["hc_lowrank"] = int(field("hc_lowrank"))

    cfg.attrs["index_head_dim"] = int(field("indexer_head_dim"))
    cfg.attrs["index_n_heads"] = int(field("indexer_n_heads"))
    cfg.attrs["index_ratio"] = int(field("indexer_compress_ratio"))
    cfg.attrs["index_budget"] = int(field("indexer_budget"))

    # PLE (upstream stores ple_layer_ids one-indexed); the disk backend
    # (#207's own established constraint, ~47.7GiB real table) expects
    # exactly one PLE layer, so this port only supports that shape.
    ple_layer_ids_1idx = file_raw.get("ple_layer_ids") or []
    ple_layer_ids = [int(i) - 1 for i in ple_layer_ids_1idx]
    for lid in ple_layer_ids:
        if layer_types[lid] != "linear_attention":
            raise ValueError(f"PLE must sit on a linear_attention layer, got layer {lid}")
    cfg.attrs["ple_layer_ids"] = ple_layer_ids
    if ple_layer_ids:
        cfg.attrs["ple_embed_dim"] = int(field("ple_embed_dim"))
        cfg.attrs["ple_conv_kernel_size"] = int(field("ple_conv_kernel_size"))
        cfg.attrs["ngram_size"] = int(field("ngram_size"))
        cfg.attrs["heads_per_ngram"] = int(field("heads_per_ngram"))
        cfg.attrs["ngram_vocab_size_base"] = int(field("ngram_vocab_size_base"))
        eos = field("eos_token_id", 0)
        cfg.attrs["ngram_boundary_token_id"] = int(eos[0] if isinstance(eos, (list, tuple)) else eos)

    cfg.attrs["linear_num_key_heads"] = int(field("linear_num_key_heads"))
    cfg.attrs["linear_num_value_heads"] = int(field("linear_num_value_heads"))
    cfg.attrs["linear_key_head_dim"] = int(field("linear_key_head_dim"))
    cfg.attrs["linear_value_head_dim"] = int(field("linear_value_head_dim"))
    cfg.attrs["linear_conv_kernel_dim"] = int(field("linear_conv_kernel_dim"))

    return cfg


def iter_weights(model_path: str, device: torch.device, *, include_moe_experts: bool = True, include_non_moe: bool = True):
    """Yields the checkpoint's tensors on their destination device. Same
    dense/expert split convention as every other MoE model here (deepseek_v4,
    glm_moe_dsa) -- routed experts stay host-resident (in-VRAM MoE only, no
    offload backend for this architecture, see ``_Qwen4ExpMoE``)."""
    embed_tokens_weight = None
    saw_lm_head = False
    for name, tensor in iter_safetensors(model_path, device):
        is_expert = ".experts." in name
        if is_expert and not include_moe_experts:
            continue
        if not is_expert and not include_non_moe:
            continue
        dest = torch.device("cpu") if is_expert else device
        placed = tensor.to(dest)
        if name == "model.embed_tokens.weight":
            embed_tokens_weight = placed
        elif name == "lm_head.weight":
            saw_lm_head = True
        yield name, placed

    if include_non_moe and not saw_lm_head and embed_tokens_weight is not None:
        hf_config = cached_load_hf_config(model_path)
        if bool(getattr(hf_config, "tie_word_embeddings", False)):
            yield "lm_head.weight", embed_tokens_weight


def _xpu_available() -> bool:
    try:
        return bool(torch.xpu.is_available())
    except Exception:
        return False


class _Qwen4ExpTopkRouter(nn.Module):
    """Plain softmax top-k router (Qwen3.5/3.6's own -- NOT the DeepSeek-V3
    grouped/sigmoid/bias-corrected router every other MoE model in this port
    uses; confirmed from upstream's own moe.py, see module note above)."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int, dtype) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor):
        logits = self.gate(hidden_states)
        probs = F.softmax(logits, dtype=torch.float32, dim=-1)
        top_w, top_idx = torch.topk(probs, self.top_k, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)
        return top_idx, top_w.to(hidden_states.dtype)


class _Qwen4ExpExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dtype) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _Qwen4ExpMoE(nn.Module):
    """Softmax-topk router + always-on sigmoid-gated shared expert (see
    module note: upstream's real ``Qwen4ExpMoE`` subclasses ``Qwen3_5MoE``).
    In-VRAM only -- no offload/cpu/hybrid backend for this architecture."""

    def __init__(self, config, dtype, layer_id: int) -> None:
        super().__init__()
        if bool(getattr(config, "use_offload_moe", False)) or bool(getattr(config, "use_cpu_moe", False)) or bool(getattr(config, "use_hybrid", False)):
            raise NotImplementedError(
                "Qwen4ExpForCausalLM only supports the in-VRAM (fused) MoE backend "
                "-- offload/cpu/hybrid are not wired yet. Pass moe_backend=\"fused\" "
                "explicitly (EngineConfig's \"auto\" default does not know this "
                "architecture can't offload)."
            )
        self.layer_id = layer_id
        self.gate = _Qwen4ExpTopkRouter(config.hidden_size, config.num_experts, config.num_experts_per_tok, dtype)
        self.experts = nn.ModuleList(
            _Qwen4ExpExpert(config.hidden_size, config.moe_intermediate_size, dtype) for _ in range(config.num_experts)
        )
        shared_inter = config.attrs["shared_expert_intermediate_size"]
        self.shared_expert = _Qwen4ExpExpert(config.hidden_size, shared_inter, dtype)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        top_idx, top_w = self.gate(hidden_states)
        shared = torch.sigmoid(self.shared_expert_gate(hidden_states)) * self.shared_expert(hidden_states)

        top_idx_cpu = top_idx.to("cpu")
        out = torch.zeros_like(hidden_states)
        for e in range(len(self.experts)):
            for slot in range(top_idx.shape[1]):
                sel_cpu = top_idx_cpu[:, slot] == e
                if not bool(sel_cpu.any()):
                    continue
                idx = sel_cpu.nonzero(as_tuple=True)[0].to(hidden_states.device)
                w = top_w.index_select(0, idx)[:, slot, None]
                y = self.experts[e](hidden_states.index_select(0, idx))
                out.index_add_(0, idx, w * y)
        return out + shared


class _IndexKVStore:
    """Raw (pre-norm, pre-rope) QSA indexer key history, one entry per token per
    QSA layer -- a second, narrower store alongside the engine's own ``ctx.kv_cache``
    (which only has room for one fixed ``[num_kv_heads, head_dim]`` shape per layer,
    already used here for the real post-norm+rope K/V). Addressed through the SAME
    page table the main pool uses, so per-request isolation matches it exactly.
    Not incremental block-compression (that's a real engine optimization upstream's
    Triton backend does): every forward recompresses the whole visible history from
    this raw store, matching this port's established "reference correctness first"
    discipline for GDN/MLA/DSA/QSA alike."""

    def __init__(self, num_qsa_layers: int, num_slots: int, dim: int, device, dtype) -> None:
        self.buf = torch.zeros(num_qsa_layers, num_slots, dim, device=device, dtype=dtype)

    def write(self, qsa_slot: int, k_idx_raw: torch.Tensor, out_loc: torch.Tensor) -> None:
        self.buf[qsa_slot][out_loc.long()] = k_idx_raw

    def read(self, qsa_slot: int, table_idx: int, pos: torch.Tensor, page_table: torch.Tensor) -> torch.Tensor:
        slots = page_table[table_idx, pos.long()]
        return self.buf[qsa_slot][slots]


class _Qwen4ExpQSAAttention(nn.Module):
    """Gated GQA with the QSA block-sparse indexer (#208's primitives), full
    checkpoint weights + real KV-cache-backed history. ``q_proj`` doubles for
    an output gate (upstream real shape); q/k get a zero-centered per-vector
    RMSNorm (``grouped_plus_one_rms_norm`` with ``num_groups=1``) then partial
    rope. The indexer's raw (pre-norm/rope) k travels through its own
    :class:`_IndexKVStore` (see its own docstring for why a second store)."""

    def __init__(self, config, dtype, layer_id: int, qsa_slot: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.qsa_slot = qsa_slot
        self.num_q = config.num_attention_heads
        self.num_kv = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rotary_dim = config.attrs["rotary_dim"]
        self.theta = config.rope_theta or 10000.0
        self.eps = config.attrs["rms_norm_eps"]
        hidden = config.hidden_size

        self._qkv_split = [self.num_q * self.head_dim * 2, self.num_kv * self.head_dim, self.num_kv * self.head_dim]
        self.qkv_proj = nn.Linear(hidden, sum(self._qkv_split), bias=False, dtype=dtype)
        self.o_proj = nn.Linear(self.num_q * self.head_dim, hidden, bias=False, dtype=dtype)
        self.q_norm = nn.Parameter(torch.zeros(self.head_dim, dtype=dtype))
        self.k_norm = nn.Parameter(torch.zeros(self.head_dim, dtype=dtype))

        self.index_n_heads = config.attrs["index_n_heads"]
        self.index_head_dim = config.attrs["index_head_dim"]
        self.index_ratio = config.attrs["index_ratio"]
        self.index_budget_blocks = config.attrs["index_budget"] // self.index_ratio
        # The indexer's own rope runs over its own (narrower) head dim, not the
        # main attention's rotary_dim -- upstream's real rope64 uses head_size =
        # index_head_dim (see module note above).
        self.index_rotary_dim = min(self.rotary_dim, self.index_head_dim)
        self._index_split = [self.index_n_heads * self.index_head_dim, self.index_head_dim]
        self.index_qk_proj = nn.Linear(hidden, sum(self._index_split), bias=False, dtype=dtype)
        self.index_q_norm = nn.Parameter(torch.zeros(self.index_head_dim, dtype=dtype))
        self.index_k_norm = nn.Parameter(torch.zeros(self.index_head_dim, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor, table_idx: int, ctx, batch) -> torch.Tensor:
        T = hidden_states.shape[0]
        qg, k, v = self.qkv_proj(hidden_states).split(self._qkv_split, dim=-1)
        qg = qg.view(T, self.num_q, self.head_dim * 2)
        q = qg[..., : self.head_dim].contiguous()
        gate = qg[..., self.head_dim :].reshape(T, self.num_q * self.head_dim)
        k = k.view(T, self.num_kv, self.head_dim).contiguous()
        v = v.view(T, self.num_kv, self.head_dim).contiguous()

        q = grouped_plus_one_rms_norm(q, self.q_norm, self.eps, 1)
        k = grouped_plus_one_rms_norm(k, self.k_norm, self.eps, 1)
        cos, sin = qsa_rope_cos_sin(positions, self.rotary_dim, self.theta)
        q = apply_partial_rope(q, cos[:, None, :], sin[:, None, :], self.rotary_dim)
        k = apply_partial_rope(k, cos[:, None, :], sin[:, None, :], self.rotary_dim)

        q_idx_raw, k_idx_raw = self.index_qk_proj(hidden_states).split(self._index_split, dim=-1)
        q_idx_raw = q_idx_raw.view(T, self.index_n_heads, self.index_head_dim)

        ctx.kv_cache.write_kv(k.transpose(0, 1), v.transpose(0, 1), positions, self.layer_id)
        if ctx.qsa_index_kv is None:
            ctx.qsa_index_kv = _IndexKVStore(
                ctx.num_qsa_layers, ctx.kv_cache.num_slots, self.index_head_dim, hidden_states.device, hidden_states.dtype
            )
        ctx.qsa_index_kv.write(self.qsa_slot, k_idx_raw, positions)

        written = positions[-1].item() + 1
        read_pos = torch.arange(written, device=hidden_states.device)
        hist_k, hist_v = ctx.kv_cache.read_kv(table_idx, read_pos, self.layer_id)
        hist_k_idx_raw = ctx.qsa_index_kv.read(self.qsa_slot, table_idx, read_pos, ctx.page_table)

        pooled, block_starts = qsa_compress_keys(hist_k_idx_raw.unsqueeze(1), self.index_ratio)
        num_blocks = pooled.shape[0]
        block_last = block_starts + self.index_ratio - 1
        if num_blocks > 0:
            scores = qsa_score(
                q_idx_raw, pooled, self.index_q_norm, self.index_k_norm, self.eps,
                positions, block_starts, self.index_rotary_dim, self.theta,
            )
            topk = qsa_topk_blocks(scores, block_last, positions, self.index_budget_blocks)
        else:
            topk = torch.full((T, self.index_budget_blocks), -1, dtype=torch.long, device=hidden_states.device)
        mask = qsa_attend_mask(topk, self.index_ratio, positions, written)

        sm_scale = self.head_dim**-0.5
        out = qsa_sparse_attend(q, hist_k, hist_v, mask, sm_scale)
        gated = out.reshape(T, self.num_q * self.head_dim) * torch.sigmoid(gate.float()).to(out.dtype)
        return self.o_proj(gated)


class _Qwen4ExpDecoderLayer(nn.Module):
    """One hyper-connection decoder layer (see module docstring's own
    ``mix``/block/``combine`` flow, duplicated per attention AND MoE)."""

    def __init__(self, config, dtype, layer_id: int, qsa_slot: int | None) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.is_linear = config.attrs["layer_types"][layer_id] == "linear_attention"
        hc_count = config.attrs["hc_count"]
        hc_lowrank = config.attrs["hc_lowrank"]
        eps = config.attrs["rms_norm_eps"]
        if self.is_linear:
            self.self_attn = None
            self.linear_attn = _GatedDeltaNet(
                config, None, dtype, layer_id,
                linear_num_key_heads=config.attrs["linear_num_key_heads"],
                linear_num_value_heads=config.attrs["linear_num_value_heads"],
                linear_key_head_dim=config.attrs["linear_key_head_dim"],
                linear_value_head_dim=config.attrs["linear_value_head_dim"],
                linear_conv_kernel_dim=config.attrs["linear_conv_kernel_dim"],
                eps=eps,
            )
        else:
            self.linear_attn = None
            self.self_attn = _Qwen4ExpQSAAttention(config, dtype, layer_id, qsa_slot)
        self.mlp = _Qwen4ExpMoE(config, dtype, layer_id)
        self.attn_hyper_connection = GatedResidual(config.hidden_size, hc_count, hc_lowrank, eps, dtype=dtype)
        self.mlp_hyper_connection = GatedResidual(config.hidden_size, hc_count, hc_lowrank, eps, dtype=dtype)
        self.ple = None
        if layer_id in config.attrs["ple_layer_ids"]:
            self.ple = PLELayer(
                config.hidden_size, hc_count, config.attrs["ple_embed_dim"],
                config.attrs["ple_conv_kernel_size"], config.attrs["ngram_size"], eps, dtype=dtype,
            )

    def forward(
        self, R: torch.Tensor, positions: torch.Tensor, table_idx: int, ctx, batch,
        ple_embeddings: torch.Tensor | None = None, ple_conv_state: torch.Tensor | None = None,
        linear_slot_idx=None,
    ):
        new_ple_conv_state = None
        if self.ple is not None:
            D, new_ple_conv_state = self.ple(R, ple_embeddings, ple_conv_state)
            R = R + D

        block_input, inject = self.attn_hyper_connection.mix(R)
        if self.is_linear:
            block_output = self.linear_attn(block_input, positions, table_idx, ctx, batch, linear_slot_idx=linear_slot_idx)
        else:
            block_output = self.self_attn(block_input, positions, table_idx, ctx, batch)
        R = self.attn_hyper_connection.combine(R, block_output, inject)

        block_input, inject = self.mlp_hyper_connection.mix(R)
        block_output = self.mlp(block_input)
        R = self.mlp_hyper_connection.combine(R, block_output, inject)
        return R, new_ple_conv_state


class Qwen4ExpForCausalLM(nn.Module):
    """Qwen3.8-Flash-Next: real forward pass for the Intel engine loop.

    The residual state is ``R [T, hc_count*hidden]`` end to end (see module
    docstring): the embedding is repeated over ``hc_count`` streams, every
    layer mixes it down to one block input and injects the block's output
    back, and the top-level mixer collapses the streams once before
    ``lm_head``. There is no input/post layernorm and no final ``norm`` --
    the hyper-connection norms are the only ones (upstream's own real
    ``model.py`` module docstring, confirmed)."""

    def __init__(self, config, device=None) -> None:
        super().__init__()
        self.config = config
        if device is None:
            device = torch.device("xpu") if _xpu_available() else torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        dtype = getattr(config, "dtype", None) or torch.float32
        self.dtype = dtype
        self.hc_count = config.attrs["hc_count"]
        vocab_size = config.vocab_size
        hidden_size = config.hidden_size

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, device=device, dtype=dtype)
        layer_types = config.attrs["layer_types"]
        qsa_layer_ids = [i for i, t in enumerate(layer_types) if t == "full_attention"]
        qsa_slot_of = {lid: i for i, lid in enumerate(qsa_layer_ids)}
        self.num_qsa_layers = len(qsa_layer_ids)
        self.layers = nn.ModuleList(
            _Qwen4ExpDecoderLayer(config, dtype, layer_id=i, qsa_slot=qsa_slot_of.get(i))
            for i in range(config.num_layers)
        )
        eps = config.attrs["rms_norm_eps"]
        self.hyper_connection_mixer = GatedResidual(
            hidden_size, self.hc_count, config.attrs["hc_lowrank"], eps, use_combine=False, dtype=dtype
        )
        from freetoken.layers import LinearReplicated

        self.lm_head = LinearReplicated(hidden_size, vocab_size, has_bias=False, dtype=dtype)

        max_slots = int(config.attrs.get("max_running_req") or 8)
        self.linear_state_pool = _LinearStatePool()
        for layer in self.layers:
            if layer.linear_attn is not None:
                ln = layer.linear_attn
                self.linear_state_pool.register(
                    ln.layer_id, max_slots, (ln.num_v_heads, ln.head_k_dim, ln.head_v_dim),
                    (ln.conv_dim, ln.conv_kernel - 1), device, dtype,
                )

        self._ple_layer_id = config.attrs["ple_layer_ids"][0] if config.attrs["ple_layer_ids"] else None
        if self._ple_layer_id is not None:
            num_ngram_heads = (config.attrs["ngram_size"] - 1) * config.attrs["heads_per_ngram"]
            # one table row-width per (ngram order, head): ple_embed_dim is the
            # TOTAL width after concatenating every head's own lookup row.
            row_width = config.attrs["ple_embed_dim"] // num_ngram_heads
            mult, sizes, offsets = derive_ngram_hash_constants(
                vocab_size=vocab_size,
                ngram_size=config.attrs["ngram_size"],
                num_ngram_heads=num_ngram_heads,
                ngram_vocab_size_base=config.attrs["ngram_vocab_size_base"],
                ple_layer_index=0,
            )
            self._ple_multipliers = torch.tensor(mult, dtype=torch.int64, device=device)
            self._ple_sizes = torch.tensor(sizes, dtype=torch.int64, device=device)
            self._ple_offsets = torch.tensor(offsets, dtype=torch.int64, device=device)
            table_rows = offsets[-1] + sizes[-1]
            self._ple_table = PleInMemoryTable(
                torch.zeros(table_rows, row_width, device=device, dtype=dtype)
            )
            self._ple_conv_state: dict = {}
            self._ple_ngram_ctx: dict = {}

        if self.device.type != "cpu":
            self.to(self.device)

    def _ple_precompute(self, table_idx: int, token_ids: torch.Tensor):
        args = self.config.attrs
        boundary = args["ngram_boundary_token_id"]
        ngram_size = args["ngram_size"]
        ctx_len = ngram_size - 1
        prior = self._ple_ngram_ctx.get(table_idx)
        if prior is None:
            prior = torch.full((ctx_len,), boundary, dtype=token_ids.dtype, device=token_ids.device)
        full_ids = torch.cat([prior, token_ids]) if ctx_len else token_ids
        row_ids = ple_ngram_row_ids(
            full_ids, boundary, ngram_size, args["heads_per_ngram"],
            self._ple_multipliers, self._ple_sizes, self._ple_offsets,
        )
        row_ids = row_ids[ctx_len:] if ctx_len else row_ids
        embeddings = self._ple_table.lookup(row_ids).reshape(token_ids.shape[0], -1)
        self._ple_ngram_ctx[table_idx] = full_ids[-ctx_len:] if ctx_len else full_ids

        conv_state = self._ple_conv_state.get(table_idx)
        if conv_state is None:
            width = self.hc_count * self.config.hidden_size
            state_len = (args["ple_conv_kernel_size"] - 1) * ngram_size
            conv_state = torch.zeros(width, state_len, device=token_ids.device, dtype=self.dtype)
        return embeddings, conv_state

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, out_loc: torch.Tensor) -> torch.Tensor:
        from freetoken.core import get_global_ctx

        ctx = get_global_ctx()
        batch = ctx.batch
        reqs = batch.reqs
        num_tokens = input_ids.shape[0]
        ctx.qsa_index_kv = getattr(ctx, "qsa_index_kv", None)
        ctx.num_qsa_layers = self.num_qsa_layers
        if ctx.linear_state_pool is None:
            ctx.linear_state_pool = self.linear_state_pool
        for req in reqs:
            if req.linear_slot_idx is None:
                req.linear_slot_idx = req.table_idx

        hidden_all = self.embed_tokens(input_ids).repeat(1, self.hc_count)
        out = torch.empty((batch.size, self.config.hidden_size), device=hidden_all.device, dtype=hidden_all.dtype)

        offset = 0
        extend_lens = batch.extend_lens
        if extend_lens is None:
            prefill = batch.is_prefill or (num_tokens > batch.size)
            extend_lens = [req.extend_len if prefill else 1 for req in reqs]
        is_decode_batch = batch.phase == "decode"
        for i, req in enumerate(reqs):
            ext = 1 if is_decode_batch else int(extend_lens[i])
            token_slice = slice(offset, offset + ext)
            R = hidden_all[token_slice]
            pos = positions[token_slice]
            tok_ids = input_ids[token_slice]
            for layer in self.layers:
                ple_embeddings = ple_conv_state = None
                if layer.ple is not None:
                    ple_embeddings, ple_conv_state = self._ple_precompute(req.table_idx, tok_ids)
                R, new_conv_state = layer(
                    R, pos, req.table_idx, ctx, batch,
                    ple_embeddings=ple_embeddings, ple_conv_state=ple_conv_state,
                    linear_slot_idx=req.linear_slot_idx,
                )
                if new_conv_state is not None:
                    self._ple_conv_state[req.table_idx] = new_conv_state
            collapsed = self.hyper_connection_mixer.mix(R)[0]
            out[i] = collapsed[-1]
            offset += ext

        return self.lm_head(out)


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
    "derive_ngram_hash_constants",
    "ple_ngram_row_ids",
    "PleInMemoryTable",
    "PleDiskTable",
    "ple_short_conv",
    "PLELayer",
    "parse_config",
    "iter_weights",
    "Qwen4ExpForCausalLM",
]
