"""GPTQ INT4 (and INT2/INT8) weight dequantization (issue `quant-xpu`, #10).

Standard AutoGPTQ packed-integer format: this is the storage layout behind
real checkpoints like the official ``Qwen/Qwen3.5-35B-A3B-GPTQ-Int4``
(``bits: 4, group_size: 128, sym: true, desc_act: false``) -- a group of
``group_size`` consecutive input features shares one scale (and, for
asymmetric quantization, one zero-point) per output channel.

Ported from (and cross-checked line-by-line against) the canonical
reference, AutoGPTQ's own ``QuantLinear`` unpack/dequant path:
https://github.com/AutoGPTQ/AutoGPTQ/blob/main/auto_gptq/nn_modules/qlinear/qlinear_cuda.py
-- bit-packing conventions like this one are exactly the kind of thing that
produces silently-wrong-but-plausible numbers if hand-derived from memory,
so this was built by pulling the real source rather than guessing (same
policy as the FP8/MXFP4 ports, #125/#129, which had a public spec to check
against instead of a reference implementation).

Storage (for a ``[K, N]`` logical weight, ``K`` = in_features, ``N`` =
out_features, packing factor ``P = 32 // bits`` int32 sub-values per word):

    qweight  int32 [K // P, N]              -- P rows of K packed per word
    qzeros   int32 [ceil(K/group_size), N // P]  -- P output channels packed per word
    scales   fp16/bf16/fp32 [ceil(K/group_size), N]
    g_idx    int32 [K]                       -- g_idx[k] = k // group_size (desc_act=False)

Dequant: ``weight[k, n] = scales[g_idx[k], n] * (unpack(qweight)[k, n] -
(unpack(qzeros)[g_idx[k], n] + 1))`` -- the ``+ 1`` on the zero-point is a
real, well-known AutoGPTQ convention (not a bug to "fix"): the packer
stores ``zero_point - 1`` because the unsigned 4-bit code 0 is reserved,
so every reader must undo it.
"""
from __future__ import annotations

import torch

_BITS = 4  # this port only implements the 4-bit case (the real checkpoint's format)


def _unpack_int32(packed: torch.Tensor, *, bits: int, dim: int) -> torch.Tensor:
    """Unpack ``bits``-wide sub-values from an int32 tensor's ``dim``, where
    each int32 word holds ``32 // bits`` consecutive values (least-significant
    bits = the first value). Inserts a new axis of size ``32 // bits``
    immediately after ``dim``, to be reshaped/flattened by the caller (the
    two callers below flatten it into the K or N axis respectively --
    matching AutoGPTQ's own two different post-unpack reshapes)."""
    p = 32 // bits
    shifts = torch.arange(0, 32, bits, dtype=torch.int32, device=packed.device)
    shape = [1] * packed.ndim
    shape.insert(dim + 1, p)
    shifts = shifts.reshape(*([1] * dim), p, *([1] * (packed.ndim - dim - 1)))
    expanded = packed.unsqueeze(dim + 1).expand(*packed.shape[: dim + 1], p, *packed.shape[dim + 1 :])
    return torch.bitwise_and(torch.bitwise_right_shift(expanded, shifts), (1 << bits) - 1)


def dequantize_gptq_int4(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reconstruct the dense ``[K, N]`` weight (``nn.Linear``'s ``in_features
    x out_features`` orientation -- transpose the result for a ``[N, K]``
    ``nn.Linear.weight``) from GPTQ-packed tensors.

    ``qweight`` is ``[K // 8, N]`` int32, ``qzeros`` is
    ``[ceil(K/group_size), N // 8]`` int32, ``scales`` is
    ``[ceil(K/group_size), N]``, ``g_idx`` is ``[K]`` int32.
    """
    if qweight.dtype != torch.int32:
        raise TypeError(f"qweight must be int32, got {qweight.dtype}")
    if qzeros.dtype != torch.int32:
        raise TypeError(f"qzeros must be int32, got {qzeros.dtype}")
    if g_idx.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"g_idx must be an integer tensor, got {g_idx.dtype}")

    k_packed, n = qweight.shape
    k = k_packed * (32 // _BITS)
    if scales.shape[1] != n or qzeros.shape[1] * (32 // _BITS) != n:
        raise ValueError(
            f"qweight/qzeros/scales shape mismatch: qweight={tuple(qweight.shape)}, "
            f"qzeros={tuple(qzeros.shape)}, scales={tuple(scales.shape)}"
        )
    if g_idx.shape[0] != k:
        raise ValueError(f"g_idx length {g_idx.shape[0]} does not match qweight's implied K={k}")

    # qweight: [K//8, N] -> unpack dim 0 (8 rows packed per int32 word) -> [K//8, 8, N] -> [K, N].
    weight = _unpack_int32(qweight, bits=_BITS, dim=0).reshape(k, n)
    # qzeros: [G, N//8] -> unpack dim 1 (8 output channels packed per int32 word) -> [G, N//8, 8] -> [G, N].
    zeros = _unpack_int32(qzeros, bits=_BITS, dim=1).reshape(qzeros.shape[0], n)
    zeros = zeros.to(torch.int32) + 1  # AutoGPTQ's stored-minus-one convention

    g_idx = g_idx.long()
    per_row_scale = scales[g_idx]  # [K, N]
    per_row_zero = zeros[g_idx]  # [K, N]
    return (per_row_scale.to(torch.float32) * (weight.to(torch.float32) - per_row_zero.to(torch.float32))).to(
        out_dtype
    )


def gptq_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x @ dequantize_gptq_int4(...).T (+ bias)``. Dequant-on-forward
    drop-in matching ``fp8_block_linear`` / ``mxfp4_linear`` / ``int8_linear``'s
    shape -- a caller driving this repeatedly should dequantize once at load
    time instead (via :func:`dequantize_gptq_int4` directly, transposed to
    ``nn.Linear``'s ``[out_features, in_features]``) and cache the result.
    """
    w = dequantize_gptq_int4(qweight, qzeros, scales, g_idx, out_dtype=x.dtype)
    return torch.nn.functional.linear(x, w.T, bias)
