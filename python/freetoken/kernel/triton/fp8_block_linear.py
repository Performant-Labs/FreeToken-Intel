"""DeepSeek-V3-style 128x128 block-FP8 weights: quantize / dequantize (issue `quant-xpu`).

Upstream NVIDIA path: python/freetoken/kernel/triton/fp8_block_linear.py

Upstream's version is a full native-FP8-tensor-core Triton GEMM (dynamic
per-token activation quant + a custom ``tl.dot`` kernel that runs in real
fp8 when the platform supports it, falling back to a bf16-emulated dot
otherwise -- see its own ``e4m3_native_cx()`` capability check). Porting
that whole kernel to Intel's Triton-XPU backend is real, separate work
(unverified whether ``triton_xpu`` even exposes an fp8 ``tl.dot`` today) --
not attempted here.

This is the **dequant-on-load** half instead, which issue `quant-xpu`'s own
accept criteria explicitly allows ("Document which dtypes are native vs
dequant-on-load"): a checkpoint's fp8 weight + per-block scale round-trips
through this module to a plain bf16 (or fp32) tensor, then flows through
this port's EXISTING, already-correct/tested bf16 fused / offload MoE and
dense-linear code paths unchanged. No custom kernel, no hardware fp8
tensor-core dependency -- just the storage-format convention, matched
exactly to upstream (and to sglang's / vLLM's ``w8a8_block_fp8`` weight
layout, which real FP8-quantized checkpoints on HuggingFace already use):

    weight_bf16[i, j] = weight_fp8[i, j] * weight_scale_inv[i // block, j // block]

``weight`` is ``[N, K]`` fp8-e4m3 (``torch.float8_e4m3fn``); ``weight_scale_inv``
is ``[ceil(N/block), ceil(K/block)]`` (bf16 or fp32 in the checkpoint; computed
here in fp32 for the intermediate multiply and cast back to the caller's
``out_dtype``). ``block`` defaults to 128, DeepSeek-V3 / sglang / vLLM's
convention (also what a real checkpoint's ``quantization_config
.weight_block_size`` would name -- read from there, not hardcoded, once
loader-side integration lands as a follow-up).

VRAM benefit even without a native fp8 GEMM: the checkpoint's on-disk /
host-resident footprint is the fp8 tensor (half the bytes of bf16) plus a
tiny per-block scale table -- dequantizing to bf16 happens only once, at
load time (or lazily per forward, for a backend that wants to keep the fp8
bytes resident and pay the dequant cost repeatedly to save VRAM -- not what
this first cut does; :func:`dequantize_block_fp8` is called once here).
"""
from __future__ import annotations

import torch

FP8 = torch.float8_e4m3fn
_BLOCK = 128


def dequantize_block_fp8(
    weight_fp8: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    *,
    block: int = _BLOCK,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reconstruct a dense ``[N, K]`` tensor from block-quantized fp8 + scales.

    ``weight_fp8`` is ``[N, K]`` (any fp8 dtype torch supports, typically
    ``torch.float8_e4m3fn``); ``weight_scale_inv`` is ``[ceil(N/block),
    ceil(K/block)]``. ``N`` / ``K`` need not be exact multiples of ``block``
    (the last row/col block of scales covers the remainder).
    """
    if weight_fp8.ndim != 2:
        raise ValueError(f"weight_fp8 must be 2-D [N, K], got shape {tuple(weight_fp8.shape)}")
    N, K = weight_fp8.shape
    expected_scale_shape = (
        (N + block - 1) // block,
        (K + block - 1) // block,
    )
    if tuple(weight_scale_inv.shape) != expected_scale_shape:
        raise ValueError(
            f"weight_scale_inv shape {tuple(weight_scale_inv.shape)} does not match "
            f"the expected {expected_scale_shape} for weight_fp8 shape {(N, K)} at block={block}"
        )
    # Upsample the per-block scale table to per-element via repeat_interleave (each
    # block's single scale broadcasts to every element in that block), then multiply
    # in fp32 for precision before casting to the caller's dtype. Trim to (N, K) in
    # case N/K are not exact multiples of block (the repeat overshoots by the pad).
    scale = weight_scale_inv.to(torch.float32)
    scale = scale.repeat_interleave(block, dim=0)[:N]
    scale = scale.repeat_interleave(block, dim=1)[:, :K]
    return (weight_fp8.to(torch.float32) * scale).to(out_dtype)


def quantize_block_fp8(
    weight: torch.Tensor,
    *,
    block: int = _BLOCK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a dense ``[N, K]`` tensor to block-fp8 + per-block scale.

    The inverse of :func:`dequantize_block_fp8` (used to build fabricated fp8
    test checkpoints, and by any future checkpoint-conversion tool -- ``ft
    checkpoint`` / issue ``ftw-checkpoint``, #11 -- that quantizes an existing
    bf16 checkpoint rather than only reading an already-quantized one).
    Per-block scale is ``max(abs(block)) / fp8_max`` (symmetric, matching
    upstream's dynamic activation quant formula), clamped away from zero so an
    all-zero block does not divide by zero.
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D [N, K], got shape {tuple(weight.shape)}")
    N, K = weight.shape
    fp8_max = torch.finfo(FP8).max
    n_blocks_row = (N + block - 1) // block
    n_blocks_col = (K + block - 1) // block
    pad_n = n_blocks_row * block - N
    pad_k = n_blocks_col * block - K
    w = weight.to(torch.float32)
    if pad_n or pad_k:
        w = torch.nn.functional.pad(w, (0, pad_k, 0, pad_n))
    blocks = w.reshape(n_blocks_row, block, n_blocks_col, block)
    amax = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-12)  # [n_blocks_row, n_blocks_col]
    scale = amax / fp8_max
    quantized = (blocks / scale[:, None, :, None]).clamp(-fp8_max, fp8_max)
    quantized = quantized.reshape(n_blocks_row * block, n_blocks_col * block)[:N, :K].to(FP8)
    return quantized, scale


def fp8_block_linear(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    *,
    block: int = _BLOCK,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x @ dequantize_block_fp8(weight_fp8, weight_scale_inv).T (+ bias)``.

    The dequant-on-forward drop-in for a quantized ``nn.Linear``-shaped
    weight (``[out_features, in_features]``, the checkpoint's native
    layout): dequantizes once per call and runs the existing bf16 matmul.
    For a weight used repeatedly across many forward calls (the common
    case), the caller should dequantize ONCE at load time instead (via
    :func:`dequantize_block_fp8` directly) and cache the bf16 result --
    this per-call convenience wrapper pays the dequant cost every call.
    """
    w = dequantize_block_fp8(weight_fp8, weight_scale_inv, block=block, out_dtype=x.dtype)
    return torch.nn.functional.linear(x, w, bias)
