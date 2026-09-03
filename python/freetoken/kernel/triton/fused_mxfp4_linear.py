"""Native MXFP4 GEMM: matmul directly against packed ``blocks``/``scales``,
never materializing a dequantized bf16 weight tensor (issue `moe-quant-
banks-native-multi`, #163, extending #139's GPTQ approach to MXFP4).

``mxfp4_linear.py``'s ``dequantize_mxfp4_blocks`` (and the offload cache's
``SlotWeightAccessor``) both dequantize an expert's packed MXFP4 weight to
a dense tensor first (a LUT gather over 16 codes + an ``exp2`` scale
multiply + a reshape), THEN run a plain matmul. This module fuses the two:
each Triton program dequantizes only the ``[QBLOCK=32, BLOCK_N]`` weight
tile it is about to consume and feeds it straight into ``tl.dot``.

Feasibility and real measured latency, both confirmed on the actual B70
(Xe2, Triton-XPU 3.7.2, torch 2.13.0+xpu):

* Correctness: bit-exact (or float32-rounding-tolerance close) to
  ``dequantize_mxfp4_blocks`` + a plain matmul across aligned/ragged
  M/K/N shapes.
* Performance: MXFP4's dequant-then-matmul fallback is notably more
  expensive than block-FP8's (a 16-way LUT gather, not a plain multiply)
  -- ~0.14ms flat across M on a 2048x768 projection. The fused kernel wins
  clearly across the ENTIRE measured range (M=1 through M=128, ~1.2x-2.2x
  faster at every point tried) with the default ``BLOCK_M=8, BLOCK_N=16``
  tile -- unlike GPTQ (#139) and block-FP8, which both lose to the
  fallback past a real crossover, no such crossover was found for MXFP4 in
  the M<=128 range this module's own sweep covered.

Design mirrors ``fused_fp8_linear``: the K-loop iterates one full
QBLOCK=32 quantization block per step (MXFP4's own block size), so each
loop iteration handles exactly one packed-nibble byte-pair -> two output
elements per byte. Unlike block-FP8 (whose block spans both the N and K
axes, so one scale serves a whole N-tile), MXFP4's scale is per (output
channel, K-block) -- one exponent byte per ROW per block, never shared
across N -- so this kernel carries a length-``BLOCK_N`` scale vector per
iteration (the same broadcast-per-column shape GPTQ's scale/zero vectors
use), not FP8's single scalar.

The E2M1 LUT (8 non-negative magnitudes: ``[0.0, 0.5, 1.0, 1.5, 2.0, 3.0,
4.0, 6.0]``, sign = the nibble's MSB -- see ``mxfp4_linear.py``'s own
module docstring) is implemented as a short ``tl.where`` chain rather than
an actual gather: Triton has no cheap small-constant-table gather
primitive, but an 8-way elementwise select over the 3 magnitude bits is
just as fast and avoids introducing one.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

_QBLOCK = 32  # MXFP4's own quantization block size (32 fp4 elements/block)


@triton.jit
def _mxfp4_magnitude_lut(code3):
    """``code3`` (the low 3 bits of an MXFP4 nibble, in [0, 7]) -> its E2M1
    magnitude. See this module's own docstring for the full LUT and why a
    ``tl.where`` chain replaces an actual gather."""
    v = tl.where(code3 == 0, 0.0, 0.5)
    v = tl.where(code3 == 2, 1.0, v)
    v = tl.where(code3 == 3, 1.5, v)
    v = tl.where(code3 == 4, 2.0, v)
    v = tl.where(code3 == 5, 3.0, v)
    v = tl.where(code3 == 6, 4.0, v)
    v = tl.where(code3 == 7, 6.0, v)
    return v


@triton.jit
def _fused_mxfp4_matmul_kernel(
    x_ptr, blk_ptr, sc_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_bn, stride_bkb, stride_bbyte,
    stride_scn, stride_sckb,
    stride_om, stride_on,
    QBLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    num_k_blocks = tl.cdiv(K, QBLOCK)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    byte_offs = tl.arange(0, QBLOCK // 2)  # 16 packed bytes per 32-element block

    for kb in range(num_k_blocks):
        # blocks: [BLOCK_N, 16] uint8 -- one MXFP4 block's packed bytes per row.
        b_ptrs = blk_ptr + offs_n[:, None] * stride_bn + kb * stride_bkb + byte_offs[None, :] * stride_bbyte
        raw = tl.load(b_ptrs, mask=mask_n[:, None], other=0)  # [BLOCK_N, 16]

        # low nibble = even element, high nibble = odd (matches
        # dequantize_mxfp4_blocks' own torch.stack((blocks & 0x0F, blocks >>
        # 4)) unpack order).
        lo = raw & 0xF
        hi = (raw >> 4) & 0xF
        lo_sign = tl.where(lo >= 8, -1.0, 1.0)
        hi_sign = tl.where(hi >= 8, -1.0, 1.0)
        lo_val = lo_sign * _mxfp4_magnitude_lut(lo & 0x7)
        hi_val = hi_sign * _mxfp4_magnitude_lut(hi & 0x7)

        s_ptrs = sc_ptr + offs_n * stride_scn + kb * stride_sckb
        s_raw = tl.load(s_ptrs, mask=mask_n, other=0)
        scale = tl.exp2(s_raw.to(tl.float32) - 127.0)  # [BLOCK_N], IEEE-754 float32 bias

        lo_val = lo_val * scale[:, None]
        hi_val = hi_val * scale[:, None]

        # interleave lo/hi -> [BLOCK_N, 32] (n-major), then transpose to feed
        # tl.dot as x[BLOCK_M, QBLOCK] @ w[QBLOCK, BLOCK_N].
        w_tile_n = tl.interleave(lo_val, hi_val)
        w_tile = tl.trans(w_tile_n)

        k_start = kb * QBLOCK
        offs_k = k_start + tl.arange(0, QBLOCK)
        mask_k = offs_k < K
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)

        acc += tl.dot(x_tile, w_tile, allow_tf32=False)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def fused_mxfp4_linear(
    x: torch.Tensor,
    weight_blocks: torch.Tensor,
    weight_scales: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    block_m: int = 8,
    block_n: int = 16,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """``x @ dequantize_mxfp4_blocks(weight_blocks, weight_scales).T (+
    bias)``, fused into one Triton kernel that never materializes the
    dense weight.

    ``weight_blocks`` is ``[N, K//32, 16]`` ``uint8``; ``weight_scales`` is
    ``[N, K//32]`` ``uint8``, matching :func:`mxfp4_linear.
    dequantize_mxfp4_blocks`'s own contract exactly.

    Defaults (``block_m=8, block_n=16``) are the measured-best tile config
    on the real B70 -- see this module's own docstring for the sweep. Unlike
    :func:`freetoken.kernel.triton.gptq_fused_linear.fused_gptq_linear` and
    :func:`freetoken.kernel.triton.fused_fp8_linear.fused_fp8_linear`, no
    real crossover past which the fallback wins was found in the M<=128
    range tested -- :func:`prefer_fused_over_dequant` still exists for API
    symmetry with those two, but its threshold is a measured upper bound
    of where "fused wins" was actually tested, not a real observed
    crossover.
    """
    if weight_blocks.dtype != torch.uint8:
        raise TypeError(f"weight_blocks must be uint8, got {weight_blocks.dtype}")
    if weight_scales.dtype != torch.uint8:
        raise TypeError(f"weight_scales must be uint8, got {weight_scales.dtype}")
    if weight_blocks.shape[-1] != 16:
        raise ValueError(f"weight_blocks' last dim must be 16 packed bytes, got {weight_blocks.shape[-1]}")

    m, k = x.shape
    n, n_blocks, _ = weight_blocks.shape
    if n_blocks * _QBLOCK != k:
        raise ValueError(f"x's K={k} does not match weight_blocks' implied K={n_blocks * _QBLOCK}")
    if tuple(weight_scales.shape) != (n, n_blocks):
        raise ValueError(
            f"weight_scales shape {tuple(weight_scales.shape)} does not match "
            f"weight_blocks' implied {(n, n_blocks)}"
        )

    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _fused_mxfp4_matmul_kernel[grid](
        x, weight_blocks, weight_scales, out,
        m, n, k,
        x.stride(0), x.stride(1),
        weight_blocks.stride(0), weight_blocks.stride(1), weight_blocks.stride(2),
        weight_scales.stride(0), weight_scales.stride(1),
        out.stride(0), out.stride(1),
        QBLOCK=_QBLOCK, BLOCK_M=block_m, BLOCK_N=block_n,
    )
    result = out.to(out_dtype if out_dtype is not None else x.dtype)
    if bias is not None:
        result = result + bias
    return result


def fused_mxfp4_expert_forward(
    x: torch.Tensor,
    blocks_gate_up: torch.Tensor,
    scales_gate_up: torch.Tensor,
    blocks_down: torch.Tensor,
    scales_down: torch.Tensor,
    *,
    intermediate: int,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """One MoE expert's SwiGLU forward (``down(silu(gate(x)) * up(x))``)
    entirely against packed MXFP4 banks, via two :func:`fused_mxfp4_linear`
    calls -- mirrors :func:`freetoken.kernel.triton.gptq_fused_linear.
    fused_gptq_expert_forward` / :func:`freetoken.kernel.triton.
    fused_fp8_linear.fused_fp8_expert_forward`'s structure exactly."""
    out_dtype = out_dtype if out_dtype is not None else x.dtype
    gu = fused_mxfp4_linear(x, blocks_gate_up, scales_gate_up, out_dtype=torch.float32)
    gate, up = gu[:, :intermediate], gu[:, intermediate:]
    h = (torch.nn.functional.silu(gate) * up).to(out_dtype)
    return fused_mxfp4_linear(h, blocks_down, scales_down, out_dtype=out_dtype)


# See this module's own docstring: no real crossover was found in the
# M<=128 range tested (unlike GPTQ's #139 or block-FP8's #163 crossovers,
# both real observed points past which the fallback started winning) --
# this is a measured upper BOUND of "fused was tested and won here", kept
# conservative rather than assuming it holds for an untested, much larger M.
_FUSED_MAX_M = 128


def prefer_fused_over_dequant(m: int) -> bool:
    """Whether :func:`fused_mxfp4_linear` (native GEMM) is expected to beat
    the plain dequant-then-matmul fallback for a batch of ``m`` rows -- see
    this module's own docstring for the measured numbers. Mirrors
    :func:`freetoken.kernel.triton.gptq_fused_linear.
    prefer_fused_over_dequant`'s role for GPTQ.
    """
    return m <= _FUSED_MAX_M
