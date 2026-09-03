"""Native compressed-tensors pack-quantized INT8 GEMM: matmul directly
against packed ``weight_packed``/``weight_scale``, never materializing a
dequantized bf16 weight tensor (issue `moe-quant-banks-native-multi`,
#163, extending #139's GPTQ approach to INT8 -- the last of the three
slices, following block-FP8/#164 and MXFP4/#165).

``int8_packed_linear.py``'s ``dequantize_int8_packed`` (and the offload
cache's ``SlotWeightAccessor``) both dequantize an expert's packed INT8
weight to a dense tensor first, THEN run a plain matmul. This module fuses
the two: each Triton program dequantizes only the ``[GROUP_SIZE,
BLOCK_N]`` weight tile it is about to consume and feeds it straight into
``tl.dot``.

Feasibility and real measured latency, both confirmed on the actual B70
(Xe2, Triton-XPU 3.7.2, torch 2.13.0+xpu):

* Correctness: bit-exact/float32-rounding-tolerance match to
  ``dequantize_int8_packed`` + a plain matmul across aligned/ragged
  M/K/N/group_size shapes.
* Performance: INT8's dequant-then-matmul fallback costs ~0.15-0.2ms flat
  across M on a 2048x768, group_size=128 projection (a real bit-unpack, so
  more expensive than block-FP8's plain multiply, similar order to
  GPTQ's). The fused kernel wins clearly through M=64 (e.g. M=1: ~0.034ms
  vs ~0.203ms, ~6x faster; M=64: ~0.114ms vs ~0.159ms) and starts losing
  at M=128 (~0.217ms vs ~0.162ms) -- closer to GPTQ's crossover (#139,
  M<=32) than block-FP8's conservative one (#164, M<=8), consistent with
  both formats sharing a real bit-unpack in their fallback's dequant cost.

Design mirrors ``fused_gptq_linear`` closely: the K-loop iterates one full
quantization group per step (``group_size`` must be a multiple of 4, the
int8 packing factor -- true of every real checkpoint's group_size found so
far, e.g. 32), unpacking the packed ``[BLOCK_N, group_size/4]`` int32 tile
into ``[BLOCK_N, group_size]`` via bitwise shift+mask (4 int8 values
densely packed per word, ``+128`` offset-binary -- see
``int8_packed_linear.py``'s own module docstring for the format,
NOT GPTQ's ``stored-minus-one`` convention). Unlike GPTQ's ``[K, N]``
weight orientation, ``weight_packed`` here is ``[N, ceil(K/4)]`` (the same
``nn.Linear``-native out/in orientation block-FP8 and MXFP4 both use), so
the K-major unpack/transpose pattern instead mirrors
``fused_fp8_linear``'s / ``fused_mxfp4_linear``'s N-row-major-then-transpose
shape, not GPTQ's.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

_BITS = 8
_ELEMS_PER_WORD = 32 // _BITS  # 4: int8 values packed per int32 word
_OFFSET = 1 << (_BITS - 1)  # compressed_tensors' flat signed<->unsigned pack offset


@triton.jit
def _fused_int8_matmul_kernel(
    x_ptr, wp_ptr, s_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wpn, stride_wpc,
    stride_sn, stride_sg,
    stride_om, stride_on,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BITS: tl.constexpr, OFFSET: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    NUM_PACK_COLS: tl.constexpr = GROUP_SIZE // 4  # int32 words spanned by one group
    offs_pc = tl.arange(0, NUM_PACK_COLS)
    shifts = tl.arange(0, 4) * BITS

    num_groups = tl.cdiv(K, GROUP_SIZE)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for g in range(num_groups):
        pc_start = g * NUM_PACK_COLS
        wp_ptrs = wp_ptr + offs_n[:, None] * stride_wpn + (pc_start + offs_pc[None, :]) * stride_wpc
        packed = tl.load(wp_ptrs, mask=mask_n[:, None], other=0)  # [BLOCK_N, NUM_PACK_COLS] int32

        # unpack -> [BLOCK_N, NUM_PACK_COLS, 4], merge (c, b) -> k = c*4+b
        # (matches weight_packed's own bit layout: byte b of word c holds
        # k = c*4+b -- see int8_packed_linear.unpack_int8_from_int32).
        w_bits = (packed[:, :, None] >> shifts[None, None, :]) & 0xFF
        w_tile_n = tl.reshape(w_bits, (BLOCK_N, GROUP_SIZE)).to(tl.float32) - OFFSET

        s_ptrs = s_ptr + offs_n * stride_sn + g * stride_sg
        scale = tl.load(s_ptrs, mask=mask_n, other=0.0).to(tl.float32)  # [BLOCK_N]
        w_tile_n = w_tile_n * scale[:, None]

        w_tile = tl.trans(w_tile_n)  # [GROUP_SIZE, BLOCK_N]

        k_start = g * GROUP_SIZE
        offs_k = k_start + tl.arange(0, GROUP_SIZE)
        mask_k = offs_k < K
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)

        acc += tl.dot(x_tile, w_tile, allow_tf32=False)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def fused_int8_linear(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    k: int,
    bias: torch.Tensor | None = None,
    block_m: int = 8,
    block_n: int = 16,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """``x @ dequantize_int8_packed(weight_packed, weight_scale, k=k).T (+
    bias)``, fused into one Triton kernel that never materializes the dense
    weight.

    ``weight_packed`` is ``[N, ceil(K/4)]`` int32; ``weight_scale`` is
    ``[N, num_groups]`` (``num_groups`` divides ``k`` evenly:
    ``group_size = k // num_groups`` must be a multiple of 4, the int8
    packing factor).

    Defaults (``block_m=8, block_n=16``) are the measured-best tile config
    on the real B70 -- see this module's own docstring for the sweep and
    the measured crossover (M<=64) past which
    :func:`prefer_fused_over_dequant` says to prefer the plain
    dequant-then-matmul fallback instead.
    """
    if weight_packed.dtype != torch.int32:
        raise TypeError(f"weight_packed must be int32, got {weight_packed.dtype}")

    m, kx = x.shape
    if kx != k:
        raise ValueError(f"x's K={kx} does not match the given k={k}")
    n, packed_cols = weight_packed.shape
    if weight_scale.shape[0] != n:
        raise ValueError(f"weight_packed/weight_scale row mismatch: {n} vs {weight_scale.shape[0]}")
    num_groups = weight_scale.shape[1]
    if k % num_groups != 0:
        raise ValueError(f"k={k} is not evenly divisible by num_groups={num_groups}")
    group_size = k // num_groups
    if group_size % 4 != 0:
        raise ValueError(f"group_size {group_size} (k={k} / num_groups={num_groups}) must be a multiple of 4")
    if group_size < 8:
        # tl.dot's own hardware minimum contraction dim is 8 (this kernel's
        # K-loop uses one full group as the dot's K each iteration) -- a
        # real checkpoint's group_size is never this small (32 or 128 in
        # every real one found so far); this only bites a synthetic test
        # fixture using an unrealistically tiny group_size.
        raise ValueError(f"group_size {group_size} is below the fused kernel's minimum of 8 (tl.dot's own hardware floor)")

    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _fused_int8_matmul_kernel[grid](
        x, weight_packed, weight_scale, out,
        m, n, k,
        x.stride(0), x.stride(1),
        weight_packed.stride(0), weight_packed.stride(1),
        weight_scale.stride(0), weight_scale.stride(1),
        out.stride(0), out.stride(1),
        GROUP_SIZE=group_size, BLOCK_M=block_m, BLOCK_N=block_n,
        BITS=_BITS, OFFSET=_OFFSET,
    )
    result = out.to(out_dtype if out_dtype is not None else x.dtype)
    if bias is not None:
        result = result + bias
    return result


def fused_int8_expert_forward(
    x: torch.Tensor,
    weight_packed_gate_up: torch.Tensor,
    weight_scale_gate_up: torch.Tensor,
    weight_packed_down: torch.Tensor,
    weight_scale_down: torch.Tensor,
    *,
    intermediate: int,
    k_gate_up: int,
    k_down: int,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """One MoE expert's SwiGLU forward (``down(silu(gate(x)) * up(x))``)
    entirely against packed compressed-tensors INT8 banks, via two
    :func:`fused_int8_linear` calls -- mirrors :func:`freetoken.kernel.
    triton.gptq_fused_linear.fused_gptq_expert_forward` / :func:`freetoken.
    kernel.triton.fused_fp8_linear.fused_fp8_expert_forward` / :func:
    `freetoken.kernel.triton.fused_mxfp4_linear.fused_mxfp4_expert_forward`'s
    structure exactly."""
    out_dtype = out_dtype if out_dtype is not None else x.dtype
    gu = fused_int8_linear(
        x, weight_packed_gate_up, weight_scale_gate_up, k=k_gate_up, out_dtype=torch.float32
    )
    gate, up = gu[:, :intermediate], gu[:, intermediate:]
    h = (torch.nn.functional.silu(gate) * up).to(out_dtype)
    return fused_int8_linear(h, weight_packed_down, weight_scale_down, k=k_down, out_dtype=out_dtype)


# Measured crossover (this module's own docstring sweep, 2048x768,
# group_size=128 on the real B70): the fused kernel wins clearly through
# M=64 and starts losing to the fallback by M=128 -- closer to GPTQ's
# crossover (#139, M<=32) than block-FP8's conservative one (#164, M<=8),
# consistent with both GPTQ and INT8 sharing a real bit-unpack cost in
# their fallback's dequant path (unlike block-FP8's plain multiply).
_FUSED_MAX_M = 64


def prefer_fused_over_dequant(m: int) -> bool:
    """Whether :func:`fused_int8_linear` (native GEMM) is expected to beat
    the plain dequant-then-matmul fallback for a batch of ``m`` rows -- see
    this module's own docstring for the measured numbers. Mirrors
    :func:`freetoken.kernel.triton.gptq_fused_linear.
    prefer_fused_over_dequant`'s role for GPTQ.
    """
    return m <= _FUSED_MAX_M
