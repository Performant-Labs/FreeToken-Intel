"""Native GPTQ-Int4 GEMM: matmul directly against packed
``qweight``/``qzeros``/``scales``, never materializing a dequantized bf16
weight tensor (issue `moe-quant-banks-native`, #139, part of epic #134).

``gptq_linear.py``'s ``gptq_linear`` (and the offload cache's
``SlotWeightAccessor``) both dequantize each expert's packed weight to
``out_dtype`` first, THEN call a plain ``torch.matmul``/``nn.functional.linear``
-- correct and RAM-safe (that's the whole point of #137), but it still pays
a real per-step compute+bandwidth cost to materialize the dense weight
before the GEMM even starts. This module fuses the two: each Triton
program dequantizes only the ``[GROUP_SIZE, BLOCK_N]`` weight tile it is
about to consume, immediately feeds it to ``tl.dot`` (Xe2's XMX tensor
cores), and never writes a dense weight tensor to memory at all.

Feasibility (issue #139 explicitly flagged this as unverified) and real
measured latency, both confirmed on the actual B70 (Xe2, Triton-XPU 3.7.2,
torch 2.13.0+xpu):

* Correctness: bit-for-bit equal (max abs diff 0.0) to
  :func:`gptq_linear.dequantize_gptq_int4_sequential_groups` + a plain
  matmul on float32 synthetic fixtures across several shapes (aligned and
  ragged K/N, single- and multi-group).
* Performance: a small tile-config sweep (``BLOCK_M`` in 1/2/4/8/16/32,
  ``BLOCK_N`` in 16/32/64) on a 2048x768, group_size=128 expert projection
  found ``BLOCK_M=8, BLOCK_N=16`` (this module's default) consistently
  near-optimal for every ``M`` up to 32, and the fused kernel wins big over
  the dequant-then-matmul fallback there -- exactly the MoE decode-step
  regime, where ``_expert_compute``'s call site already batches by
  (expert, resident slot), so ``M`` there is "how many routed tokens hit
  this one expert this step", typically small:

  ============  ============  ============
  M             fallback (ms)  fused (ms)
  ============  ============  ============
  1              0.122          0.031
  4              0.122          0.031
  8              0.122          0.033
  16             0.120          0.065
  32             0.120          0.063
  64             0.123          0.132
  128            0.129          0.251
  ============  ============  ============

  Past M~=64 the fused kernel falls behind the fallback (which is backed by
  a heavily-tuned oneDNN/oneMKL GEMM) -- so this is not a universal
  replacement; see :func:`prefer_fused_over_dequant` for the measured
  crossover this module encodes.

Design: the K-loop iterates one full quantization group per step
(``GROUP_SIZE`` must be a multiple of 8, the int4 packing factor -- true of
every real GPTQ checkpoint's ``group_size``, e.g. 128). Unlike a generic
tiled-K matmul, this means every element in one loop iteration shares
exactly one scale/zero-point row, so there is no per-element group lookup
inside the hot loop: the packed ``[GROUP_SIZE/8, BLOCK_N]`` int32 tile is
unpacked into ``[GROUP_SIZE/8, 8, BLOCK_N]`` via bitwise shift+mask, then
reshaped to ``[GROUP_SIZE, BLOCK_N]`` (this reshape's row-major (r, b) ->
r*8+b merge exactly matches ``qweight``'s own bit layout -- byte ``b`` of
packed row ``r`` holds logical row ``k = r*8+b``, the same convention
``gptq_linear._unpack_int32`` decodes), dequantized in one elementwise op,
and fed straight into ``tl.dot`` against the correspondingly-contiguous
``x[:, k_start:k_start+GROUP_SIZE]`` slice.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

_BITS = 4
_P = 32 // _BITS  # 8: int4 values packed per int32 word


@triton.jit
def _fused_gptq_matmul_kernel(
    x_ptr, qweight_ptr, qzeros_ptr, scales_ptr, out_ptr,
    M, N, K, G,
    stride_xm, stride_xk,
    stride_qwk, stride_qwn,
    stride_qzg, stride_qzn,
    stride_sg, stride_sn,
    stride_om, stride_on,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BITS: tl.constexpr, P: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    NUM_PACK_ROWS: tl.constexpr = GROUP_SIZE // P  # qweight rows spanned by one group
    offs_pr = tl.arange(0, NUM_PACK_ROWS)
    shifts = tl.arange(0, P) * BITS

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for g in range(G):
        k_pack_start = g * NUM_PACK_ROWS
        qw_ptrs = qweight_ptr + (k_pack_start + offs_pr[:, None]) * stride_qwk + offs_n[None, :] * stride_qwn
        packed = tl.load(qw_ptrs, mask=mask_n[None, :], other=0)  # [NUM_PACK_ROWS, BLOCK_N] int32

        # unpack -> [NUM_PACK_ROWS, P, BLOCK_N], then merge (r, b) -> k = r*P+b
        # (matches qweight's own bit layout: byte b of pack-row r holds k = r*P+b).
        w_bits = (packed[:, None, :] >> shifts[None, :, None]) & 0xF
        w_tile = tl.reshape(w_bits, (GROUP_SIZE, BLOCK_N)).to(tl.float32)

        n_pack = offs_n // P
        n_shift = (offs_n % P) * BITS
        qz_ptrs = qzeros_ptr + g * stride_qzg + n_pack * stride_qzn
        qz_word = tl.load(qz_ptrs, mask=mask_n, other=0)
        # AutoGPTQ's stored-minus-one zero-point convention (see gptq_linear.py's module docstring).
        zero = ((qz_word >> n_shift) & 0xF).to(tl.float32) + 1.0  # [BLOCK_N]

        s_ptrs = scales_ptr + g * stride_sg + offs_n * stride_sn
        scale = tl.load(s_ptrs, mask=mask_n, other=0.0).to(tl.float32)  # [BLOCK_N]

        deq_tile = (w_tile - zero[None, :]) * scale[None, :]  # [GROUP_SIZE, BLOCK_N]

        offs_k = g * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)  # [BLOCK_M, GROUP_SIZE]

        acc += tl.dot(x_tile, deq_tile, allow_tf32=False)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def fused_gptq_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int,
    bias: torch.Tensor | None = None,
    block_m: int = 8,
    block_n: int = 16,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """``x @ dequantize_gptq_int4_sequential_groups(...).T (+ bias)``, fused
    into one Triton kernel that never materializes the dense weight.

    Only implements the ``desc_act=False`` case (sequential groups,
    ``g_idx[k] = k // group_size``) -- the same restriction
    :func:`gptq_linear.dequantize_gptq_int4_sequential_groups` has, and the
    only case any real checkpoint this project has loaded uses.

    ``group_size`` must be a multiple of 8 (the int4 packing factor) -- true
    of every real GPTQ checkpoint's ``group_size`` (128 is standard).

    Defaults (``block_m=8, block_n=16``) are the measured-best tile config
    for M<=32 on the real B70 -- see this module's own docstring for the
    sweep. Above that M, :func:`prefer_fused_over_dequant` says to prefer
    the plain dequant-then-matmul fallback (:func:`gptq_linear.gptq_linear`)
    instead; callers driving both batch regimes should pick per call via
    that helper, not hardcode one path.
    """
    if group_size % _P != 0:
        raise ValueError(f"group_size {group_size} must be a multiple of {_P} (the int4 packing factor)")
    if qweight.dtype != torch.int32:
        raise TypeError(f"qweight must be int32, got {qweight.dtype}")
    if qzeros.dtype != torch.int32:
        raise TypeError(f"qzeros must be int32, got {qzeros.dtype}")

    m, k = x.shape
    k_packed, n = qweight.shape
    if k_packed * _P != k:
        raise ValueError(f"x's K={k} does not match qweight's implied K={k_packed * _P}")
    if scales.shape[1] != n or qzeros.shape[1] * _P != n:
        raise ValueError(
            f"qweight/qzeros/scales shape mismatch: qweight={tuple(qweight.shape)}, "
            f"qzeros={tuple(qzeros.shape)}, scales={tuple(scales.shape)}"
        )
    g = -(-k // group_size)

    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _fused_gptq_matmul_kernel[grid](
        x, qweight, qzeros, scales, out,
        m, n, k, g,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0), out.stride(1),
        GROUP_SIZE=group_size,
        BLOCK_M=block_m, BLOCK_N=block_n,
        BITS=_BITS, P=_P,
    )
    result = out.to(out_dtype if out_dtype is not None else x.dtype)
    if bias is not None:
        result = result + bias
    return result


def fused_gptq_expert_forward(
    x: torch.Tensor,
    qweight_gate_up: torch.Tensor,
    qzeros_gate_up: torch.Tensor,
    scales_gate_up: torch.Tensor,
    qweight_down: torch.Tensor,
    qzeros_down: torch.Tensor,
    scales_down: torch.Tensor,
    *,
    group_size: int,
    intermediate: int,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """One MoE expert's SwiGLU forward (``down(silu(gate(x)) * up(x))``)
    entirely against packed GPTQ banks, via two :func:`fused_gptq_linear`
    calls -- never materializing a dense gate/up/down weight (issue #139).

    Same math as the plain-tensor ``_expert_compute`` helper duplicated in
    ``qwen3_moe``/``qwen3_5_moe`` (``down(silu(gate(x)) * up(x))``, gate/up
    fused as one bank whose output columns split ``[0:intermediate]`` /
    ``[intermediate:2*intermediate]``), but each of the two GEMMs runs
    fused against the packed tensors from
    :meth:`freetoken.moe.offload_cache.SlotWeightAccessor.get_gptq_packed`
    instead of a pre-dequantized dense weight.
    """
    out_dtype = out_dtype if out_dtype is not None else x.dtype
    gu = fused_gptq_linear(
        x, qweight_gate_up, qzeros_gate_up, scales_gate_up, group_size=group_size, out_dtype=torch.float32
    )
    gate, up = gu[:, :intermediate], gu[:, intermediate:]
    h = (torch.nn.functional.silu(gate) * up).to(out_dtype)
    return fused_gptq_linear(h, qweight_down, qzeros_down, scales_down, group_size=group_size, out_dtype=out_dtype)


# Measured crossover (this module's own docstring sweep, 2048x768,
# group_size=128 on the real B70): the fused kernel wins clearly through
# M=32 and starts losing to the fallback by M=64. Not re-measured per
# checkpoint shape -- a real per-(N, K, group_size) profile, mirroring
# freetoken.engine.cache_budget's own measured-not-guessed philosophy,
# would replace this fixed threshold if a real checkpoint's numbers ever
# disagree with it.
_FUSED_MAX_M = 32


def prefer_fused_over_dequant(m: int) -> bool:
    """Whether :func:`fused_gptq_linear` (native GEMM) is expected to beat
    :func:`gptq_linear.gptq_linear` (dequant-then-matmul) for a batch of
    ``m`` rows -- see this module's own docstring for the measured numbers
    this threshold is drawn from.

    Not wired into any forward path yet (issue #139): the offload cache's
    ``SlotWeightAccessor`` currently always dequantizes to a dense tensor
    first (see its own docstring), so a caller wanting the fused kernel
    today must bypass it and call :func:`fused_gptq_linear` directly against
    the packed bank views. Exposed here so that future wiring has a real,
    measured dispatch rule to start from instead of a guess.
    """
    return m <= _FUSED_MAX_M
