"""Native block-FP8 GEMM: matmul directly against packed
``weight_fp8``/``weight_scale_inv``, never materializing a dequantized bf16
weight tensor (issue `moe-quant-banks-native-multi`, #163, extending #139's
GPTQ approach to the other three offload-bank formats).

``fp8_block_linear.py``'s ``dequantize_block_fp8`` (and the offload cache's
``SlotWeightAccessor``) both dequantize an expert's packed block-FP8 weight
to a dense tensor first, THEN run a plain matmul. This module fuses the
two: each Triton program dequantizes only the ``[QBLOCK, BLOCK_N]`` weight
tile it is about to consume and feeds it straight into ``tl.dot``.

Feasibility and real measured latency, both confirmed on the actual B70
(Xe2, Triton-XPU 3.7.2, torch 2.13.0+xpu):

* Correctness: matches ``dequantize_block_fp8`` + a plain matmul to
  float32 rounding tolerance (max abs diff ~3e-4 on a 2048x768 projection)
  across aligned and ragged M/K/N shapes.
* Performance: block-FP8's existing dequant-then-matmul fallback is
  ALREADY fast -- ``dequantize_block_fp8`` is a single reshape+broadcast-
  multiply (no bit-unpacking, unlike GPTQ/INT8), so the fallback's own
  cost is small and roughly flat across M (~0.05ms on a 2048x768
  projection, every M tried). The fused kernel only wins clearly at M=1
  (~0.036ms vs ~0.053ms, ~32% faster -- still real, and it is exactly the
  M every offload forward call site actually uses, see
  ``fused_gptq_linear``'s own docstring for why), roughly ties through
  M=8, and loses beyond M=16. This is a much smaller margin than GPTQ's
  ~4x win (#139) -- block-FP8 was never as expensive to dequantize in the
  first place.

Design mirrors ``gptq_fused_linear``: the K-loop iterates one full
QBLOCK=128 quantization block per step, so (unlike GPTQ's group-vs-column
scale) every element in one loop iteration shares exactly ONE scalar scale
(block-FP8 quantizes both the N and K axes into ``block``-sized tiles, and
this kernel's ``BLOCK_N`` is required to divide ``block`` evenly and
started at a multiple of it, so one program's N-tile never crosses an
N-axis quantization block boundary either) -- simpler than GPTQ's
per-output-channel scale vector, no group/column indexing needed inside
the hot loop at all.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

_QBLOCK = 128  # block-FP8's quantization block size (DeepSeek-V3 / sglang / vLLM convention)


@triton.jit
def _fused_fp8_matmul_kernel(
    x_ptr, w_ptr, s_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_sn, stride_sk,
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
    n_block_idx = (pid_n * BLOCK_N) // QBLOCK  # BLOCK_N divides QBLOCK -> one N-tile, one N-block

    num_k_blocks = tl.cdiv(K, QBLOCK)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kb in range(num_k_blocks):
        k_start = kb * QBLOCK
        offs_k = k_start + tl.arange(0, QBLOCK)
        mask_k = offs_k < K

        # weight_fp8 is [N, K] (nn.Linear's own out/in orientation) -- loaded
        # here as a [QBLOCK, BLOCK_N] (k-major, n-minor) tile so it feeds
        # tl.dot directly as x[BLOCK_M, QBLOCK] @ w[QBLOCK, BLOCK_N], no
        # separate transpose.
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
        w_tile = tl.load(w_ptrs, mask=mask_n[None, :] & mask_k[:, None], other=0.0).to(tl.float32)

        scale = tl.load(s_ptr + n_block_idx * stride_sn + kb * stride_sk).to(tl.float32)
        w_deq = w_tile * scale

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)

        acc += tl.dot(x_tile, w_deq, allow_tf32=False)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def fused_fp8_linear(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    block: int = _QBLOCK,
    block_m: int = 1,
    block_n: int = 16,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """``x @ dequantize_block_fp8(weight_fp8, weight_scale_inv).T (+ bias)``,
    fused into one Triton kernel that never materializes the dense weight.

    ``weight_fp8`` is ``[N, K]`` (``nn.Linear``'s own out/in orientation);
    ``weight_scale_inv`` is ``[ceil(N/block), ceil(K/block)]``. ``block_n``
    must divide ``block`` evenly (so one N-tile never crosses a quantization
    block boundary) -- true of every default/measured-best config this
    module ships.

    Defaults (``block_m=1, block_n=16``) are the measured-best config for
    M=1 on the real B70 (the batch size every real offload forward call
    site actually uses) -- see this module's own docstring for the numbers
    and the (much smaller than GPTQ's) crossover past which the plain
    dequant-then-matmul fallback wins instead.
    """
    if block % block_n != 0:
        raise ValueError(f"block_n {block_n} must divide block {block} evenly")
    if weight_fp8.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise TypeError(f"weight_fp8 must be an fp8 dtype, got {weight_fp8.dtype}")

    m, k = x.shape
    n, k2 = weight_fp8.shape
    if k != k2:
        raise ValueError(f"x's K={k} does not match weight_fp8's K={k2}")
    expected_scale_shape = (-(-n // block), -(-k // block))
    if tuple(weight_scale_inv.shape) != expected_scale_shape:
        raise ValueError(
            f"weight_scale_inv shape {tuple(weight_scale_inv.shape)} does not match "
            f"the expected {expected_scale_shape} for weight_fp8 shape {(n, k)} at block={block}"
        )

    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _fused_fp8_matmul_kernel[grid](
        x, weight_fp8, weight_scale_inv, out,
        m, n, k,
        x.stride(0), x.stride(1),
        weight_fp8.stride(0), weight_fp8.stride(1),
        weight_scale_inv.stride(0), weight_scale_inv.stride(1),
        out.stride(0), out.stride(1),
        QBLOCK=block, BLOCK_M=block_m, BLOCK_N=block_n,
    )
    result = out.to(out_dtype if out_dtype is not None else x.dtype)
    if bias is not None:
        result = result + bias
    return result


def fused_fp8_expert_forward(
    x: torch.Tensor,
    weight_fp8_gate_up: torch.Tensor,
    weight_scale_inv_gate_up: torch.Tensor,
    weight_fp8_down: torch.Tensor,
    weight_scale_inv_down: torch.Tensor,
    *,
    intermediate: int,
    block: int = _QBLOCK,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """One MoE expert's SwiGLU forward (``down(silu(gate(x)) * up(x))``)
    entirely against packed block-FP8 banks, via two :func:`fused_fp8_linear`
    calls -- mirrors :func:`freetoken.kernel.triton.gptq_fused_linear.
    fused_gptq_expert_forward`'s structure exactly (gate/up fused as one
    bank whose output rows split ``[0:intermediate]`` / ``[intermediate:
    2*intermediate]``)."""
    out_dtype = out_dtype if out_dtype is not None else x.dtype
    gu = fused_fp8_linear(x, weight_fp8_gate_up, weight_scale_inv_gate_up, block=block, out_dtype=torch.float32)
    gate, up = gu[:, :intermediate], gu[:, intermediate:]
    h = (torch.nn.functional.silu(gate) * up).to(out_dtype)
    return fused_fp8_linear(h, weight_fp8_down, weight_scale_inv_down, block=block, out_dtype=out_dtype)


# Measured crossover (this module's own docstring sweep, 2048x768 on the
# real B70): the fused kernel clearly wins only at M=1, roughly ties
# through M=8, and loses beyond M=16 -- a much smaller/more conservative
# window than GPTQ's (#139's own _FUSED_MAX_M=32), matching block-FP8's
# already-cheap dequant (a single reshape+multiply, no bit-unpacking).
_FUSED_MAX_M = 8


def prefer_fused_over_dequant(m: int) -> bool:
    """Whether :func:`fused_fp8_linear` (native GEMM) is expected to beat
    the plain dequant-then-matmul fallback for a batch of ``m`` rows -- see
    this module's own docstring for the measured numbers this threshold is
    drawn from. Mirrors :func:`freetoken.kernel.triton.gptq_fused_linear.
    prefer_fused_over_dequant`'s role for GPTQ.
    """
    return m <= _FUSED_MAX_M
