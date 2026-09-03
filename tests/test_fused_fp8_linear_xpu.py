"""``fused_fp8_linear``: native block-FP8 GEMM against packed tensors,
never materializing a dense dequantized weight (issue `moe-quant-banks-
native-multi`, #163, extending #139's GPTQ approach to block-FP8).

Needs a real Triton-XPU compile + real fp8 tensor loads -- no meaningful
CPU-only synthetic-fixture version exists, the same reasoning as
test_gptq_fused_linear_xpu.py. ``xpu``-marked: deselected on a torch-free /
no-XPU box (see ``conftest.py``).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

XPU = pytest.mark.skipif(not torch.xpu.is_available(), reason="no XPU available")

from freetoken.kernel.triton.fp8_block_linear import dequantize_block_fp8, quantize_block_fp8
from freetoken.kernel.triton.fused_fp8_linear import fused_fp8_linear

DEVICE = "xpu"


@XPU
@pytest.mark.xpu
@pytest.mark.parametrize(
    "m,k,n,block",
    [
        (8, 256, 384, 128),  # aligned, single-block-per-tile-row K
        (5, 300, 213, 128),  # ragged M/K/N, real checkpoint block size
        (32, 2048, 768, 128),  # realistic MoE expert projection shape
        (1, 128, 128, 128),  # exactly one quant block
    ],
)
def test_fused_matches_dequant_then_matmul(m, k, n, block):
    torch.manual_seed(0)
    w = torch.randn(n, k, device=DEVICE)
    w_fp8, scale = quantize_block_fp8(w, block=block)
    x = torch.randn(m, k, device=DEVICE, dtype=torch.float32)

    ref_w = dequantize_block_fp8(w_fp8, scale, block=block, out_dtype=torch.float32)
    ref_out = x @ ref_w.T

    fused_out = fused_fp8_linear(x, w_fp8, scale, block=block, out_dtype=torch.float32)

    torch.testing.assert_close(fused_out, ref_out, atol=2e-3, rtol=2e-2)


@XPU
@pytest.mark.xpu
def test_fused_applies_bias():
    torch.manual_seed(0)
    m, k, n, block = 4, 128, 32, 128
    w = torch.randn(n, k, device=DEVICE)
    w_fp8, scale = quantize_block_fp8(w, block=block)
    x = torch.randn(m, k, device=DEVICE, dtype=torch.float32)
    bias = torch.randn(n, device=DEVICE, dtype=torch.float32)

    ref_w = dequantize_block_fp8(w_fp8, scale, block=block, out_dtype=torch.float32)
    ref_out = x @ ref_w.T + bias

    fused_out = fused_fp8_linear(x, w_fp8, scale, block=block, bias=bias, out_dtype=torch.float32)

    torch.testing.assert_close(fused_out, ref_out, atol=2e-3, rtol=2e-2)


@XPU
@pytest.mark.xpu
def test_fused_output_dtype_defaults_to_input_dtype():
    m, k, n, block = 4, 128, 32, 128
    w = torch.randn(n, k, device=DEVICE)
    w_fp8, scale = quantize_block_fp8(w, block=block)
    x = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)

    out = fused_fp8_linear(x, w_fp8, scale, block=block)

    assert out.dtype == torch.bfloat16


@XPU
@pytest.mark.xpu
def test_block_n_not_dividing_block_raises():
    m, k, n, block = 4, 128, 32, 128
    w = torch.randn(n, k, device=DEVICE)
    w_fp8, scale = quantize_block_fp8(w, block=block)
    x = torch.randn(m, k, device=DEVICE, dtype=torch.float32)
    with pytest.raises(ValueError, match="must divide"):
        fused_fp8_linear(x, w_fp8, scale, block=block, block_n=24)
