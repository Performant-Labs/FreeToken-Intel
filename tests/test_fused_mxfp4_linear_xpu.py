"""``fused_mxfp4_linear``: native MXFP4 GEMM against packed tensors, never
materializing a dense dequantized weight (issue `moe-quant-banks-native-
multi`, #163, extending #139's GPTQ approach to MXFP4).

Needs a real Triton-XPU compile -- no meaningful CPU-only synthetic-fixture
version exists, the same reasoning as test_gptq_fused_linear_xpu.py /
test_fused_fp8_linear_xpu.py. ``xpu``-marked: deselected on a torch-free /
no-XPU box (see ``conftest.py``).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

XPU = pytest.mark.skipif(not torch.xpu.is_available(), reason="no XPU available")

from freetoken.kernel.triton.fused_mxfp4_linear import fused_mxfp4_linear
from freetoken.kernel.triton.mxfp4_linear import dequantize_mxfp4_blocks, quantize_mxfp4_blocks

DEVICE = "xpu"


@XPU
@pytest.mark.xpu
@pytest.mark.parametrize(
    "m,k,n",
    [
        (8, 64, 32),  # aligned, small
        (5, 96, 213),  # ragged M/N, multiple K-blocks
        (32, 2048, 768),  # realistic MoE expert projection shape
        (1, 32, 32),  # exactly one quant block
    ],
)
def test_fused_matches_dequant_then_matmul(m, k, n):
    torch.manual_seed(0)
    w = torch.randn(n, k, device=DEVICE)
    blocks, scales = quantize_mxfp4_blocks(w)
    x = torch.randn(m, k, device=DEVICE, dtype=torch.float32)

    ref_w = dequantize_mxfp4_blocks(blocks, scales, out_dtype=torch.float32)
    ref_out = x @ ref_w.T

    fused_out = fused_mxfp4_linear(x, blocks, scales, out_dtype=torch.float32)

    torch.testing.assert_close(fused_out, ref_out, atol=2e-3, rtol=2e-2)


@XPU
@pytest.mark.xpu
def test_fused_applies_bias():
    torch.manual_seed(0)
    m, k, n = 4, 64, 32
    w = torch.randn(n, k, device=DEVICE)
    blocks, scales = quantize_mxfp4_blocks(w)
    x = torch.randn(m, k, device=DEVICE, dtype=torch.float32)
    bias = torch.randn(n, device=DEVICE, dtype=torch.float32)

    ref_w = dequantize_mxfp4_blocks(blocks, scales, out_dtype=torch.float32)
    ref_out = x @ ref_w.T + bias

    fused_out = fused_mxfp4_linear(x, blocks, scales, bias=bias, out_dtype=torch.float32)

    torch.testing.assert_close(fused_out, ref_out, atol=2e-3, rtol=2e-2)


@XPU
@pytest.mark.xpu
def test_fused_output_dtype_defaults_to_input_dtype():
    m, k, n = 4, 64, 32
    w = torch.randn(n, k, device=DEVICE)
    blocks, scales = quantize_mxfp4_blocks(w)
    x = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)

    out = fused_mxfp4_linear(x, blocks, scales)

    assert out.dtype == torch.bfloat16


@XPU
@pytest.mark.xpu
def test_wrong_blocks_dtype_raises():
    m, k, n = 4, 64, 32
    x = torch.randn(m, k, device=DEVICE, dtype=torch.float32)
    blocks = torch.zeros(n, k // 32, 16, device=DEVICE, dtype=torch.int32)
    scales = torch.zeros(n, k // 32, device=DEVICE, dtype=torch.uint8)
    with pytest.raises(TypeError, match="uint8"):
        fused_mxfp4_linear(x, blocks, scales)
