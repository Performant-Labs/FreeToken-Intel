"""Correctness tests for MXFP4 quantize/dequantize (issue quant-xpu, #10).

Pure torch, no XPU dependency, CPU-testable -- same pattern as
test_fp8_block_quant.py.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from freetoken.kernel.triton.mxfp4_linear import (
    dequantize_mxfp4_blocks,
    mxfp4_linear,
    quantize_mxfp4_blocks,
)


def _relative_error(out: torch.Tensor, ref: torch.Tensor) -> float:
    return ((out - ref).norm() / ref.norm().clamp_min(1e-12)).item()


def test_dequantize_matches_upstreams_known_encoding():
    """Hand-built blocks/scales, independent of quantize_mxfp4_blocks, so this
    checks dequantize_mxfp4_blocks's bit layout against the spec directly
    (not just self-consistency with our own quantizer)."""
    # Byte 0x10 = nibbles (0, 1) -> LUT[0]=0.0, LUT[1]=0.5. Byte 0x32 = nibbles
    # (2, 3) -> LUT[2]=1.0, LUT[3]=1.5. Remaining 14 bytes all zero -> LUT[0].
    block = torch.zeros(16, dtype=torch.uint8)
    block[0] = 0x10  # low nibble 0 -> 0.0, high nibble 1 -> 0.5
    block[1] = 0x32  # low nibble 2 -> 1.0, high nibble 3 -> 1.5
    scale = torch.tensor(127, dtype=torch.uint8)  # 2**(127-127) == 1.0

    out = dequantize_mxfp4_blocks(block.unsqueeze(0), scale.unsqueeze(0), out_dtype=torch.float32)
    expected = torch.zeros(32)
    expected[0], expected[1], expected[2], expected[3] = 0.0, 0.5, 1.0, 1.5
    torch.testing.assert_close(out.squeeze(0), expected)


def test_scale_applies_as_power_of_two():
    block = torch.zeros(16, dtype=torch.uint8)
    block[0] = 0x02  # low nibble 2 -> LUT[2] = 1.0
    scale = torch.tensor(128, dtype=torch.uint8)  # 2**(128-127) == 2.0

    out = dequantize_mxfp4_blocks(block.unsqueeze(0), scale.unsqueeze(0), out_dtype=torch.float32)
    assert out[0].item() == pytest.approx(2.0)


def test_dequantize_rejects_wrong_block_pack_dim():
    with pytest.raises(ValueError, match="16 bytes"):
        dequantize_mxfp4_blocks(
            torch.zeros(1, 15, dtype=torch.uint8), torch.zeros(1, dtype=torch.uint8)
        )


def test_dequantize_rejects_non_uint8():
    with pytest.raises(TypeError, match="uint8"):
        dequantize_mxfp4_blocks(torch.zeros(1, 16, dtype=torch.int32), torch.zeros(1, dtype=torch.uint8))


def test_dequantize_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="mismatch"):
        dequantize_mxfp4_blocks(torch.zeros(2, 16, dtype=torch.uint8), torch.zeros(3, dtype=torch.uint8))


def test_quantize_round_trip_within_mxfp4_precision():
    torch.manual_seed(0)
    w = torch.randn(64, 64) * 0.5  # K=64 -> 2 blocks of 32 per row
    blocks, scales = quantize_mxfp4_blocks(w)
    assert blocks.dtype == torch.uint8
    assert blocks.shape == (64, 2, 16)
    assert scales.shape == (64, 2)

    back = dequantize_mxfp4_blocks(blocks, scales, out_dtype=torch.float32)
    err = _relative_error(back, w)
    # 4-bit (2-mantissa-bit) precision is much coarser than fp8 -- a generous
    # but real whole-tensor bound, not per-element.
    assert err < 0.2, f"MXFP4 round-trip relative L2 error too large: {err:.4f}"


def test_quantize_rejects_k_not_a_multiple_of_32():
    with pytest.raises(ValueError, match="multiple of 32"):
        quantize_mxfp4_blocks(torch.randn(4, 33))


def test_quantize_handles_an_all_zero_block():
    w = torch.zeros(1, 32)
    blocks, scales = quantize_mxfp4_blocks(w)
    back = dequantize_mxfp4_blocks(blocks, scales, out_dtype=torch.float32)
    torch.testing.assert_close(back, w)


def test_mxfp4_linear_matches_fp32_reference_matmul():
    torch.manual_seed(1)
    x = torch.randn(4, 64)
    w = torch.randn(16, 64) * 0.5  # [out_features, in_features], K=64
    blocks, scales = quantize_mxfp4_blocks(w)

    out = mxfp4_linear(x, blocks, scales)
    ref = torch.nn.functional.linear(x, w)
    err = _relative_error(out, ref)
    assert err < 0.2, f"MXFP4-quantized matmul relative L2 error too large: {err:.4f}"


def test_mxfp4_linear_with_bias():
    torch.manual_seed(2)
    x = torch.randn(2, 32)
    w = torch.randn(8, 32) * 0.5
    bias = torch.randn(8)
    blocks, scales = quantize_mxfp4_blocks(w)

    out = mxfp4_linear(x, blocks, scales, bias=bias)
    ref = torch.nn.functional.linear(x, w, bias)
    err = _relative_error(out - bias, ref - bias)
    assert err < 0.2, f"MXFP4-quantized matmul (with bias) relative L2 error too large: {err:.4f}"
