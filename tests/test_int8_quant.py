"""Correctness tests for per-channel symmetric INT8 quantization (issue
quant-xpu, #10). Pure torch, no XPU dependency, CPU-testable -- same
pattern as test_fp8_block_quant.py / test_mxfp4_quant.py.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from freetoken.kernel.triton.int8_linear import (
    dequantize_int8_channel,
    int8_linear,
    quantize_int8_channel,
)


def _relative_error(out: torch.Tensor, ref: torch.Tensor) -> float:
    return ((out - ref).norm() / ref.norm().clamp_min(1e-12)).item()


def test_round_trip_matches_original_within_int8_precision():
    torch.manual_seed(0)
    w = torch.randn(64, 128) * 0.5
    q, scale = quantize_int8_channel(w)
    assert q.dtype == torch.int8
    assert scale.shape == (64,)

    back = dequantize_int8_channel(q, scale, out_dtype=torch.float32)
    err = _relative_error(back, w)
    assert err < 0.02, f"int8 round-trip relative L2 error too large: {err:.4f}"


def test_scale_is_per_row_not_global():
    """A row with a much larger magnitude must not blow out a smaller row's
    precision -- the whole point of per-channel over a single global scale."""
    w = torch.zeros(2, 128)
    w[0] = torch.full((128,), 100.0)  # huge row
    w[1] = torch.full((128,), 0.01)  # tiny row
    q, scale = quantize_int8_channel(w)
    back = dequantize_int8_channel(q, scale, out_dtype=torch.float32)
    # The tiny row must still be represented (not flattened to all-zero by a
    # global scale sized for the huge row).
    assert back[1].abs().max().item() > 0.0
    torch.testing.assert_close(back[0], w[0], rtol=0.02, atol=0.02)
    torch.testing.assert_close(back[1], w[1], rtol=0.02, atol=1e-4)


def test_dequantize_rejects_non_int8():
    with pytest.raises(TypeError, match="int8"):
        dequantize_int8_channel(torch.zeros(2, 4, dtype=torch.int32), torch.ones(2))


def test_dequantize_rejects_mismatched_scale_shape():
    q = torch.zeros(4, 8, dtype=torch.int8)
    bad_scale = torch.ones(3)
    with pytest.raises(ValueError, match="does not match"):
        dequantize_int8_channel(q, bad_scale)


def test_all_zero_row_round_trips_to_zero():
    w = torch.zeros(1, 16)
    q, scale = quantize_int8_channel(w)
    back = dequantize_int8_channel(q, scale, out_dtype=torch.float32)
    torch.testing.assert_close(back, w)


def test_quantized_values_stay_within_int8_range():
    torch.manual_seed(1)
    w = torch.randn(8, 32) * 10.0
    q, _ = quantize_int8_channel(w)
    assert q.min().item() >= -127
    assert q.max().item() <= 127


def test_int8_linear_matches_fp32_reference_matmul():
    torch.manual_seed(2)
    x = torch.randn(4, 128)
    w = torch.randn(32, 128) * 0.5
    q, scale = quantize_int8_channel(w)

    out = int8_linear(x, q, scale)
    ref = torch.nn.functional.linear(x, w)
    err = _relative_error(out, ref)
    assert err < 0.02, f"int8-quantized matmul relative L2 error too large: {err:.4f}"


def test_int8_linear_with_bias():
    torch.manual_seed(3)
    x = torch.randn(2, 64)
    w = torch.randn(8, 64) * 0.5
    bias = torch.randn(8)
    q, scale = quantize_int8_channel(w)

    out = int8_linear(x, q, scale, bias=bias)
    ref = torch.nn.functional.linear(x, w, bias)
    err = _relative_error(out - bias, ref - bias)
    assert err < 0.02, f"int8-quantized matmul (with bias) relative L2 error too large: {err:.4f}"
