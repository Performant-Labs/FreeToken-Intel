"""Correctness tests for block-FP8 quantize/dequantize (issue quant-xpu, #10).

These are the storage-format primitives, not a kernel -- pure torch, no
triton, no XPU dependency (fp8 tensor creation/cast works identically on CPU
and XPU per direct probe on this box). Runs in the CPU venv.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from freetoken.kernel.triton.fp8_block_linear import (
    dequantize_block_fp8,
    fp8_block_linear,
    quantize_block_fp8,
)


def test_round_trip_matches_original_within_fp8_precision():
    torch.manual_seed(0)
    w = torch.randn(256, 256) * 0.5
    q, scale = quantize_block_fp8(w, block=128)
    assert q.dtype == torch.float8_e4m3fn
    assert scale.shape == (2, 2)

    back = dequantize_block_fp8(q, scale, block=128, out_dtype=torch.float32)
    # fp8-e4m3 has ~2 decimal digits of precision; a generous but real bound.
    torch.testing.assert_close(back, w, rtol=0.1, atol=0.05)


def test_non_multiple_of_block_shape_round_trips():
    """N/K need not be exact multiples of block (the last block covers the
    remainder) -- a real checkpoint's dims are not guaranteed 128-aligned."""
    torch.manual_seed(1)
    w = torch.randn(200, 130) * 0.3
    q, scale = quantize_block_fp8(w, block=128)
    assert q.shape == (200, 130)
    assert scale.shape == (2, 2)  # ceil(200/128)=2, ceil(130/128)=2

    back = dequantize_block_fp8(q, scale, block=128, out_dtype=torch.float32)
    assert back.shape == w.shape
    torch.testing.assert_close(back, w, rtol=0.1, atol=0.05)


def test_dequantize_rejects_mismatched_scale_shape():
    w = torch.randn(256, 256)
    q, scale = quantize_block_fp8(w, block=128)
    bad_scale = scale[:1]  # wrong shape
    with pytest.raises(ValueError, match="does not match"):
        dequantize_block_fp8(q, bad_scale, block=128)


def _relative_error(out: torch.Tensor, ref: torch.Tensor) -> float:
    return ((out - ref).norm() / ref.norm().clamp_min(1e-12)).item()


def test_fp8_block_linear_matches_fp32_reference_matmul():
    """A matmul SUMS quantization error over K, and any block containing one
    large-magnitude weight sets that block's whole scale -- costing precision
    for the block's smaller elements. That is expected, real block-fp8 lossy
    behavior (the same tradeoff real production fp8 checkpoints accept), not
    a correctness bug, so this checks the aggregate (whole-tensor relative
    L2 error) rather than every individual output element."""
    torch.manual_seed(2)
    x = torch.randn(4, 256)
    w = torch.randn(64, 256) * 0.5  # [out_features, in_features]
    q, scale = quantize_block_fp8(w, block=128)

    out = fp8_block_linear(x, q, scale, block=128)
    ref = torch.nn.functional.linear(x, w)
    err = _relative_error(out, ref)
    assert err < 0.05, f"fp8-quantized matmul relative L2 error too large: {err:.4f}"


def test_fp8_block_linear_with_bias():
    torch.manual_seed(3)
    x = torch.randn(2, 128)
    w = torch.randn(16, 128) * 0.5
    bias = torch.randn(16)
    q, scale = quantize_block_fp8(w, block=128)

    out = fp8_block_linear(x, q, scale, block=128, bias=bias)
    ref = torch.nn.functional.linear(x, w, bias)
    err = _relative_error(out - bias, ref - bias)  # isolate the matmul term
    assert err < 0.05, f"fp8-quantized matmul (with bias) relative L2 error too large: {err:.4f}"
