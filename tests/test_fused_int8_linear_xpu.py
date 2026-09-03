"""``fused_int8_linear``: native compressed-tensors pack-quantized INT8
GEMM against packed tensors, never materializing a dense dequantized
weight (issue `moe-quant-banks-native-multi`, #163, extending #139's GPTQ
approach to INT8).

Needs a real Triton-XPU compile -- no meaningful CPU-only synthetic-
fixture version exists, the same reasoning as test_gptq_fused_linear_xpu.py
/ test_fused_fp8_linear_xpu.py / test_fused_mxfp4_linear_xpu.py.
``xpu``-marked: deselected on a torch-free / no-XPU box (see
``conftest.py``).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

XPU = pytest.mark.skipif(not torch.xpu.is_available(), reason="no XPU available")

from freetoken.kernel.triton.fused_int8_linear import fused_int8_linear
from freetoken.kernel.triton.int8_packed_linear import dequantize_int8_packed

DEVICE = "xpu"


def _pack_int8(codes: torch.Tensor) -> torch.Tensor:
    n, k = codes.shape
    codes64 = (codes.to(torch.int64) + 128).reshape(n, k // 4, 4)
    word = sum(codes64[:, :, i] << (8 * i) for i in range(4))
    word = torch.where(word >= (1 << 31), word - (1 << 32), word)
    return word.to(torch.int32)


def _random_fixture(n: int, k: int, group_size: int, *, seed: int):
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(-127, 128, (n, k), generator=g, dtype=torch.int8).to(DEVICE)
    num_groups = k // group_size
    scale = (torch.rand(n, num_groups, generator=g) * 0.1 + 0.01).to(torch.float32).to(DEVICE)
    return _pack_int8(codes), scale


@XPU
@pytest.mark.xpu
@pytest.mark.parametrize(
    "m,k,n,group_size",
    [
        (8, 64, 32, 32),  # single group (K == group_size)
        (5, 384, 504, 128),  # ragged M/N, real checkpoint group_size, multiple groups
        (32, 2048, 768, 128),  # realistic MoE expert projection shape
    ],
)
def test_fused_matches_dequant_then_matmul(m, k, n, group_size):
    torch.manual_seed(0)
    weight_packed, weight_scale = _random_fixture(n, k, group_size, seed=100 + k)
    x = torch.randn(m, k, device=DEVICE, dtype=torch.float32)

    ref_w = dequantize_int8_packed(weight_packed, weight_scale, k=k, out_dtype=torch.float32)
    ref_out = x @ ref_w.T

    fused_out = fused_int8_linear(x, weight_packed, weight_scale, k=k, out_dtype=torch.float32)

    torch.testing.assert_close(fused_out, ref_out, atol=2e-3, rtol=2e-2)


@XPU
@pytest.mark.xpu
def test_fused_applies_bias():
    torch.manual_seed(0)
    m, k, n, group_size = 4, 128, 32, 128
    weight_packed, weight_scale = _random_fixture(n, k, group_size, seed=7)
    x = torch.randn(m, k, device=DEVICE, dtype=torch.float32)
    bias = torch.randn(n, device=DEVICE, dtype=torch.float32)

    ref_w = dequantize_int8_packed(weight_packed, weight_scale, k=k, out_dtype=torch.float32)
    ref_out = x @ ref_w.T + bias

    fused_out = fused_int8_linear(x, weight_packed, weight_scale, k=k, bias=bias, out_dtype=torch.float32)

    torch.testing.assert_close(fused_out, ref_out, atol=2e-3, rtol=2e-2)


@XPU
@pytest.mark.xpu
def test_fused_output_dtype_defaults_to_input_dtype():
    m, k, n, group_size = 4, 128, 32, 128
    weight_packed, weight_scale = _random_fixture(n, k, group_size, seed=11)
    x = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)

    out = fused_int8_linear(x, weight_packed, weight_scale, k=k)

    assert out.dtype == torch.bfloat16


@XPU
@pytest.mark.xpu
def test_group_size_below_tl_dot_minimum_raises():
    weight_packed, weight_scale = _random_fixture(16, 16, 4, seed=2)  # group_size=4 < tl.dot's minimum of 8
    x = torch.randn(4, 16, device=DEVICE, dtype=torch.float32)
    with pytest.raises(ValueError, match="below the fused kernel's minimum"):
        fused_int8_linear(x, weight_packed, weight_scale, k=16)


@XPU
@pytest.mark.xpu
def test_group_size_not_multiple_of_pack_factor_raises():
    weight_packed, weight_scale = _random_fixture(32, 64, 32, seed=1)
    # 5 groups over K=64 -> group_size 64/5, not an integer -- exercised via
    # a scale tensor with a non-dividing group count instead.
    bad_scale = torch.zeros(32, 5, device=DEVICE, dtype=torch.float32)
    x = torch.randn(4, 64, device=DEVICE, dtype=torch.float32)
    with pytest.raises(ValueError, match="not evenly divisible"):
        fused_int8_linear(x, weight_packed, bad_scale, k=64)
