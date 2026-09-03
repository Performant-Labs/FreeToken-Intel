"""``fused_gptq_linear``: native GPTQ-Int4 GEMM against packed tensors,
never materializing a dense dequantized weight (issue `moe-quant-banks-
native`, #139, part of epic #134).

Needs a real Triton-XPU compile + real Xe2 tensor-core dispatch (``tl.dot``)
-- no meaningful CPU-only synthetic-fixture version exists, unlike the
dequant-only kernels (``gptq_linear.py`` etc.), which are plain tensor ops
runnable anywhere. ``xpu``-marked: deselected on a torch-free / no-XPU box
(see ``conftest.py``).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

XPU = pytest.mark.skipif(not torch.xpu.is_available(), reason="no XPU available")

from freetoken.kernel.triton.gptq_fused_linear import fused_gptq_linear
from freetoken.kernel.triton.gptq_linear import dequantize_gptq_int4_sequential_groups

DEVICE = "xpu"
P = 8  # int4 values packed per int32 word


def _random_gptq_fixture(k: int, n: int, group_size: int, *, seed: int):
    g = torch.Generator().manual_seed(seed)
    k_packed = k // P
    num_groups = k // group_size
    qweight = torch.randint(0, 2**31 - 1, (k_packed, n), generator=g, dtype=torch.int32).to(DEVICE)
    qzeros = torch.randint(0, 2**31 - 1, (num_groups, n // P), generator=g, dtype=torch.int32).to(DEVICE)
    scales = (torch.rand(num_groups, n, generator=g) * 0.1 + 0.01).to(torch.float32).to(DEVICE)
    return qweight, qzeros, scales


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
    qweight, qzeros, scales = _random_gptq_fixture(k, n, group_size, seed=100 + k)
    x = torch.randn(m, k, device=DEVICE, dtype=torch.float32)

    ref_w = dequantize_gptq_int4_sequential_groups(qweight, qzeros, scales, group_size=group_size, out_dtype=torch.float32)
    ref_out = x @ ref_w

    fused_out = fused_gptq_linear(x, qweight, qzeros, scales, group_size=group_size, out_dtype=torch.float32)

    torch.testing.assert_close(fused_out, ref_out, atol=1e-3, rtol=1e-3)


@XPU
@pytest.mark.xpu
def test_fused_applies_bias():
    torch.manual_seed(0)
    m, k, n, group_size = 4, 128, 32, 128
    qweight, qzeros, scales = _random_gptq_fixture(k, n, group_size, seed=7)
    x = torch.randn(m, k, device=DEVICE, dtype=torch.float32)
    bias = torch.randn(n, device=DEVICE, dtype=torch.float32)

    ref_w = dequantize_gptq_int4_sequential_groups(qweight, qzeros, scales, group_size=group_size, out_dtype=torch.float32)
    ref_out = x @ ref_w + bias

    fused_out = fused_gptq_linear(x, qweight, qzeros, scales, group_size=group_size, bias=bias, out_dtype=torch.float32)

    torch.testing.assert_close(fused_out, ref_out, atol=1e-3, rtol=1e-3)


@XPU
@pytest.mark.xpu
def test_fused_output_dtype_defaults_to_input_dtype():
    m, k, n, group_size = 4, 128, 32, 128
    qweight, qzeros, scales = _random_gptq_fixture(k, n, group_size, seed=11)
    x = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)

    out = fused_gptq_linear(x, qweight, qzeros, scales, group_size=group_size)

    assert out.dtype == torch.bfloat16


@XPU
@pytest.mark.xpu
def test_group_size_not_multiple_of_pack_factor_raises():
    qweight, qzeros, scales = _random_gptq_fixture(64, 32, 32, seed=1)
    x = torch.randn(4, 64, device=DEVICE, dtype=torch.float32)
    with pytest.raises(ValueError, match="multiple of"):
        fused_gptq_linear(x, qweight, qzeros, scales, group_size=5)
