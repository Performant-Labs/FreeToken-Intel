"""Correctness tests for GPTQ INT4 dequantization (issue quant-xpu, #10).

Pure torch, no XPU dependency, CPU-testable. The known-encoding test hand-
packs qweight/qzeros bytes via plain Python bit arithmetic (independent of
dequantize_gptq_int4's own torch bit-ops), so it checks the unpacking math
against the AutoGPTQ reference layout directly, not just self-consistency.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from freetoken.kernel.triton.gptq_linear import (
    dequantize_gptq_int4,
    dequantize_gptq_int4_sequential_groups,
    gptq_linear,
)


def _pack_nibbles(codes: list[int]) -> int:
    """8 4-bit codes -> one packed int32 word (LSB = codes[0], matching
    AutoGPTQ: shift amounts are 0, 4, 8, ..., 28)."""
    word = 0
    for i, c in enumerate(codes):
        assert 0 <= c < 16
        word |= c << (4 * i)
    # A packed word with bit 31 set (e.g. every nibble = 8) is a legitimate
    # int32 bit pattern, but torch.tensor(..., dtype=torch.int32) rejects a
    # Python int >= 2**31 as an overflow -- fold to the equivalent
    # two's-complement negative value first (see also test_weight_gptq_banks).
    if word >= 1 << 31:
        word -= 1 << 32
    return word


def test_dequantize_matches_hand_packed_known_encoding():
    K, N, group_size = 8, 8, 8  # one group covers the whole K
    # Every column shares the same 8 4-bit weight codes 0..7 (row k -> code k).
    codes = list(range(8))
    qweight = torch.tensor([[_pack_nibbles(codes)] * N], dtype=torch.int32)
    assert qweight.shape == (1, N)

    # Zero-point 8 (the standard symmetric midpoint) for every output channel:
    # this checkpoint format (issue #147) stores the true zero-point directly,
    # so the raw nibble is 8, not the legacy AutoGPTQ-v1 stored-minus-one 7.
    qzeros = torch.tensor([[_pack_nibbles([8] * 8)]], dtype=torch.int32)
    assert qzeros.shape == (1, N // 8)

    scales = torch.full((1, N), 0.5)
    g_idx = torch.zeros(K, dtype=torch.int32)  # single group

    out = dequantize_gptq_int4(qweight, qzeros, scales, g_idx, out_dtype=torch.float32)
    assert out.shape == (K, N)
    expected_col = torch.tensor([0.5 * (c - 8) for c in codes])  # [-4, -3.5, ..., -0.5]
    for n in range(N):
        torch.testing.assert_close(out[:, n], expected_col)


def test_dequantize_respects_per_group_scale_and_zero():
    """Two groups (group_size=4) with different scale/zero -- the low half
    of K must use group 0's params, the high half group 1's."""
    K, N, group_size = 8, 8, 4
    codes = [1] * 8  # every row's weight code is 1 (arbitrary, constant)
    qweight = torch.tensor([[_pack_nibbles(codes)] * N], dtype=torch.int32)
    qzeros = torch.tensor(
        [[_pack_nibbles([8] * 8)], [_pack_nibbles([4] * 8)]], dtype=torch.int32
    )  # group0 zero=8, group1 zero=4
    scales = torch.tensor([[1.0] * N, [10.0] * N])
    g_idx = torch.tensor([i // group_size for i in range(K)], dtype=torch.int32)

    out = dequantize_gptq_int4(qweight, qzeros, scales, g_idx, out_dtype=torch.float32)
    # group 0 (rows 0-3): scale=1.0, zero=8 -> 1.0 * (1 - 8) = -7
    torch.testing.assert_close(out[:4, 0], torch.full((4,), -7.0))
    # group 1 (rows 4-7): scale=10.0, zero=4 -> 10.0 * (1 - 4) = -30
    torch.testing.assert_close(out[4:, 0], torch.full((4,), -30.0))


def test_dequantize_rejects_non_int32_qweight():
    with pytest.raises(TypeError, match="qweight must be int32"):
        dequantize_gptq_int4(
            torch.zeros(1, 8, dtype=torch.int64),
            torch.zeros(1, 1, dtype=torch.int32),
            torch.zeros(1, 8),
            torch.zeros(8, dtype=torch.int32),
        )


def test_dequantize_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="mismatch"):
        dequantize_gptq_int4(
            torch.zeros(1, 8, dtype=torch.int32),
            torch.zeros(1, 1, dtype=torch.int32),
            torch.zeros(1, 4),  # wrong N
            torch.zeros(8, dtype=torch.int32),
        )


def test_dequantize_rejects_g_idx_length_mismatch():
    with pytest.raises(ValueError, match="g_idx"):
        dequantize_gptq_int4(
            torch.zeros(1, 8, dtype=torch.int32),
            torch.zeros(1, 1, dtype=torch.int32),
            torch.zeros(1, 8),
            torch.zeros(4, dtype=torch.int32),  # should be K=8
        )


def test_gptq_linear_matches_manual_dequant_matmul():
    K, N, group_size = 8, 8, 8
    codes = [(k * 3) % 16 for k in range(8)]
    qweight = torch.tensor([[_pack_nibbles(codes)] * N], dtype=torch.int32)
    qzeros = torch.tensor([[_pack_nibbles([8] * 8)]], dtype=torch.int32)
    scales = torch.full((1, N), 0.25)
    g_idx = torch.zeros(K, dtype=torch.int32)

    torch.manual_seed(0)
    x = torch.randn(3, K)
    w = dequantize_gptq_int4(qweight, qzeros, scales, g_idx, out_dtype=torch.float32)
    ref = torch.nn.functional.linear(x, w.T)

    out = gptq_linear(x, qweight, qzeros, scales, g_idx)
    torch.testing.assert_close(out, ref)


def test_sequential_groups_matches_explicit_g_idx():
    """dequantize_gptq_int4_sequential_groups (issue #137's compute-time
    dequant helper -- no stored g_idx) must match dequantize_gptq_int4 given
    the equivalent explicit g_idx=[k // group_size for k in range(K)]."""
    K, N, group_size = 16, 8, 4
    torch.manual_seed(0)
    codes = [(k * 5) % 16 for k in range(8)]
    qweight = torch.tensor([[_pack_nibbles(codes)] * N for _ in range(K // 8)], dtype=torch.int32)
    n_groups = K // group_size
    qzeros = torch.tensor([[_pack_nibbles([3 + g] * 8)] for g in range(n_groups)], dtype=torch.int32)  # [n_groups, N//8]
    scales = torch.stack([torch.full((N,), 0.1 * (g + 1)) for g in range(n_groups)])
    explicit_g_idx = torch.tensor([k // group_size for k in range(K)], dtype=torch.int32)

    expected = dequantize_gptq_int4(qweight, qzeros, scales, explicit_g_idx, out_dtype=torch.float32)
    got = dequantize_gptq_int4_sequential_groups(qweight, qzeros, scales, group_size=group_size, out_dtype=torch.float32)
    torch.testing.assert_close(got, expected)


def test_sequential_groups_rejects_nothing_extra_and_shapes_correctly():
    K, N, group_size = 8, 8, 8
    codes = [(k * 3) % 16 for k in range(8)]
    qweight = torch.tensor([[_pack_nibbles(codes)] * N], dtype=torch.int32)
    qzeros = torch.tensor([[_pack_nibbles([8] * 8)]], dtype=torch.int32)
    scales = torch.full((1, N), 0.25)

    out = dequantize_gptq_int4_sequential_groups(qweight, qzeros, scales, group_size=group_size)
    assert out.shape == (K, N)
    assert out.dtype == torch.bfloat16
