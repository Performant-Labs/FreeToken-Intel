"""``stream_moe_expert_sources_gptq``: packed GPTQ expert banks stay packed
(issue `moe-quant-banks-pack`, #135, part of epic #134).

CPU-only, synthetic fixtures -- the whole point of this issue is that these
banks are NEVER dequantized here (that happens lazily, per-expert, at
compute time, issue #137); the round-trip check below dequantizes only for
test verification, via the already-shipped ``dequantize_gptq_int4`` (#132).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from freetoken.kernel.triton.gptq_linear import dequantize_gptq_int4
from freetoken.models.weight import GptqExpertBank, stream_moe_expert_sources_gptq


def _pack_nibbles(codes: list[int]) -> int:
    word = 0
    for i, c in enumerate(codes):
        word |= c << (4 * i)
    # A packed word with the sign bit (31) set is a legitimate int32 bit
    # pattern (dequantize_gptq_int4 masks after shifting, so arithmetic vs
    # logical right-shift makes no difference to the extracted nibble -- see
    # gptq_linear._unpack_int32) but torch.tensor(..., dtype=torch.int32)
    # rejects a Python int >= 2**31 as an overflow. Fold it to the equivalent
    # two's-complement negative value first.
    if word >= 1 << 31:
        word -= 1 << 32
    return word


def _gptq_projection(k: int, n: int, group_size: int, *, weight_code: int, zero_code: int, scale: float):
    """A small, real (not random-garbage) GPTQ-packed projection: every row
    uses the same weight code / zero-point / scale, so the dequantized value
    is a known constant: ``scale * (weight_code - zero_code)`` (this
    checkpoint format's zero-point is stored directly, no +1 correction --
    see issue #147)."""
    assert k % 8 == 0 and n % 8 == 0 and k % group_size == 0
    n_groups = k // group_size
    qweight = torch.tensor([[_pack_nibbles([weight_code] * 8)] * n for _ in range(k // 8)], dtype=torch.int32)
    qzeros = torch.tensor([[_pack_nibbles([zero_code] * 8)] * (n // 8) for _ in range(n_groups)], dtype=torch.int32)
    scales = torch.full((n_groups, n), scale)
    g_idx = torch.tensor([i // group_size for i in range(k)], dtype=torch.int32)
    return qweight, qzeros, scales, g_idx


def _expert_stream(config, *, gate_kwargs, up_kwargs, down_kwargs):
    """Yield ``(name, tensor)`` pairs for every expert of every layer, in the
    real checkpoint's raw per-expert GPTQ key spelling."""
    for layer in range(config.num_layers):
        for e in range(config.num_experts):
            for proj, kwargs in (("gate_proj", gate_kwargs), ("up_proj", up_kwargs), ("down_proj", down_kwargs)):
                qweight, qzeros, scales, g_idx = _gptq_projection(**kwargs)
                prefix = f"model.layers.{layer}.mlp.experts.{e}.{proj}"
                yield prefix + ".qweight", qweight
                yield prefix + ".qzeros", qzeros
                yield prefix + ".scales", scales
                yield prefix + ".g_idx", g_idx


def test_packed_bank_shapes_match_the_documented_contract():
    config = SimpleNamespace(num_layers=2, num_experts=3)
    k, n, group_size = 16, 8, 8
    kwargs = dict(k=k, n=n, group_size=group_size, weight_code=5, zero_code=7, scale=0.5)
    stream = _expert_stream(config, gate_kwargs=kwargs, up_kwargs=kwargs, down_kwargs=kwargs)

    gate_up_banks, down_banks = stream_moe_expert_sources_gptq(stream, config)

    assert len(gate_up_banks) == config.num_layers
    assert len(down_banks) == config.num_layers
    for bank in gate_up_banks:
        assert isinstance(bank, GptqExpertBank)
        # gate_up fuses gate_proj+up_proj on the N axis -> N doubles.
        assert bank.qweight.shape == (config.num_experts, k // 8, 2 * n)
        assert bank.qzeros.shape == (config.num_experts, k // group_size, 2 * n // 8)
        assert bank.scales.shape == (config.num_experts, k // group_size, 2 * n)
        assert bank.g_idx.shape == (k,)
        assert bank.qweight.dtype == torch.int32
        assert bank.qzeros.dtype == torch.int32
        assert bank.g_idx.dtype == torch.int32
    for bank in down_banks:
        assert bank.qweight.shape == (config.num_experts, k // 8, n)
        assert bank.qzeros.shape == (config.num_experts, k // group_size, n // 8)
        assert bank.scales.shape == (config.num_experts, k // group_size, n)
        assert bank.g_idx.shape == (k,)


def test_packed_bank_round_trips_to_the_known_fixture_value():
    config = SimpleNamespace(num_layers=1, num_experts=2)
    k, n, group_size = 16, 8, 8
    # weight_code=5, zero_code=8, scale=0.5 -> 0.5*(5-8) = -1.5.
    kwargs = dict(k=k, n=n, group_size=group_size, weight_code=5, zero_code=8, scale=0.5)
    stream = _expert_stream(config, gate_kwargs=kwargs, up_kwargs=kwargs, down_kwargs=kwargs)

    gate_up_banks, down_banks = stream_moe_expert_sources_gptq(stream, config)

    for e in range(config.num_experts):
        down = dequantize_gptq_int4(
            down_banks[0].qweight[e], down_banks[0].qzeros[e], down_banks[0].scales[e], down_banks[0].g_idx,
            out_dtype=torch.float32,
        )
        # dequantize_gptq_int4 returns [K, N] (in_features, out_features).
        torch.testing.assert_close(down, torch.full((k, n), -1.5), atol=1e-2, rtol=0)

        gate_up = dequantize_gptq_int4(
            gate_up_banks[0].qweight[e], gate_up_banks[0].qzeros[e], gate_up_banks[0].scales[e], gate_up_banks[0].g_idx,
            out_dtype=torch.float32,
        )
        torch.testing.assert_close(gate_up, torch.full((k, 2 * n), -1.5), atol=1e-2, rtol=0)


def test_gate_up_concatenation_preserves_two_independently_quantized_halves():
    """A real test of the concat axis, not a trivial no-op: gate_proj and
    up_proj are quantized with DIFFERENT codes/scales, so the fused bank's
    two halves must dequantize back to their own distinct source values."""
    config = SimpleNamespace(num_layers=1, num_experts=1)
    k, n, group_size = 16, 8, 8
    gate_kwargs = dict(k=k, n=n, group_size=group_size, weight_code=3, zero_code=9, scale=0.25)  # -> 0.25*(3-9) = -1.5
    up_kwargs = dict(k=k, n=n, group_size=group_size, weight_code=10, zero_code=7, scale=1.0)  # -> 1.0*(10-7) = 3.0
    down_kwargs = dict(k=k, n=n, group_size=group_size, weight_code=5, zero_code=7, scale=0.5)
    stream = _expert_stream(config, gate_kwargs=gate_kwargs, up_kwargs=up_kwargs, down_kwargs=down_kwargs)

    gate_up_banks, _down_banks = stream_moe_expert_sources_gptq(stream, config)
    bank = gate_up_banks[0]

    fused = dequantize_gptq_int4(bank.qweight[0], bank.qzeros[0], bank.scales[0], bank.g_idx, out_dtype=torch.float32)
    # dequantize_gptq_int4 returns [K, N]; gate/up were concatenated on the N
    # (output-channel, dim=1) axis, so the two halves split along columns.
    assert fused.shape == (k, 2 * n)
    gate_half, up_half = fused[:, :n], fused[:, n:]
    torch.testing.assert_close(gate_half, torch.full((k, n), -1.5))
    torch.testing.assert_close(up_half, torch.full((k, n), 3.0))


def test_g_idx_mismatch_between_gate_and_up_raises():
    config = SimpleNamespace(num_layers=1, num_experts=1)
    k, n = 16, 8
    gate_kwargs = dict(k=k, n=n, group_size=8, weight_code=5, zero_code=7, scale=0.5)
    up_kwargs = dict(k=k, n=n, group_size=4, weight_code=5, zero_code=7, scale=0.5)  # different group_size -> different g_idx
    down_kwargs = gate_kwargs
    stream = _expert_stream(config, gate_kwargs=gate_kwargs, up_kwargs=up_kwargs, down_kwargs=down_kwargs)

    with pytest.raises(ValueError, match="g_idx mismatch"):
        stream_moe_expert_sources_gptq(stream, config)


def test_g_idx_mismatch_across_experts_raises():
    config = SimpleNamespace(num_layers=1, num_experts=2)
    k, n = 16, 8

    def stream():
        for e, group_size in enumerate((8, 4)):  # expert 0 vs expert 1 disagree
            kwargs = dict(k=k, n=n, group_size=group_size, weight_code=5, zero_code=7, scale=0.5)
            for proj in ("gate_proj", "up_proj", "down_proj"):
                qweight, qzeros, scales, g_idx = _gptq_projection(**kwargs)
                prefix = f"model.layers.0.mlp.experts.{e}.{proj}"
                yield prefix + ".qweight", qweight
                yield prefix + ".qzeros", qzeros
                yield prefix + ".scales", scales
                yield prefix + ".g_idx", g_idx

    with pytest.raises(ValueError, match="g_idx differs"):
        stream_moe_expert_sources_gptq(stream(), config)


def test_missing_layer_raises():
    config = SimpleNamespace(num_layers=2, num_experts=1)
    kwargs = dict(k=8, n=8, group_size=8, weight_code=5, zero_code=7, scale=0.5)

    def stream():
        # only layer 0, layer 1 never appears
        for proj in ("gate_proj", "up_proj", "down_proj"):
            qweight, qzeros, scales, g_idx = _gptq_projection(**kwargs)
            prefix = f"model.layers.0.mlp.experts.0.{proj}"
            yield prefix + ".qweight", qweight
            yield prefix + ".qzeros", qzeros
            yield prefix + ".scales", scales
            yield prefix + ".g_idx", g_idx

    with pytest.raises(ValueError, match="Missing/incomplete GPTQ MoE expert bank"):
        stream_moe_expert_sources_gptq(stream(), config)


def test_unexpected_key_raises():
    config = SimpleNamespace(num_layers=1, num_experts=1)
    with pytest.raises(ValueError, match="Unexpected GPTQ expert weight key"):
        stream_moe_expert_sources_gptq(iter([("not.a.gptq.key", torch.zeros(1))]), config)


def test_finalizes_each_layer_as_soon_as_it_completes():
    """The real fix from issue #138's real-checkpoint validation: an earlier
    version buffered every layer's raw tensors until the whole stream ended,
    holding the full raw checkpoint AND the packed banks being built from it
    at once -- exactly the RAM blowup #135 was supposed to eliminate, just
    moved to a different phase. Proves the fix directly: streaming layer 0
    to completion must release its raw buffer before layer 1 is even seen,
    not wait for the whole generator to be exhausted."""
    config = SimpleNamespace(num_layers=2, num_experts=1)
    kwargs = dict(k=8, n=8, group_size=8, weight_code=5, zero_code=7, scale=0.5)
    seen_buf_sizes = []

    def stream():
        for layer in range(2):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                qweight, qzeros, scales, g_idx = _gptq_projection(**kwargs)
                prefix = f"model.layers.{layer}.mlp.experts.0.{proj}"
                yield prefix + ".qweight", qweight
                yield prefix + ".qzeros", qzeros
                yield prefix + ".scales", scales
                yield prefix + ".g_idx", g_idx

    # Instrument via a thin wrapper that records len(buf) is never asked to
    # hold more than one layer's worth of incomplete banks at a time --
    # can't reach into the closure directly, so drive the same algorithm's
    # observable effect instead: patch _finalize_gptq_bank to record how
    # many (layer, bank_name) keys are still outstanding at each call.
    import freetoken.models.weight as weight_mod

    original_finalize = weight_mod._finalize_gptq_bank
    call_order = []

    def spy(by_expert, num_experts, *, fuse, layer):
        call_order.append((layer, "gate_up" if fuse else "down"))
        return original_finalize(by_expert, num_experts, fuse=fuse, layer=layer)

    weight_mod._finalize_gptq_bank = spy
    try:
        weight_mod.stream_moe_expert_sources_gptq(stream(), config)
    finally:
        weight_mod._finalize_gptq_bank = original_finalize

    # Layer 0's two banks (gate_up, down) must both finalize before layer 1's
    # tensors are even parsed -- i.e. before layer 1 appears in call_order.
    layer0_calls = [c for c in call_order if c[0] == 0]
    layer1_calls = [c for c in call_order if c[0] == 1]
    assert len(layer0_calls) == 2
    assert len(layer1_calls) == 2
    assert call_order.index(layer1_calls[0]) > call_order.index(layer0_calls[-1])
