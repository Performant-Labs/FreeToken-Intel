"""``stream_moe_expert_sources_int8``: packed per-channel-INT8 expert banks
stay packed (issue `moe-quant-banks-int8`, #154, part of epic #140).

CPU-only, synthetic fixtures -- the whole point of this issue is that these
banks are NEVER dequantized here (that happens lazily, per-expert, at
compute time via ``SlotWeightAccessor``); the round-trip check below
dequantizes only for test verification, via the already-shipped
``dequantize_int8_channel`` (#130).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from freetoken.kernel.triton.int8_linear import dequantize_int8_channel
from freetoken.models.weight import Int8ExpertBank, stream_moe_expert_sources_int8


def _int8_projection(k: int, n: int, *, code: int, scale: float):
    """A small, real (not random-garbage) per-channel-INT8-packed
    projection: every row uses the same int8 code / scale, so the
    dequantized value is a known constant: ``scale * code``."""
    weight = torch.full((n, k), code, dtype=torch.int8)
    scale_t = torch.full((n,), scale, dtype=torch.float32)
    return weight, scale_t


def _expert_stream(config, *, gate_kwargs, up_kwargs, down_kwargs):
    """Yield ``(name, tensor)`` pairs for every expert of every layer, in the
    UNVERIFIED assumed per-expert INT8 key spelling (see
    ``_parse_int8_expert_key``'s docstring)."""
    for layer in range(config.num_layers):
        for e in range(config.num_experts):
            for proj, kwargs in (("gate_proj", gate_kwargs), ("up_proj", up_kwargs), ("down_proj", down_kwargs)):
                weight, scale = _int8_projection(**kwargs)
                prefix = f"model.layers.{layer}.mlp.experts.{e}.{proj}"
                yield prefix + ".weight", weight
                yield prefix + ".weight_scale", scale


def test_packed_bank_shapes_match_the_documented_contract():
    config = SimpleNamespace(num_layers=2, num_experts=3)
    k, n = 16, 8
    kwargs = dict(k=k, n=n, code=5, scale=0.5)
    stream = _expert_stream(config, gate_kwargs=kwargs, up_kwargs=kwargs, down_kwargs=kwargs)

    gate_up_banks, down_banks = stream_moe_expert_sources_int8(stream, config)

    assert len(gate_up_banks) == config.num_layers
    assert len(down_banks) == config.num_layers
    for bank in gate_up_banks:
        assert isinstance(bank, Int8ExpertBank)
        # gate_up fuses gate_proj+up_proj on the N (row/output-channel) axis -> N doubles.
        assert bank.weight.shape == (config.num_experts, 2 * n, k)
        assert bank.scale.shape == (config.num_experts, 2 * n)
        assert bank.weight.dtype == torch.int8
        assert bank.scale.dtype == torch.float32
    for bank in down_banks:
        assert bank.weight.shape == (config.num_experts, n, k)
        assert bank.scale.shape == (config.num_experts, n)


def test_packed_bank_round_trips_to_the_known_fixture_value():
    config = SimpleNamespace(num_layers=1, num_experts=2)
    k, n = 16, 8
    kwargs = dict(k=k, n=n, code=5, scale=0.5)  # -> 0.5 * 5 = 2.5
    stream = _expert_stream(config, gate_kwargs=kwargs, up_kwargs=kwargs, down_kwargs=kwargs)

    gate_up_banks, down_banks = stream_moe_expert_sources_int8(stream, config)

    for e in range(config.num_experts):
        down = dequantize_int8_channel(down_banks[0].weight[e], down_banks[0].scale[e], out_dtype=torch.float32)
        torch.testing.assert_close(down, torch.full((n, k), 2.5))

        gate_up = dequantize_int8_channel(gate_up_banks[0].weight[e], gate_up_banks[0].scale[e], out_dtype=torch.float32)
        torch.testing.assert_close(gate_up, torch.full((2 * n, k), 2.5))


def test_gate_up_concatenation_preserves_two_independently_quantized_halves():
    """A real test of the concat axis, not a trivial no-op: gate_proj and
    up_proj are quantized with DIFFERENT codes/scales, so the fused bank's
    two halves must dequantize back to their own distinct source values."""
    config = SimpleNamespace(num_layers=1, num_experts=1)
    k, n = 16, 8
    gate_kwargs = dict(k=k, n=n, code=3, scale=0.25)  # -> 0.75
    up_kwargs = dict(k=k, n=n, code=10, scale=1.0)  # -> 10.0
    down_kwargs = dict(k=k, n=n, code=5, scale=0.5)
    stream = _expert_stream(config, gate_kwargs=gate_kwargs, up_kwargs=up_kwargs, down_kwargs=down_kwargs)

    gate_up_banks, _down_banks = stream_moe_expert_sources_int8(stream, config)
    bank = gate_up_banks[0]

    fused = dequantize_int8_channel(bank.weight[0], bank.scale[0], out_dtype=torch.float32)
    # gate/up were concatenated on the N (row, dim=0) axis, so the two halves
    # split along rows.
    assert fused.shape == (2 * n, k)
    gate_half, up_half = fused[:n], fused[n:]
    torch.testing.assert_close(gate_half, torch.full((n, k), 0.75))
    torch.testing.assert_close(up_half, torch.full((n, k), 10.0))


def test_missing_layer_raises():
    config = SimpleNamespace(num_layers=2, num_experts=1)
    kwargs = dict(k=8, n=8, code=5, scale=0.5)

    def stream():
        # only layer 0, layer 1 never appears
        for proj in ("gate_proj", "up_proj", "down_proj"):
            weight, scale = _int8_projection(**kwargs)
            prefix = f"model.layers.0.mlp.experts.0.{proj}"
            yield prefix + ".weight", weight
            yield prefix + ".weight_scale", scale

    with pytest.raises(ValueError, match="Missing/incomplete INT8 MoE expert bank"):
        stream_moe_expert_sources_int8(stream(), config)


def test_unexpected_key_raises():
    config = SimpleNamespace(num_layers=1, num_experts=1)
    with pytest.raises(ValueError, match="Unexpected INT8 expert weight key"):
        stream_moe_expert_sources_int8(iter([("not.an.int8.key", torch.zeros(1))]), config)


def test_finalizes_each_layer_as_soon_as_it_completes():
    """Mirrors the real fix issue #145 made for the GPTQ streamer: streaming
    layer 0 to completion must release its raw buffer before layer 1 is even
    seen, not wait for the whole generator to be exhausted."""
    config = SimpleNamespace(num_layers=2, num_experts=1)
    kwargs = dict(k=8, n=8, code=5, scale=0.5)

    def stream():
        for layer in range(2):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                weight, scale = _int8_projection(**kwargs)
                prefix = f"model.layers.{layer}.mlp.experts.0.{proj}"
                yield prefix + ".weight", weight
                yield prefix + ".weight_scale", scale

    import freetoken.models.weight as weight_mod

    original_finalize = weight_mod._finalize_int8_bank
    call_order = []

    def spy(by_expert, num_experts, *, fuse, layer):
        call_order.append((layer, "gate_up" if fuse else "down"))
        return original_finalize(by_expert, num_experts, fuse=fuse, layer=layer)

    weight_mod._finalize_int8_bank = spy
    try:
        weight_mod.stream_moe_expert_sources_int8(stream(), config)
    finally:
        weight_mod._finalize_int8_bank = original_finalize

    layer0_calls = [c for c in call_order if c[0] == 0]
    layer1_calls = [c for c in call_order if c[0] == 1]
    assert len(layer0_calls) == 2
    assert len(layer1_calls) == 2
    assert call_order.index(layer1_calls[0]) > call_order.index(layer0_calls[-1])
