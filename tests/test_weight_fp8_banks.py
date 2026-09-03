"""``stream_moe_expert_sources_fp8``: packed block-FP8 expert banks stay
packed (issue `moe-quant-banks-fp8`, #152, part of epic #140/#134).

CPU-only, synthetic fixtures -- the whole point of this issue is that these
banks are NEVER dequantized here (that happens lazily, per-expert, at
compute time, via SlotWeightAccessor); the round-trip check below
dequantizes only for test verification, via the already-shipped
``dequantize_block_fp8`` (issue #125).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from freetoken.kernel.triton.fp8_block_linear import dequantize_block_fp8, quantize_block_fp8
from freetoken.models.weight import Fp8BlockExpertBank, stream_moe_expert_sources_fp8


def _fp8_projection(n: int, k: int, value: float, *, block: int = 8):
    """A small, real (not random-garbage) block-FP8-packed [N, K] projection:
    every element is the same constant, so the dequantized value round-trips
    (within fp8's coarse quantization error) to a known constant."""
    dense = torch.full((n, k), value)
    weight, scale = quantize_block_fp8(dense, block=block)
    return weight, scale


def _expert_stream(config, *, gate_value, up_value, down_value, n: int, k: int, block: int):
    """Yield ``(name, tensor)`` pairs for every expert of every layer, in the
    real checkpoint's raw per-expert block-FP8 key spelling (verified against
    deepseek-ai/DeepSeek-V3's own model.safetensors.index.json)."""
    for layer in range(config.num_layers):
        for e in range(config.num_experts):
            for proj, value in (("gate_proj", gate_value), ("up_proj", up_value), ("down_proj", down_value)):
                weight, scale = _fp8_projection(n, k, value, block=block)
                prefix = f"model.layers.{layer}.mlp.experts.{e}.{proj}"
                yield prefix + ".weight", weight
                yield prefix + ".weight_scale_inv", scale


def test_packed_bank_shapes_match_the_documented_contract():
    config = SimpleNamespace(num_layers=2, num_experts=3)
    n, k, block = 8, 16, 8
    stream = _expert_stream(config, gate_value=1.0, up_value=1.0, down_value=1.0, n=n, k=k, block=block)

    gate_up_banks, down_banks = stream_moe_expert_sources_fp8(stream, config, block=block)

    assert len(gate_up_banks) == config.num_layers
    assert len(down_banks) == config.num_layers
    for bank in gate_up_banks:
        assert isinstance(bank, Fp8BlockExpertBank)
        # gate_up fuses gate_proj+up_proj on the N axis -> N doubles.
        assert bank.weight.shape == (config.num_experts, 2 * n, k)
        assert bank.weight_scale_inv.shape == (config.num_experts, 2 * n // block, k // block)
        assert bank.weight.dtype == torch.float8_e4m3fn
    for bank in down_banks:
        assert bank.weight.shape == (config.num_experts, n, k)
        assert bank.weight_scale_inv.shape == (config.num_experts, n // block, k // block)


def test_packed_bank_round_trips_to_the_known_fixture_value():
    config = SimpleNamespace(num_layers=1, num_experts=2)
    n, k, block = 8, 16, 8
    stream = _expert_stream(config, gate_value=2.0, up_value=2.0, down_value=-3.0, n=n, k=k, block=block)

    gate_up_banks, down_banks = stream_moe_expert_sources_fp8(stream, config, block=block)

    for e in range(config.num_experts):
        down = dequantize_block_fp8(
            down_banks[0].weight[e], down_banks[0].weight_scale_inv[e], block=block, out_dtype=torch.float32
        )
        torch.testing.assert_close(down, torch.full((n, k), -3.0), atol=0.05, rtol=0.05)

        gate_up = dequantize_block_fp8(
            gate_up_banks[0].weight[e], gate_up_banks[0].weight_scale_inv[e], block=block, out_dtype=torch.float32
        )
        torch.testing.assert_close(gate_up, torch.full((2 * n, k), 2.0), atol=0.05, rtol=0.05)


def test_gate_up_concatenation_preserves_two_independently_quantized_halves():
    """A real test of the concat axis, not a trivial no-op: gate_proj and
    up_proj are quantized with DIFFERENT values, so the fused bank's two
    halves must dequantize back to their own distinct source values."""
    config = SimpleNamespace(num_layers=1, num_experts=1)
    n, k, block = 8, 16, 8
    stream = _expert_stream(config, gate_value=1.5, up_value=-2.5, down_value=0.5, n=n, k=k, block=block)

    gate_up_banks, _down_banks = stream_moe_expert_sources_fp8(stream, config, block=block)
    bank = gate_up_banks[0]

    fused = dequantize_block_fp8(bank.weight[0], bank.weight_scale_inv[0], block=block, out_dtype=torch.float32)
    # gate/up were concatenated on the N (dim 0) axis, so the two halves split
    # along rows.
    assert fused.shape == (2 * n, k)
    gate_half, up_half = fused[:n], fused[n:]
    torch.testing.assert_close(gate_half, torch.full((n, k), 1.5), atol=0.05, rtol=0.05)
    torch.testing.assert_close(up_half, torch.full((n, k), -2.5), atol=0.05, rtol=0.05)


def test_gate_proj_n_not_multiple_of_block_raises():
    config = SimpleNamespace(num_layers=1, num_experts=1)
    n, k, block = 8, 16, 8

    def stream():
        for proj, value in (("gate_proj", 1.0), ("up_proj", 1.0), ("down_proj", 1.0)):
            # Fabricate an N=5 projection directly (not a multiple of block=8)
            # -- quantize_block_fp8 itself pads internally, but the *stored*
            # weight shape here is what stream_moe_expert_sources_fp8 checks.
            weight, scale = _fp8_projection(5, k, value, block=block)
            prefix = f"model.layers.0.mlp.experts.0.{proj}"
            yield prefix + ".weight", weight
            yield prefix + ".weight_scale_inv", scale

    with pytest.raises(ValueError, match="not a multiple of block"):
        stream_moe_expert_sources_fp8(stream(), config, block=block)


def test_missing_layer_raises():
    config = SimpleNamespace(num_layers=2, num_experts=1)
    n, k, block = 8, 8, 8

    def stream():
        # only layer 0, layer 1 never appears
        for proj in ("gate_proj", "up_proj", "down_proj"):
            weight, scale = _fp8_projection(n, k, 1.0, block=block)
            prefix = "model.layers.0.mlp.experts.0." + proj
            yield prefix + ".weight", weight
            yield prefix + ".weight_scale_inv", scale

    with pytest.raises(ValueError, match="Missing/incomplete block-FP8 MoE expert bank"):
        stream_moe_expert_sources_fp8(stream(), config, block=block)


def test_unexpected_key_raises():
    config = SimpleNamespace(num_layers=1, num_experts=1)
    with pytest.raises(ValueError, match="Unexpected block-FP8 expert weight key"):
        stream_moe_expert_sources_fp8(iter([("not.a.fp8.key", torch.zeros(1))]), config)


def test_finalizes_each_layer_as_soon_as_it_completes():
    """Mirrors the real fix from GPTQ's issue #145: streaming layer 0 to
    completion must release its raw buffer before layer 1 is even seen, not
    wait for the whole generator to be exhausted."""
    config = SimpleNamespace(num_layers=2, num_experts=1)
    n, k, block = 8, 8, 8

    def stream():
        for layer in range(2):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                weight, scale = _fp8_projection(n, k, 1.0, block=block)
                prefix = f"model.layers.{layer}.mlp.experts.0.{proj}"
                yield prefix + ".weight", weight
                yield prefix + ".weight_scale_inv", scale

    import freetoken.models.weight as weight_mod

    original_finalize = weight_mod._finalize_fp8_bank
    call_order = []

    def spy(by_expert, num_experts, *, fuse, layer, block):
        call_order.append((layer, "gate_up" if fuse else "down"))
        return original_finalize(by_expert, num_experts, fuse=fuse, layer=layer, block=block)

    weight_mod._finalize_fp8_bank = spy
    try:
        weight_mod.stream_moe_expert_sources_fp8(stream(), config, block=block)
    finally:
        weight_mod._finalize_fp8_bank = original_finalize

    layer0_calls = [c for c in call_order if c[0] == 0]
    layer1_calls = [c for c in call_order if c[0] == 1]
    assert len(layer0_calls) == 2
    assert len(layer1_calls) == 2
    assert call_order.index(layer1_calls[0]) > call_order.index(layer0_calls[-1])
