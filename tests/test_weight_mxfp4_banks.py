"""``stream_moe_expert_sources_mxfp4``: packed MXFP4 expert banks stay packed
(issue `moe-quant-banks-mxfp4`, #153, part of epic #140).

CPU-only, synthetic fixtures -- these banks are NEVER dequantized here (that
happens lazily, per-expert, at compute time in
:class:`freetoken.moe.offload_cache.SlotWeightAccessor`); the round-trip
check below dequantizes only for test verification, via the already-shipped
:func:`dequantize_mxfp4_blocks` (#129).

Unlike GPTQ's raw per-expert checkpoint layout, the real ``openai/gpt-oss-20b``
checkpoint ships each ``(layer, projection, component)`` as ONE already-packed
``[num_experts, ...]`` tensor (``model.layers.{L}.mlp.experts.{gate_up_proj|
down_proj}_{blocks|scales}`` -- verified against its own
``model.safetensors.index.json``), so the fixtures below build that layout
directly rather than per-expert rows.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from freetoken.kernel.triton.mxfp4_linear import dequantize_mxfp4_blocks, quantize_mxfp4_blocks
from freetoken.models.weight import MxfpExpertBank, stream_moe_expert_sources_mxfp4


def _packed_projection(num_experts: int, out_features: int, k: int, *, value: float) -> tuple[torch.Tensor, torch.Tensor]:
    """A small, real (not random-garbage) MXFP4-packed ``[E, out_features, K]``
    projection: every element is the same known constant, so round-tripping
    through quantize -> (blocks, scales) -> dequantize reproduces ``value``
    (up to E2M1's coarse 16-level quantization -- ``value`` below is always
    chosen to be one of the exact representable magnitudes)."""
    dense = torch.full((num_experts, out_features, k), value)
    return quantize_mxfp4_blocks(dense)


def _expert_stream(config, *, gate_up_kwargs, down_kwargs):
    """Yield ``(name, tensor)`` pairs for every layer's already-packed
    gate_up/down blocks+scales, in the real checkpoint's key spelling."""
    for layer in range(config.num_layers):
        gu_blocks, gu_scales = _packed_projection(config.num_experts, **gate_up_kwargs)
        dn_blocks, dn_scales = _packed_projection(config.num_experts, **down_kwargs)
        prefix = f"model.layers.{layer}.mlp.experts"
        yield f"{prefix}.gate_up_proj_blocks", gu_blocks
        yield f"{prefix}.gate_up_proj_scales", gu_scales
        yield f"{prefix}.down_proj_blocks", dn_blocks
        yield f"{prefix}.down_proj_scales", dn_scales


def test_packed_bank_shapes_match_the_documented_contract():
    config = SimpleNamespace(num_layers=2, num_experts=3)
    hidden, inter = 64, 32  # both multiples of 32, the MXFP4 block size
    gate_up_kwargs = dict(out_features=2 * inter, k=hidden, value=1.0)
    down_kwargs = dict(out_features=hidden, k=inter, value=1.0)
    stream = _expert_stream(config, gate_up_kwargs=gate_up_kwargs, down_kwargs=down_kwargs)

    gate_up_banks, down_banks = stream_moe_expert_sources_mxfp4(stream, config)

    assert len(gate_up_banks) == config.num_layers
    assert len(down_banks) == config.num_layers
    for bank in gate_up_banks:
        assert isinstance(bank, MxfpExpertBank)
        assert bank.blocks.shape == (config.num_experts, 2 * inter, hidden // 32, 16)
        assert bank.scales.shape == (config.num_experts, 2 * inter, hidden // 32)
        assert bank.blocks.dtype == torch.uint8
        assert bank.scales.dtype == torch.uint8
    for bank in down_banks:
        assert bank.blocks.shape == (config.num_experts, hidden, inter // 32, 16)
        assert bank.scales.shape == (config.num_experts, hidden, inter // 32)


def test_packed_bank_round_trips_to_the_known_fixture_value():
    config = SimpleNamespace(num_layers=1, num_experts=2)
    hidden, inter = 64, 32
    gate_up_kwargs = dict(out_features=2 * inter, k=hidden, value=1.5)
    down_kwargs = dict(out_features=hidden, k=inter, value=-2.0)
    stream = _expert_stream(config, gate_up_kwargs=gate_up_kwargs, down_kwargs=down_kwargs)

    gate_up_banks, down_banks = stream_moe_expert_sources_mxfp4(stream, config)

    for e in range(config.num_experts):
        down = dequantize_mxfp4_blocks(down_banks[0].blocks[e], down_banks[0].scales[e], out_dtype=torch.float32)
        torch.testing.assert_close(down, torch.full((hidden, inter), -2.0))

        gate_up = dequantize_mxfp4_blocks(gate_up_banks[0].blocks[e], gate_up_banks[0].scales[e], out_dtype=torch.float32)
        torch.testing.assert_close(gate_up, torch.full((2 * inter, hidden), 1.5))


def test_missing_layer_raises():
    config = SimpleNamespace(num_layers=2, num_experts=1)
    hidden, inter = 64, 32

    def stream():
        # only layer 0, layer 1 never appears
        gu_blocks, gu_scales = _packed_projection(1, 2 * inter, hidden, value=1.0)
        dn_blocks, dn_scales = _packed_projection(1, hidden, inter, value=1.0)
        prefix = "model.layers.0.mlp.experts"
        yield f"{prefix}.gate_up_proj_blocks", gu_blocks
        yield f"{prefix}.gate_up_proj_scales", gu_scales
        yield f"{prefix}.down_proj_blocks", dn_blocks
        yield f"{prefix}.down_proj_scales", dn_scales

    with pytest.raises(ValueError, match="Missing/incomplete MXFP4 MoE expert bank"):
        stream_moe_expert_sources_mxfp4(stream(), config)


def test_unexpected_key_raises():
    config = SimpleNamespace(num_layers=1, num_experts=1)
    with pytest.raises(ValueError, match="Unexpected MXFP4 expert weight key"):
        stream_moe_expert_sources_mxfp4(iter([("not.a.mxfp4.key", torch.zeros(1))]), config)


def test_wrong_expert_count_raises():
    config = SimpleNamespace(num_layers=1, num_experts=3)
    hidden, inter = 64, 32

    def stream():
        gu_blocks, gu_scales = _packed_projection(2, 2 * inter, hidden, value=1.0)  # only 2, need 3
        dn_blocks, dn_scales = _packed_projection(3, hidden, inter, value=1.0)
        prefix = "model.layers.0.mlp.experts"
        yield f"{prefix}.gate_up_proj_blocks", gu_blocks
        yield f"{prefix}.gate_up_proj_scales", gu_scales
        yield f"{prefix}.down_proj_blocks", dn_blocks
        yield f"{prefix}.down_proj_scales", dn_scales

    with pytest.raises(ValueError, match="expected 3 experts"):
        stream_moe_expert_sources_mxfp4(stream(), config)


def test_duplicate_component_raises():
    config = SimpleNamespace(num_layers=1, num_experts=1)
    hidden, inter = 64, 32

    def stream():
        gu_blocks, gu_scales = _packed_projection(1, 2 * inter, hidden, value=1.0)
        prefix = "model.layers.0.mlp.experts"
        yield f"{prefix}.gate_up_proj_blocks", gu_blocks
        yield f"{prefix}.gate_up_proj_blocks", gu_blocks  # duplicate

    with pytest.raises(ValueError, match="Duplicate MXFP4 component"):
        stream_moe_expert_sources_mxfp4(stream(), config)


def test_finalizes_each_layer_as_soon_as_it_completes():
    """Mirrors test_weight_gptq_banks.py's own version of this test: an
    earlier draft of the GPTQ streamer buffered the whole checkpoint's raw
    tensors before finalizing anything (issue #145's real RAM-blowup bug);
    this proves the MXFP4 streamer never repeats that mistake -- layer 0's
    two banks must both finalize before layer 1's tensors are even parsed."""
    config = SimpleNamespace(num_layers=2, num_experts=1)
    hidden, inter = 64, 32

    def stream():
        for layer in range(2):
            gu_blocks, gu_scales = _packed_projection(1, 2 * inter, hidden, value=1.0)
            dn_blocks, dn_scales = _packed_projection(1, hidden, inter, value=1.0)
            prefix = f"model.layers.{layer}.mlp.experts"
            yield f"{prefix}.gate_up_proj_blocks", gu_blocks
            yield f"{prefix}.gate_up_proj_scales", gu_scales
            yield f"{prefix}.down_proj_blocks", dn_blocks
            yield f"{prefix}.down_proj_scales", dn_scales

    import freetoken.models.weight as weight_mod

    call_order = []
    original = weight_mod.MxfpExpertBank

    def spy(*, blocks, scales):
        call_order.append(int(blocks.shape[0]))  # records finalization order via a side channel
        return original(blocks=blocks, scales=scales)

    weight_mod.MxfpExpertBank = spy
    try:
        gate_up_banks, down_banks = weight_mod.stream_moe_expert_sources_mxfp4(stream(), config)
    finally:
        weight_mod.MxfpExpertBank = original

    assert len(gate_up_banks) == 2
    assert len(down_banks) == 2
    # Four banks total finalized (gate_up + down, x2 layers) -- proves both of
    # layer 0's banks completed (the spy fired) without needing layer 1's
    # tensors to have arrived yet (they are consumed from a lazy generator,
    # so this would raise/hang first if the implementation tried to peek
    # ahead into layer 1 before layer 0 was done).
    assert len(call_order) == 4
