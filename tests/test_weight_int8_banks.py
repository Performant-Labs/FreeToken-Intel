"""``stream_moe_expert_sources_int8``: packed compressed-tensors
pack-quantized INT8 expert banks stay packed (issue `moe-quant-banks-int8`,
#154, part of epic #140).

Corrected from an earlier, unverified draft (plain unpacked int8 tensors) --
this format (int32-packed, group-aware dequant, a separate ``weight_shape``
tensor) was verified bit-exact against the real ``compressed_tensors``
library on a real checkpoint's actual tensor bytes
(``rj1013/gemma-4-26B-A4B-it_q8``); see ``int8_packed_linear.py``'s module
docstring for the full verification. This file uses small hand-packed
CPU-only fixtures -- the whole point of this issue is that these banks are
NEVER dequantized here at load time (that happens lazily, per-expert, at
compute time via ``SlotWeightAccessor``); the round-trip check below
dequantizes only for test verification, via the already-verified
``dequantize_int8_packed``.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from freetoken.kernel.triton.int8_packed_linear import dequantize_int8_packed
from freetoken.models.weight import Int8ExpertBank, stream_moe_expert_sources_int8


def _pack_int8_row(codes: list[int]) -> int:
    """4 int8 codes -> one packed int32 word (byte 0 = codes[0], matching
    compressed_tensors' own dense num_bits=8 packing -- see
    int8_packed_linear.py's module docstring)."""
    word = 0
    for i, c in enumerate(codes):
        assert -128 <= c < 128
        word |= (c + 128) << (8 * i)
    if word >= 1 << 31:
        word -= 1 << 32
    return word


def _int8_projection(k: int, n: int, group_size: int, *, code: int, scale: float):
    """A small, real (not random-garbage) pack-quantized INT8 projection:
    every element uses the same int8 code / scale, so the dequantized value
    is a known constant: ``scale * code``. ``k`` must be a multiple of 4
    (dense packing, no partial words -- every real tensor checked so far is)
    and of ``group_size``."""
    assert k % 4 == 0 and k % group_size == 0
    num_groups = k // group_size
    packed_row = [_pack_int8_row([code] * 4) for _ in range(k // 4)]
    weight_packed = torch.tensor([packed_row] * n, dtype=torch.int32)
    weight_scale = torch.full((n, num_groups), scale, dtype=torch.float32)
    weight_shape = torch.tensor([n, k], dtype=torch.int64)
    return weight_packed, weight_scale, weight_shape


def _expert_stream(config, *, gate_kwargs, up_kwargs, down_kwargs):
    """Yield ``(name, tensor)`` pairs for every expert of every layer, in the
    real per-expert INT8 key spelling (no ``mlp.`` segment -- verified
    against ``rj1013/gemma-4-26B-A4B-it_q8``'s real
    ``model.safetensors.index.json``; see ``_parse_int8_expert_key``'s
    docstring)."""
    for layer in range(config.num_layers):
        for e in range(config.num_experts):
            for proj, kwargs in (("gate_proj", gate_kwargs), ("up_proj", up_kwargs), ("down_proj", down_kwargs)):
                weight_packed, weight_scale, weight_shape = _int8_projection(**kwargs)
                prefix = f"model.language_model.layers.{layer}.experts.{e}.{proj}"
                yield prefix + ".weight_packed", weight_packed
                yield prefix + ".weight_scale", weight_scale
                yield prefix + ".weight_shape", weight_shape


def test_packed_bank_shapes_match_the_documented_contract():
    config = SimpleNamespace(num_layers=2, num_experts=3)
    k, n, group_size = 16, 8, 4
    kwargs = dict(k=k, n=n, group_size=group_size, code=5, scale=0.5)
    stream = _expert_stream(config, gate_kwargs=kwargs, up_kwargs=kwargs, down_kwargs=kwargs)

    gate_up_banks, down_banks = stream_moe_expert_sources_int8(stream, config)

    assert len(gate_up_banks) == config.num_layers
    assert len(down_banks) == config.num_layers
    for bank in gate_up_banks:
        assert isinstance(bank, Int8ExpertBank)
        # gate_up fuses gate_proj+up_proj on the N (row/output-channel) axis -> N doubles.
        assert bank.weight_packed.shape == (config.num_experts, 2 * n, k // 4)
        assert bank.weight_scale.shape == (config.num_experts, 2 * n, k // group_size)
        assert bank.weight_packed.dtype == torch.int32
        assert bank.weight_scale.dtype == torch.float32
        assert bank.k == k
    for bank in down_banks:
        assert bank.weight_packed.shape == (config.num_experts, n, k // 4)
        assert bank.weight_scale.shape == (config.num_experts, n, k // group_size)
        assert bank.k == k


def test_packed_bank_round_trips_to_the_known_fixture_value():
    config = SimpleNamespace(num_layers=1, num_experts=2)
    k, n, group_size = 16, 8, 4
    kwargs = dict(k=k, n=n, group_size=group_size, code=5, scale=0.5)  # -> 0.5 * 5 = 2.5
    stream = _expert_stream(config, gate_kwargs=kwargs, up_kwargs=kwargs, down_kwargs=kwargs)

    gate_up_banks, down_banks = stream_moe_expert_sources_int8(stream, config)

    for e in range(config.num_experts):
        down = dequantize_int8_packed(
            down_banks[0].weight_packed[e], down_banks[0].weight_scale[e], k=down_banks[0].k, out_dtype=torch.float32
        )
        torch.testing.assert_close(down, torch.full((n, k), 2.5))

        gate_up = dequantize_int8_packed(
            gate_up_banks[0].weight_packed[e], gate_up_banks[0].weight_scale[e],
            k=gate_up_banks[0].k, out_dtype=torch.float32,
        )
        torch.testing.assert_close(gate_up, torch.full((2 * n, k), 2.5))


def test_gate_up_concatenation_preserves_two_independently_quantized_halves():
    """A real test of the concat axis, not a trivial no-op: gate_proj and
    up_proj are quantized with DIFFERENT codes/scales, so the fused bank's
    two halves must dequantize back to their own distinct source values."""
    config = SimpleNamespace(num_layers=1, num_experts=1)
    k, n, group_size = 16, 8, 4
    gate_kwargs = dict(k=k, n=n, group_size=group_size, code=3, scale=0.25)  # -> 0.75
    up_kwargs = dict(k=k, n=n, group_size=group_size, code=10, scale=1.0)  # -> 10.0
    down_kwargs = dict(k=k, n=n, group_size=group_size, code=5, scale=0.5)
    stream = _expert_stream(config, gate_kwargs=gate_kwargs, up_kwargs=up_kwargs, down_kwargs=down_kwargs)

    gate_up_banks, _down_banks = stream_moe_expert_sources_int8(stream, config)
    bank = gate_up_banks[0]

    fused = dequantize_int8_packed(bank.weight_packed[0], bank.weight_scale[0], k=bank.k, out_dtype=torch.float32)
    # gate/up were concatenated on the N (row, dim=0) axis, so the two halves
    # split along rows.
    assert fused.shape == (2 * n, k)
    gate_half, up_half = fused[:n], fused[n:]
    torch.testing.assert_close(gate_half, torch.full((n, k), 0.75))
    torch.testing.assert_close(up_half, torch.full((n, k), 10.0))


def test_group_boundaries_use_the_correct_scale_per_group():
    """Two groups (group_size=4, k=8) with DIFFERENT per-group scales on the
    SAME row -- proves group indexing (not just per-row scaling) is real,
    not degenerated to the per-channel case by the fixture."""
    config = SimpleNamespace(num_layers=1, num_experts=1)
    k, n, group_size = 8, 4, 4
    packed_row = [_pack_int8_row([5] * 4), _pack_int8_row([5] * 4)]  # code=5 in both groups
    weight_packed = torch.tensor([packed_row] * n, dtype=torch.int32)
    # group 0 (cols 0-3): scale 0.5 -> 2.5; group 1 (cols 4-7): scale 2.0 -> 10.0
    weight_scale = torch.tensor([[0.5, 2.0]] * n, dtype=torch.float32)
    weight_shape = torch.tensor([n, k], dtype=torch.int64)

    def stream():
        for proj in ("gate_proj", "up_proj", "down_proj"):
            prefix = f"model.language_model.layers.0.experts.0.{proj}"
            yield prefix + ".weight_packed", weight_packed
            yield prefix + ".weight_scale", weight_scale
            yield prefix + ".weight_shape", weight_shape

    gate_up_banks, down_banks = stream_moe_expert_sources_int8(stream(), config)
    down = dequantize_int8_packed(down_banks[0].weight_packed[0], down_banks[0].weight_scale[0], k=k, out_dtype=torch.float32)
    torch.testing.assert_close(down[:, :4], torch.full((n, 4), 2.5))
    torch.testing.assert_close(down[:, 4:], torch.full((n, 4), 10.0))


def test_k_mismatch_across_experts_raises():
    config = SimpleNamespace(num_layers=1, num_experts=2)

    def stream():
        for e, k in enumerate((8, 16)):  # expert 0 vs expert 1 disagree
            kwargs = dict(k=k, n=4, group_size=k, code=5, scale=0.5)
            for proj in ("gate_proj", "up_proj", "down_proj"):
                weight_packed, weight_scale, weight_shape = _int8_projection(**kwargs)
                prefix = f"model.language_model.layers.0.experts.{e}.{proj}"
                yield prefix + ".weight_packed", weight_packed
                yield prefix + ".weight_scale", weight_scale
                yield prefix + ".weight_shape", weight_shape

    with pytest.raises(ValueError, match="K=.*differs from expert 0"):
        stream_moe_expert_sources_int8(stream(), config)


def test_gate_up_k_mismatch_raises():
    config = SimpleNamespace(num_layers=1, num_experts=1)

    def stream():
        gate_packed, gate_scale, gate_shape = _int8_projection(k=8, n=4, group_size=8, code=5, scale=0.5)
        up_packed, up_scale, up_shape = _int8_projection(k=16, n=4, group_size=16, code=5, scale=0.5)
        down_packed, down_scale, down_shape = _int8_projection(k=8, n=4, group_size=8, code=5, scale=0.5)
        prefix = "model.language_model.layers.0.experts.0"
        yield prefix + ".gate_proj.weight_packed", gate_packed
        yield prefix + ".gate_proj.weight_scale", gate_scale
        yield prefix + ".gate_proj.weight_shape", gate_shape
        yield prefix + ".up_proj.weight_packed", up_packed
        yield prefix + ".up_proj.weight_scale", up_scale
        yield prefix + ".up_proj.weight_shape", up_shape
        yield prefix + ".down_proj.weight_packed", down_packed
        yield prefix + ".down_proj.weight_scale", down_scale
        yield prefix + ".down_proj.weight_shape", down_shape

    with pytest.raises(ValueError, match="gate_proj K=.*!= up_proj K="):
        stream_moe_expert_sources_int8(stream(), config)


def test_missing_layer_raises():
    config = SimpleNamespace(num_layers=2, num_experts=1)
    kwargs = dict(k=8, n=8, group_size=8, code=5, scale=0.5)

    def stream():
        # only layer 0, layer 1 never appears
        for proj in ("gate_proj", "up_proj", "down_proj"):
            weight_packed, weight_scale, weight_shape = _int8_projection(**kwargs)
            prefix = f"model.language_model.layers.0.experts.0.{proj}"
            yield prefix + ".weight_packed", weight_packed
            yield prefix + ".weight_scale", weight_scale
            yield prefix + ".weight_shape", weight_shape

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
    kwargs = dict(k=8, n=8, group_size=8, code=5, scale=0.5)

    def stream():
        for layer in range(2):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                weight_packed, weight_scale, weight_shape = _int8_projection(**kwargs)
                prefix = f"model.language_model.layers.{layer}.experts.0.{proj}"
                yield prefix + ".weight_packed", weight_packed
                yield prefix + ".weight_scale", weight_scale
                yield prefix + ".weight_shape", weight_shape

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
