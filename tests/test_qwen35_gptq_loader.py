"""qwen3_5_moe.iter_weights: GPTQ-packed projections dequantize correctly
before entering the rest of the loader pipeline (issue quant-xpu, #10).

Found while preparing to load the real Qwen/Qwen3.5-35B-A3B-GPTQ-Int4
checkpoint: only the routed experts' gate_proj/up_proj/down_proj are ever
GPTQ-quantized in that checkpoint (per its own quantization_config.dynamic
exclusions) -- everything else stays plain bf16/fp16. These tests exercise
_dequantize_gptq_stream directly (the narrow, real thing this PR adds) with
small hand-built tensors, independent of the full multimodal fixture in
test_models_qwen35_loader.py (which already covers the rest of iter_weights
end to end for a plain, non-quantized checkpoint).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from freetoken.kernel.triton.gptq_linear import dequantize_gptq_int4
from freetoken.models.qwen3_5_moe import _dequantize_gptq_stream


def _pack_nibbles(codes: list[int]) -> int:
    word = 0
    for i, c in enumerate(codes):
        word |= c << (4 * i)
    # Fold a word with bit 31 set (e.g. every nibble = 8) to its two's-
    # complement negative value -- torch.tensor(..., dtype=torch.int32)
    # rejects a Python int >= 2**31 as an overflow otherwise.
    if word >= 1 << 31:
        word -= 1 << 32
    return word


def _fake_gptq_projection(k: int, n: int, group_size: int):
    """A small, real (not random-garbage) GPTQ-packed projection: constant
    weight code 5, zero-point 8 (stored directly, no +1 correction -- issue
    #147), scale 0.5 -- every dequantized element is 0.5 * (5 - 8) = -1.5."""
    assert k % 8 == 0 and n % 8 == 0 and k % group_size == 0
    n_groups = k // group_size
    qweight = torch.tensor([[_pack_nibbles([5] * 8)] * n for _ in range(k // 8)], dtype=torch.int32)
    qzeros = torch.tensor([[_pack_nibbles([8] * 8)] * (n // 8) for _ in range(n_groups)], dtype=torch.int32)
    scales = torch.full((n_groups, n), 0.5)
    g_idx = torch.tensor([i // group_size for i in range(k)], dtype=torch.int32)
    return qweight, qzeros, scales, g_idx


def test_passthrough_for_a_stream_with_no_gptq_tensors():
    pairs = [("a.weight", torch.randn(4, 4)), ("b.bias", torch.randn(4))]
    out = list(_dequantize_gptq_stream(iter(pairs)))
    assert [name for name, _ in out] == ["a.weight", "b.bias"]
    for (_, expected), (_, got) in zip(pairs, out):
        torch.testing.assert_close(got, expected)


def test_dequantizes_a_complete_gptq_projection_to_a_single_weight_tensor():
    k, n, group_size = 16, 8, 16
    qweight, qzeros, scales, g_idx = _fake_gptq_projection(k, n, group_size)
    prefix = "model.language_model.layers.0.mlp.experts.0.gate_proj"
    pairs = [
        (prefix + ".qweight", qweight),
        (prefix + ".qzeros", qzeros),
        (prefix + ".scales", scales),
        (prefix + ".g_idx", g_idx),
    ]
    out = list(_dequantize_gptq_stream(iter(pairs)))
    assert len(out) == 1
    name, tensor = out[0]
    assert name == prefix + ".weight"
    # nn.Linear convention: [out_features, in_features] = [n, k].
    assert tensor.shape == (n, k)
    torch.testing.assert_close(tensor, torch.full((n, k), -1.5, dtype=tensor.dtype), atol=1e-2, rtol=0)


def test_dequantized_tensor_matches_dequantize_gptq_int4_transposed():
    k, n, group_size = 16, 8, 16
    qweight, qzeros, scales, g_idx = _fake_gptq_projection(k, n, group_size)
    prefix = "p"
    pairs = [(prefix + s, t) for s, t in zip((".qweight", ".qzeros", ".scales", ".g_idx"), (qweight, qzeros, scales, g_idx))]
    (_, got) = next(iter(_dequantize_gptq_stream(iter(pairs))))
    expected = dequantize_gptq_int4(qweight, qzeros, scales, g_idx, out_dtype=torch.bfloat16).T.contiguous()
    torch.testing.assert_close(got, expected)


def test_gptq_and_plain_tensors_interleave_correctly():
    k, n, group_size = 8, 8, 8
    qweight, qzeros, scales, g_idx = _fake_gptq_projection(k, n, group_size)
    prefix = "model.language_model.layers.0.mlp.experts.0.down_proj"
    pairs = [
        ("model.language_model.embed_tokens.weight", torch.randn(4, 4)),
        (prefix + ".qweight", qweight),
        ("model.language_model.layers.0.linear_attn.norm.weight", torch.randn(4)),
        (prefix + ".g_idx", g_idx),
        (prefix + ".scales", scales),
        (prefix + ".qzeros", qzeros),
    ]
    out = dict(_dequantize_gptq_stream(iter(pairs)))
    assert set(out) == {
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.linear_attn.norm.weight",
        prefix + ".weight",
    }


def test_raises_on_an_incomplete_projection_at_stream_end():
    k, n, group_size = 8, 8, 8
    qweight, qzeros, scales, _g_idx = _fake_gptq_projection(k, n, group_size)
    pairs = [("p.qweight", qweight), ("p.qzeros", qzeros), ("p.scales", scales)]  # missing g_idx
    with pytest.raises(ValueError, match="incomplete"):
        list(_dequantize_gptq_stream(iter(pairs)))


# ---------------------------------------------------------------------------
# iter_weights bank-only mode (issue moe-quant-banks-pack, #135): GPTQ
# tensors must pass through RAW (not dequantized) when include_moe_experts=
# True, include_non_moe=False -- the offload MoE-bank fabricator's call
# shape. Dequantizing every expert to bf16 at load time is a 4x expansion
# that blows the host RAM budget for a real-scale checkpoint (issue #134).
# ---------------------------------------------------------------------------


def test_bank_only_mode_yields_raw_gptq_components_not_dequantized(monkeypatch, tmp_path):
    import freetoken.models.qwen3_5_moe as qwen35moe

    k, n, group_size = 8, 8, 8
    qweight, qzeros, scales, g_idx = _fake_gptq_projection(k, n, group_size)
    prefix = "model.language_model.layers.0.mlp.experts.0.gate_proj"
    raw_pairs = [
        (prefix + ".qweight", qweight),
        (prefix + ".qzeros", qzeros),
        (prefix + ".scales", scales),
        (prefix + ".g_idx", g_idx),
    ]

    monkeypatch.setattr(qwen35moe, "_iter_safetensors", lambda model_path, device: iter(raw_pairs))
    monkeypatch.setattr(qwen35moe, "_cfg_for_path", lambda model_path: None)  # fall back to the substring heuristic

    out = dict(
        qwen35moe.iter_weights(
            "unused-path", torch.device("cpu"), include_moe_experts=True, include_non_moe=False
        )
    )
    # Raw component names survive (remapped model.language_model.* -> model.*),
    # not a single dequantized "...gate_proj.weight".
    assert set(out) == {
        "model.layers.0.mlp.experts.0.gate_proj.qweight",
        "model.layers.0.mlp.experts.0.gate_proj.qzeros",
        "model.layers.0.mlp.experts.0.gate_proj.scales",
        "model.layers.0.mlp.experts.0.gate_proj.g_idx",
    }
    assert out["model.layers.0.mlp.experts.0.gate_proj.qweight"].dtype == torch.int32
    torch.testing.assert_close(out["model.layers.0.mlp.experts.0.gate_proj.qweight"], qweight)
    torch.testing.assert_close(out["model.layers.0.mlp.experts.0.gate_proj.g_idx"], g_idx)


def test_bank_only_mode_still_drops_non_expert_tensors(monkeypatch):
    import freetoken.models.qwen3_5_moe as qwen35moe

    k, n, group_size = 8, 8, 8
    qweight, qzeros, scales, g_idx = _fake_gptq_projection(k, n, group_size)
    prefix = "model.language_model.layers.0.mlp.experts.0.down_proj"
    raw_pairs = [
        ("model.language_model.embed_tokens.weight", torch.randn(4, 4)),
        (prefix + ".qweight", qweight),
        (prefix + ".qzeros", qzeros),
        (prefix + ".scales", scales),
        (prefix + ".g_idx", g_idx),
        ("model.language_model.layers.0.linear_attn.norm.weight", torch.randn(4)),
    ]
    monkeypatch.setattr(qwen35moe, "_iter_safetensors", lambda model_path, device: iter(raw_pairs))
    monkeypatch.setattr(qwen35moe, "_cfg_for_path", lambda model_path: None)

    out = dict(
        qwen35moe.iter_weights(
            "unused-path", torch.device("cpu"), include_moe_experts=True, include_non_moe=False
        )
    )
    assert set(out) == {
        "model.layers.0.mlp.experts.0.down_proj.qweight",
        "model.layers.0.mlp.experts.0.down_proj.qzeros",
        "model.layers.0.mlp.experts.0.down_proj.scales",
        "model.layers.0.mlp.experts.0.down_proj.g_idx",
    }


def test_non_bank_only_mode_still_dequantizes_eagerly(monkeypatch):
    """The non-bank-only call shape (e.g. a hypothetical fully-in-VRAM
    consumer) keeps today's eager-dequant behavior -- only the offload
    bank-fabricator's call shape (include_non_moe=False) changed."""
    import freetoken.models.qwen3_5_moe as qwen35moe

    k, n, group_size = 8, 8, 8
    qweight, qzeros, scales, g_idx = _fake_gptq_projection(k, n, group_size)
    prefix = "model.language_model.layers.0.mlp.experts.0.gate_proj"
    raw_pairs = [
        (prefix + ".qweight", qweight),
        (prefix + ".qzeros", qzeros),
        (prefix + ".scales", scales),
        (prefix + ".g_idx", g_idx),
    ]
    monkeypatch.setattr(qwen35moe, "_iter_safetensors", lambda model_path, device: iter(raw_pairs))
    monkeypatch.setattr(qwen35moe, "_cfg_for_path", lambda model_path: None)

    out = dict(
        qwen35moe.iter_weights(
            "unused-path", torch.device("cpu"), include_moe_experts=True, include_non_moe=True
        )
    )
    assert set(out) == {"model.layers.0.mlp.experts.0.gate_proj.weight"}
    assert out["model.layers.0.mlp.experts.0.gate_proj.weight"].dtype == torch.bfloat16
