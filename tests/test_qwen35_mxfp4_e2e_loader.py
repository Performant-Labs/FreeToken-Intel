"""End-to-end: load_model() wires an MXFP4-quantized checkpoint's experts
into a real "mxfp4" OffloadMoeCache and runs a forward step (issue
`moe-quant-banks-native-multi`, #163's own noted coverage gap: MXFP4 had
only SlotWeightAccessor-level XPU coverage, not a full checkpoint-loader
test the way GPTQ's #138 has).

Mirrors test_qwen35_gptq_e2e_loader.py's / test_qwen35_fp8_e2e_loader.py's
own approach -- a small (few-KB) fabricated MXFP4 checkpoint, not a real
one, proving the load_moe_expert_sources -> _attach_offload_cache ->
SlotWeightAccessor wiring is correct. Real MXFP4 checkpoints
(openai/gpt-oss-20b) use a different model architecture (gpt_oss, not yet
ported -- see issue #23) than this fixture's qwen3_5_moe shell; see
test_qwen35_fp8_e2e_loader.py's own docstring for why that's fine (the
dispatch is architecture-independent, keyed off quantization_config + key
spelling).

Unlike GPTQ/FP8's raw per-expert-indexed layout, a real MXFP4 checkpoint
ships each (layer, projection, component) as ONE already-fused
``[num_experts, ...]`` tensor (see stream_moe_expert_sources_mxfp4's own
docstring) -- gate_proj/up_proj are already pre-fused into one
``gate_up_proj`` tensor upstream too, so this fixture's weight-building
looks structurally different from GPTQ's/FP8's per-expert loop.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from freetoken.kernel.triton.mxfp4_linear import quantize_mxfp4_blocks
from freetoken.models.loader import load_model

from tests.test_qwen35_gptq_e2e_loader import _drive_prefill

H, I, E, V, L = 64, 32, 4, 64, 1
NH, NKV, HD = 4, 2, 16
Q_PROJ_DIM = NH * HD * 2
O_PROJ_DIM = NH * HD
KV_PROJ_DIM = NKV * HD


def _mxfp4_expert_weights() -> dict:
    """One full-attention layer's dense weights (unquantized -- attention is
    excluded from MXFP4 quantization, matching the real gpt-oss convention)
    plus MXFP4-packed routed experts (fused per-projection, not per-expert
    -- see this module's own docstring) and a plain (unquantized) shared
    expert."""
    w = {
        "model.language_model.embed_tokens.weight": torch.randn(V, H),
        "model.language_model.norm.weight": torch.randn(H),
        "lm_head.weight": torch.randn(V, H),
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(Q_PROJ_DIM, H),
        "model.language_model.layers.0.self_attn.k_proj.weight": torch.randn(KV_PROJ_DIM, H),
        "model.language_model.layers.0.self_attn.v_proj.weight": torch.randn(KV_PROJ_DIM, H),
        "model.language_model.layers.0.self_attn.o_proj.weight": torch.randn(H, O_PROJ_DIM),
        "model.language_model.layers.0.self_attn.q_norm.weight": torch.randn(HD),
        "model.language_model.layers.0.self_attn.k_norm.weight": torch.randn(HD),
        "model.language_model.layers.0.mlp.gate.weight": torch.randn(E, H),
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": torch.randn(I, H),
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight": torch.randn(I, H),
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight": torch.randn(H, I),
        "model.language_model.layers.0.mlp.shared_expert_gate.weight": torch.randn(1, H),
    }
    base = "model.language_model.layers.0.mlp.experts"
    gate_up_dense = torch.randn(E, 2 * I, H) * 0.3
    down_dense = torch.randn(E, H, I) * 0.3
    gu_blocks, gu_scales = quantize_mxfp4_blocks(gate_up_dense)
    dn_blocks, dn_scales = quantize_mxfp4_blocks(down_dense)
    w[f"{base}.gate_up_proj_blocks"] = gu_blocks
    w[f"{base}.gate_up_proj_scales"] = gu_scales
    w[f"{base}.down_proj_blocks"] = dn_blocks
    w[f"{base}.down_proj_scales"] = dn_scales
    return w


@pytest.fixture(scope="module")
def qwen35_mxfp4_ckpt(tmp_path_factory):
    from safetensors.torch import save_file

    path = tmp_path_factory.mktemp("qwen35_mxfp4")
    text_config = {
        "hidden_size": H,
        "num_hidden_layers": L,
        "num_attention_heads": NH,
        "num_key_value_heads": NKV,
        "num_experts": E,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": I,
        "shared_expert_intermediate_size": I,
        "vocab_size": V,
        "max_position_embeddings": 128,
        "head_dim": HD,
        "attn_output_gate": False,
        "partial_rotary_factor": 0.5,
        "full_attention_interval": 1,
        "layer_types": ["full_attention"],
        "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.5},
    }
    config = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "tie_word_embeddings": True,
        "text_config": text_config,
        "quantization_config": {"quant_method": "mxfp4"},
    }
    weights = _mxfp4_expert_weights()
    (path / "config.json").write_text(json.dumps(config))
    save_file({k: v.contiguous() for k, v in weights.items()}, str(path / "model.safetensors"))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in weights}})
    )
    return str(path)


def test_load_model_builds_an_mxfp4_offload_cache(qwen35_mxfp4_ckpt):
    model, _ = load_model(qwen35_mxfp4_ckpt, torch.device("cpu"), dtype=torch.float32, moe_backend="offload")

    cache = model.moe_cache
    assert cache.quant_format == "mxfp4"
    assert set(cache.bank_sources) == {
        "blocks_gate_up", "scales_gate_up", "blocks_down", "scales_down",
    }


def test_mxfp4_checkpoint_forward_step_produces_finite_logits(qwen35_mxfp4_ckpt):
    """The real point of this test (mirrors #138's own for GPTQ): an
    MXFP4-quantized checkpoint's forward pass runs end-to-end --
    SlotWeightAccessor dequantizes the resident packed experts on the fly,
    reading from the cache load_moe_expert_sources/_attach_offload_cache
    actually built."""
    model, _ = load_model(qwen35_mxfp4_ckpt, torch.device("cpu"), dtype=torch.float32, moe_backend="offload")
    seq = torch.randint(0, V, (5,))
    T = seq.shape[0]
    logits, _ = _drive_prefill(model, seq, T)
    assert logits.shape == (1, V)
    assert torch.isfinite(logits).all()
