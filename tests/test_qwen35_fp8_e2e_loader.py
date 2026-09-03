"""End-to-end: load_model() wires a block-FP8-quantized checkpoint's experts
into a real "fp8_block" OffloadMoeCache and runs a forward step (issue
`moe-quant-banks-native-multi`, #163's own noted coverage gap: block-FP8
had only SlotWeightAccessor-level XPU coverage, not a full checkpoint-
loader test the way GPTQ's #138 has).

Mirrors test_qwen35_gptq_e2e_loader.py's own approach exactly -- a small
(few-KB) fabricated block-FP8 checkpoint, not a real multi-GB one, proving
the load_moe_expert_sources -> _attach_offload_cache -> SlotWeightAccessor
wiring is correct before ever touching real hardware / a real checkpoint
at scale. Real block-FP8 checkpoints (DeepSeek-V3) use a different model
architecture (deepseek_v3, not yet ported -- see issue #21) than this
fixture's qwen3_5_moe shell; that's fine, since block-FP8's own
quant_method/key-spelling detection in load_moe_expert_sources is
architecture-independent (it dispatches purely off quantization_config +
per-expert key spelling, not the model type) -- this test's job is to
prove FreeToken's OWN loader wiring is correct, not that a real Qwen3.5
checkpoint ships in this format.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from freetoken.kernel.triton.fp8_block_linear import quantize_block_fp8
from freetoken.models.loader import load_model

from tests.test_qwen35_gptq_e2e_loader import _drive_prefill

H, I, E, V, L = 64, 128, 4, 64, 1
# stream_moe_expert_sources_fp8's own dispatch (load_moe_expert_sources)
# hardcodes block=128 (never reads it from the checkpoint's own
# quantization_config.weight_block_size) -- I (moe_intermediate_size) must
# be a multiple of it so the gate_proj/up_proj N-axis fuse-concat's own
# alignment check (see _finalize_fp8_bank) doesn't reject this fixture.
BLOCK = 128
NH, NKV, HD = 4, 2, 16
Q_PROJ_DIM = NH * HD * 2
O_PROJ_DIM = NH * HD
KV_PROJ_DIM = NKV * HD


def _fp8_expert_weights() -> dict:
    """One full-attention layer's dense weights (unquantized -- attention is
    excluded from FP8 quantization, matching the real DeepSeek-V3 convention)
    plus block-FP8-packed routed experts and a plain (unquantized) shared
    expert -- mirrors _qwen35_gptq_weights' own structure exactly."""
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
    for e in range(E):
        base = f"model.language_model.layers.0.mlp.experts.{e}"
        for proj, n, k in (("gate_proj", I, H), ("up_proj", I, H), ("down_proj", H, I)):
            dense = torch.randn(n, k) * ((e + 1) * 0.1)  # distinguishable per-expert magnitude
            weight_fp8, scale = quantize_block_fp8(dense, block=BLOCK)
            w[f"{base}.{proj}.weight"] = weight_fp8
            w[f"{base}.{proj}.weight_scale_inv"] = scale
    return w


@pytest.fixture(scope="module")
def qwen35_fp8_ckpt(tmp_path_factory):
    from safetensors.torch import save_file

    path = tmp_path_factory.mktemp("qwen35_fp8")
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
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [BLOCK, BLOCK],
        },
    }
    weights = _fp8_expert_weights()
    (path / "config.json").write_text(json.dumps(config))
    save_file({k: v.contiguous() for k, v in weights.items()}, str(path / "model.safetensors"))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in weights}})
    )
    return str(path)


def test_load_model_builds_an_fp8_block_offload_cache(qwen35_fp8_ckpt):
    model, _ = load_model(qwen35_fp8_ckpt, torch.device("cpu"), dtype=torch.float32, moe_backend="offload")

    cache = model.moe_cache
    assert cache.quant_format == "fp8_block"
    assert set(cache.bank_sources) == {
        "weight_gate_up", "scale_gate_up", "weight_down", "scale_down",
    }


def test_fp8_checkpoint_forward_step_produces_finite_logits(qwen35_fp8_ckpt):
    """The real point of this test (mirrors #138's own for GPTQ): a
    block-FP8-quantized checkpoint's forward pass runs end-to-end --
    SlotWeightAccessor dequantizes the resident packed experts on the fly,
    reading from the cache load_moe_expert_sources/_attach_offload_cache
    actually built."""
    model, _ = load_model(qwen35_fp8_ckpt, torch.device("cpu"), dtype=torch.float32, moe_backend="offload")
    seq = torch.randint(0, V, (5,))
    T = seq.shape[0]
    logits, _ = _drive_prefill(model, seq, T)
    assert logits.shape == (1, V)
    assert torch.isfinite(logits).all()
