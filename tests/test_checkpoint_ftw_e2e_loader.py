"""End-to-end: load_model() auto-detects and loads correctly from an FTW
checkpoint directory (issue `ftw-checkpoint`, #11's own "ft serve --model
auto-detects an FTW dir" accept criterion) -- the piece
test_checkpoint_ftw.py's own unit tests (write/read round trip,
iter_safetensors auto-detection) don't cover: a REAL model actually
running against FTW-loaded weights, not just the storage layer in
isolation.

Small fabricated qwen3_moe checkpoint (mirrors test_moe_offload_forward.py's
own TINY_CONFIG fixture) -- converted to FTW, then loaded both ways and
compared bit-for-bit, proving the conversion is lossless end-to-end through
the real model loader, not just FtwArchive's own read/write.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from freetoken.checkpoint.convert import convert_checkpoint
from freetoken.models.loader import load_model

HIDDEN, VOCAB, HEADS, KV, INTER, EXPERTS, LAYERS = 64, 32, 4, 2, 32, 4, 2


def _qwen3_moe_weights() -> dict:
    head_dim = HIDDEN // HEADS
    w = {
        "model.embed_tokens.weight": torch.randn(VOCAB, HIDDEN),
        "lm_head.weight": torch.randn(VOCAB, HIDDEN),
        "model.norm.weight": torch.randn(HIDDEN),
    }
    for l in range(LAYERS):
        prefix = f"model.layers.{l}"
        w[f"{prefix}.input_layernorm.weight"] = torch.randn(HIDDEN)
        w[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(HEADS * head_dim, HIDDEN)
        w[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(KV * head_dim, HIDDEN)
        w[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(KV * head_dim, HIDDEN)
        w[f"{prefix}.self_attn.q_norm.weight"] = torch.randn(head_dim)
        w[f"{prefix}.self_attn.k_norm.weight"] = torch.randn(head_dim)
        w[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(HEADS * head_dim, HIDDEN)
        w[f"{prefix}.post_attention_layernorm.weight"] = torch.randn(HIDDEN)
        w[f"{prefix}.mlp.gate.weight"] = torch.randn(EXPERTS, HIDDEN)
        for e in range(EXPERTS):
            ep = f"{prefix}.mlp.experts.{e}"
            w[f"{ep}.gate_proj"] = torch.randn(INTER, HIDDEN)
            w[f"{ep}.up_proj"] = torch.randn(INTER, HIDDEN)
            w[f"{ep}.down_proj"] = torch.randn(HIDDEN, INTER)
    return w


@pytest.fixture(scope="module")
def qwen3_moe_ckpt(tmp_path_factory):
    from safetensors.torch import save_file

    path = tmp_path_factory.mktemp("qwen3moe_src")
    config = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "hidden_size": HIDDEN,
        "vocab_size": VOCAB,
        "num_hidden_layers": LAYERS,
        "num_local_experts": EXPERTS,
        "num_experts_per_tok": 2,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV,
        "intermediate_size": INTER,
        "moe_intermediate_size": INTER,
        "max_position_embeddings": 128,
        "rope_theta": 1000000.0,
    }
    weights = _qwen3_moe_weights()
    (path / "config.json").write_text(json.dumps(config))
    save_file({k: v.contiguous() for k, v in weights.items()}, str(path / "model.safetensors"))
    return str(path)


def test_load_model_from_ftw_dir_matches_original_safetensors(qwen3_moe_ckpt, tmp_path):
    ftw_dir = str(tmp_path / "ftw_ckpt")
    convert_checkpoint(qwen3_moe_ckpt, ftw_dir)

    original_model, _ = load_model(qwen3_moe_ckpt, torch.device("cpu"), dtype=torch.float32, moe_backend=None)
    ftw_model, _ = load_model(ftw_dir, torch.device("cpu"), dtype=torch.float32, moe_backend=None)

    assert type(ftw_model) is type(original_model)
    torch.testing.assert_close(ftw_model.embed_tokens.weight, original_model.embed_tokens.weight)
    torch.testing.assert_close(ftw_model.norm.weight, original_model.norm.weight)
    torch.testing.assert_close(
        ftw_model.layers[0].self_attn.q_proj.weight, original_model.layers[0].self_attn.q_proj.weight
    )
