"""Tests for the Qwen3.5/3.6 (``qwen3_5_moe``) config adapter -- torch-free.

The hero model is a **hybrid-attention multimodal MoE**: 30 linear-attention
(Gated-DeltaNet) layers + 10 full-GQA layers, a 256-way router top-8 **plus an
always-on shared expert** per MoE layer, and a text tower nested under
``text_config`` (the vision tower is out of scope for text serving).

``parse_config`` is the one checkpoint adapter that needs no torch, so it runs
on the CPU-only ``.venv`` (no torch there). The weight-routing half
(``iter_weights``) and the full ``load_model`` path need torch and live in
``test_models_qwen35_loader.py`` (torch-gated; run under the XPU venv / nightly).

The forward pass is a later issue; here we only assert the parsed config carries
the fields the loader reads (the MoE bank fabricator keys off ``is_moe`` /
``num_moe_layers`` / ``num_experts`` / ``moe_intermediate_size``) and stashes the
full raw ``text_config`` (``layer_types``, partial-RoPE, the output-gate flag,
the linear-attention head dims) in ``config.attrs`` for the forward pass.
"""
from __future__ import annotations

from freetoken.models.config import ModelConfig
from freetoken.models.qwen3_5_moe import parse_config
from freetoken.utils.hf import RawConfigShim

# A text tower shaped like the real Qwen3.6-35B-A3B config (small dims so a test
# could also run it under torch if needed). 4 layers, every 4th full-attention
# (3 linear + 1 full), matching the hero model's full_attention_interval=4.
_TEXT_CONFIG = {
    "hidden_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512,
    "vocab_size": 1000,
    "max_position_embeddings": 4096,
    "full_attention_interval": 4,
    "layer_types": [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ],
    "partial_rotary_factor": 0.25,
    "attn_output_gate": True,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.25},
    "rope_theta": 10000000.0,
}


def _hf_config(overrides=None):
    """A multimodal HF-style config (nested ``text_config``), as ``config.json``
    ships it: the top level carries the vision fields, the language tower is
    nested under ``text_config``."""
    text_config = dict(_TEXT_CONFIG)
    if overrides:
        text_config.update(overrides)
    return RawConfigShim(
        {
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "vision_config": {"hidden_size": 7, "num_chunks": 3},
            "text_config": text_config,
        }
    )


# --- parse_config: the fields the loader reads --------------------------------


def test_parse_config_multimodal_reads_text_config():
    cfg = parse_config(_hf_config())
    assert cfg.architectures == ["Qwen3_5MoeForConditionalGeneration"]
    # Read off the nested text tower, not the vision top-level.
    assert cfg.hidden_size == 128
    assert cfg.vocab_size == 1000
    assert cfg.num_layers == 4
    assert cfg.num_attention_heads == 16
    assert cfg.num_key_value_heads == 2
    assert cfg.max_position_embeddings == 4096
    assert cfg.rope_theta == 10000000.0


def test_parse_config_moe_plumbing_fields():
    cfg = parse_config(_hf_config())
    # The MoE bank fabricator / loader keys off these first-class fields.
    assert cfg.is_moe is True
    assert cfg.num_experts == 256
    assert cfg.moe_intermediate_size == 512
    assert cfg.num_experts_per_tok == 8
    # No leading dense layers -> every layer is MoE (num_moe_layers = 4).
    assert cfg.first_k_dense_replace == 0
    assert cfg.num_moe_layers == 4
    assert isinstance(cfg, ModelConfig)


def test_parse_config_stows_full_text_config_in_attrs():
    cfg = parse_config(_hf_config())
    tc = cfg.attrs["text_config"]
    assert tc["layer_types"] == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    assert tc["full_attention_interval"] == 4
    assert tc["attn_output_gate"] is True
    assert tc["partial_rotary_factor"] == 0.25
    assert tc["head_dim"] == 256
    assert tc["linear_num_key_heads"] == 16
    assert tc["linear_num_value_heads"] == 32
    assert tc["shared_expert_intermediate_size"] == 512
    # The hybrid split the forward pass will consume: 3 linear + 1 full.
    assert sum(1 for t in tc["layer_types"] if t == "full_attention") == 1
    assert sum(1 for t in tc["layer_types"] if t == "linear_attention") == 3


def test_parse_config_offload_flag_is_not_a_ckpt_field():
    assert parse_config(_hf_config()).use_offload_moe is False
    assert parse_config(_hf_config(), use_offload_moe=True).use_offload_moe is True


def test_parse_config_flat_config_falls_back_to_top_level():
    # A config that is already the flat text object (no text_config nesting, e.g.
    # a hand-built one) must parse using its own top-level fields.
    flat = RawConfigShim(
        {
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 16,
            "vocab_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
        }
    )
    cfg = parse_config(flat)
    assert cfg.hidden_size == 64
    assert cfg.num_layers == 2
    assert cfg.num_experts == 4
    assert cfg.is_moe is True
    assert cfg.num_moe_layers == 2


def test_parse_config_expert_count_spelling():
    # The real Qwen3.5/3.6 config stores the count under num_experts; a
    # transformers-style config may store it under num_local_experts. Both must
    # resolve to the same place (num_local_experts takes precedence if present).
    assert parse_config(_hf_config({"num_experts": 256})).num_experts == 256
    cfg = parse_config(_hf_config({"num_experts": 7, "num_local_experts": 13}))
    assert cfg.num_experts == 13


# (Instantiating the model class needs torch -- the constructor rebinds the
# instance to an nn.Module -- so that lives in the torch-gated loader tests,
# not here. This file must stay importable and runnable on the torch-free CPU
# box.)
