"""Tests for GGUF KV-metadata -> ModelConfig mapping (issue
`models-gguf-config-tokenizer`, #272).

Uses the same real GGUF fixture `test_gguf_reader.py` already downloads
(`ggml-org/models`' `tinyllamas/stories260K.gguf`, a ``llama``-architecture
checkpoint) for consistency. Per the issue's own test strategy, the
config-mapping logic is architecture-keyed but architecture-agnostic in
*spelling* (every architecture uses the same ``{arch}.block_count`` etc.
convention), so this fixture's ``llama`` architecture already exercises the
generic mapping end to end even though the real target family is
``qwen35moe``. The ``qwen35moe``-specific (Qwen3.5/3.6 linear-attention)
branch is covered separately below with synthetic metadata, since no tiny
real ``qwen35moe`` GGUF fixture is available.
"""
from __future__ import annotations

import pytest

from freetoken.models.config import ModelConfig
from freetoken.models.gguf import GGUFFormatError, parse_config

_HF_REPO = "ggml-org/models"
_F32_FIXTURE = "tinyllamas/stories260K.gguf"


def _hf_download(filename: str) -> str:
    huggingface_hub = pytest.importorskip("huggingface_hub")
    try:
        return huggingface_hub.hf_hub_download(_HF_REPO, filename)
    except Exception as exc:  # pragma: no cover - network/offline environment
        pytest.skip(f"could not download real GGUF fixture {filename} from the Hub: {exc}")


@pytest.fixture(scope="module")
def f32_gguf_path() -> str:
    return _hf_download(_F32_FIXTURE)


@pytest.mark.slow
def test_parse_config_matches_hand_verified_fixture_hyperparameters(f32_gguf_path):
    # Same hand-verified values test_gguf_reader.py's own
    # test_load_gguf_metadata_matches_hand_verified_values asserts against the
    # raw metadata: block_count=5, context_length=2048, embedding_length=64,
    # feed_forward_length=172, attention.head_count=8, head_count_kv=4.
    cfg = parse_config(f32_gguf_path)
    assert isinstance(cfg, ModelConfig)
    assert cfg.hidden_size == 64
    assert cfg.num_layers == 5
    assert cfg.intermediate_size == 172
    assert cfg.num_attention_heads == 8
    assert cfg.num_key_value_heads == 4
    assert cfg.max_position_embeddings == 2048
    # No explicit {arch}.vocab_size KV in this fixture: falls back to the
    # embedded tokenizer's own token-list length (512, hand-verified in
    # test_gguf_reader.py).
    assert cfg.vocab_size == 512
    # A plain dense llama checkpoint: no expert_count KV at all.
    assert cfg.is_moe is False
    assert cfg.num_experts is None


@pytest.mark.slow
def test_parse_config_accepts_pre_parsed_metadata_dict(f32_gguf_path):
    from freetoken.models.gguf import load_gguf_metadata

    meta = load_gguf_metadata(f32_gguf_path)
    cfg = parse_config(meta)
    assert cfg.hidden_size == 64
    assert cfg.num_layers == 5


@pytest.mark.slow
def test_parse_config_passes_through_backend_flags(f32_gguf_path):
    cfg = parse_config(
        f32_gguf_path,
        use_offload_moe=True,
        use_cpu_moe=False,
        use_hybrid=True,
        moe_cpu_layers="auto",
        model_path=f32_gguf_path,
    )
    assert cfg.use_offload_moe is True
    assert cfg.use_hybrid is True
    assert cfg.moe_cpu_layers == "auto"


def test_parse_config_raises_on_missing_architecture():
    with pytest.raises(GGUFFormatError):
        parse_config({"general.name": "no-arch-here"})


def test_parse_config_maps_generic_moe_fields_from_synthetic_metadata():
    # A synthetic qwen3moe-shaped metadata dict: real key spellings (verified
    # against the real ggml-org/llama.cpp `gguf-py/gguf/constants.py` and
    # `conversion/qwen.py` while implementing this), fabricated values --
    # exercises the generic dense-MoE mapping (_MOE_KEYS) without needing a
    # real qwen3moe GGUF fixture on disk for this one shape check.
    meta = {
        "general.architecture": "qwen3moe",
        "general.name": "qwen3moe-test",
        "qwen3moe.block_count": 4,
        "qwen3moe.embedding_length": 128,
        "qwen3moe.feed_forward_length": 256,
        "qwen3moe.attention.head_count": 8,
        "qwen3moe.attention.head_count_kv": 2,
        "qwen3moe.expert_count": 16,
        "qwen3moe.expert_used_count": 4,
        "qwen3moe.expert_feed_forward_length": 64,
        "qwen3moe.leading_dense_block_count": 1,
        "qwen3moe.rope.freq_base": 1000000.0,
    }
    cfg = parse_config(meta)
    assert cfg.hidden_size == 128
    assert cfg.num_layers == 4
    assert cfg.is_moe is True
    assert cfg.num_experts == 16
    assert cfg.num_experts_per_tok == 4
    assert cfg.moe_intermediate_size == 64
    assert cfg.first_k_dense_replace == 1
    # num_moe_layers derived: total layers minus the dense prefix.
    assert cfg.num_moe_layers == 3
    assert cfg.rope_theta == pytest.approx(1000000.0)


def test_parse_config_maps_qwen35moe_linear_attention_attrs_from_synthetic_metadata():
    # The real target family (epic #199's "Why": Qwen3.6-35B-A3B). Key
    # spellings verified against conversion/qwen.py's Qwen3NextModel /
    # Qwen3_5MoeTextModel.set_gguf_parameters (ggml-org/llama.cpp@master) --
    # qwen35moe inherits Qwen3NextModel's SSM/full-attention-interval keys via
    # _LinearAttentionVReorderBase. No tiny real qwen35moe GGUF fixture is
    # available, so this is synthetic (the generic-mapping path above is
    # covered against a real file).
    meta = {
        "general.architecture": "qwen35moe",
        "general.name": "qwen3.6-test",
        "qwen35moe.block_count": 6,
        "qwen35moe.embedding_length": 256,
        "qwen35moe.feed_forward_length": 512,
        "qwen35moe.attention.head_count": 16,
        "qwen35moe.attention.head_count_kv": 2,
        "qwen35moe.attention.key_length": 128,
        "qwen35moe.expert_count": 256,
        "qwen35moe.expert_used_count": 8,
        "qwen35moe.expert_feed_forward_length": 96,
        "qwen35moe.ssm.conv_kernel": 4,
        "qwen35moe.ssm.state_size": 128,
        "qwen35moe.ssm.group_count": 16,
        "qwen35moe.ssm.time_step_rank": 32,
        "qwen35moe.ssm.inner_size": 128 * 32,
        "qwen35moe.full_attention_interval": 4,
        "qwen35moe.attention.recurrent_layers": [True, True, True, False, True, True],
        "qwen35moe.rope.dimension_count": 32,
    }
    cfg = parse_config(meta)
    assert cfg.hidden_size == 256
    assert cfg.head_dim == 128
    assert cfg.is_moe is True
    assert cfg.num_experts == 256
    assert cfg.num_experts_per_tok == 8

    linear = cfg.attrs["gguf_linear_attention"]
    assert linear["linear_conv_kernel_dim"] == 4
    assert linear["linear_key_head_dim"] == 128
    assert linear["linear_num_key_heads"] == 16
    assert linear["linear_num_value_heads"] == 32
    assert linear["linear_value_head_dim"] == 128
    assert linear["full_attention_interval"] == 4
    assert linear["recurrent_layers"] == [True, True, True, False, True, True]
    assert linear["partial_rotary_dim"] == 32


def test_import_is_torch_free():
    import freetoken.models.gguf as gguf_mod

    assert gguf_mod is not None
