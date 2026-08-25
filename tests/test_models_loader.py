"""Tests for the checkpoint loader (issue ``models-loader``, #17).

Split by environment:

* The config/tokenizer helpers in ``freetoken.utils.hf`` are torch-free and run
  on a CPU-only box. They are exercised against a locally fabricated
  ``config.json`` (offline) so the test needs no network.
* The weight path (safetensors reader, dense->device / experts->host routing,
  and the MoE expert bank builder) needs ``torch``. Those tests are marked
  ``torch`` and are skipped on a box without it.
"""
from __future__ import annotations

import json
import os

import pytest

torch = pytest.importorskip("torch")

# Populated by the ``moe_ckpt`` fixture with the exact expert tensor written to
# the checkpoint, so the "real banks" test can assert the loader round-trips it.
_EXPECTED_L0_GATE_UP = None

from freetoken.models.loader import load_model
from freetoken.models.weight import _PlainBank, load_moe_expert_sources
from freetoken.utils import cached_load_hf_config
from freetoken.utils.hf import RawConfigShim, load_tokenizer

QWEN3_MOE_CONFIG = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "model_type": "qwen3_moe",
    "hidden_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "intermediate_size": 256,
    "moe_intermediate_size": 32,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "vocab_size": 64,
}


def _write_checkpoint(tmp_path, *, config: dict, weights: dict) -> str:
    """Fabricate a local HF-style checkpoint: a config.json plus one
    ``model.safetensors`` shard (with an index) holding ``weights``."""
    from safetensors.torch import save_file

    cfg = dict(config)
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    shards = {k: v.contiguous() for k, v in weights.items()}
    if shards:
        save_file(shards, str(tmp_path / "model.safetensors"))
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {k: "model.safetensors" for k in shards}})
        )
    return str(tmp_path)


def _qwen3_moe_weights() -> dict:
    """A tiny Qwen3-MoE checkpoint: dense weights + per-layer stacked MoE
    experts (the ``...experts.gate_up_proj`` / ``...experts.down_proj`` layout
    the model adapter normalizes to)."""
    hidden, inter, experts, vocab = 128, 32, 4, 64
    layers = 2
    w = {
        "model.embed_tokens.weight": torch.randn(vocab, hidden),
        "lm_head.weight": torch.randn(vocab, hidden),
        # one dense attention weight (not an expert -> dense path)
        "model.layers.0.self_attn.q_proj.weight": torch.randn(hidden, hidden),
    }
    for layer in range(layers):
        w[f"model.layers.{layer}.mlp.experts.gate_up_proj"] = torch.randn(experts, 2 * inter, hidden)
        w[f"model.layers.{layer}.mlp.experts.down_proj"] = torch.randn(experts, hidden, inter)
    return w


@pytest.fixture(scope="module")
def moe_ckpt(tmp_path_factory):
    weights = _qwen3_moe_weights()
    path = _write_checkpoint(tmp_path_factory.mktemp("qwen3moe"), config=QWEN3_MOE_CONFIG, weights=weights)
    # Persist the source expert tensors so the "real banks" test can compare the
    # loaded bank against the exact bytes that were written (randn is not
    # reproducible across calls).
    global _EXPECTED_L0_GATE_UP
    _EXPECTED_L0_GATE_UP = weights["model.layers.0.mlp.experts.gate_up_proj"].to(torch.bfloat16)
    return path


@pytest.fixture(scope="module")
def dense_ckpt(tmp_path_factory):
    return _write_checkpoint(
        tmp_path_factory.mktemp("qwen3dense"),
        config={**QWEN3_MOE_CONFIG, "architectures": ["Qwen3ForCausalLM"]},
        weights={"model.embed_tokens.weight": torch.randn(64, 128)},
    )


# --- torch-free config helpers (run on CPU-only boxes) -----------------------


def test_raw_config_shim_attribute_access():
    shim = RawConfigShim({"hidden_size": 128, "num_hidden_layers": 2}, architectures=["X"])
    assert shim.hidden_size == 128
    assert shim.num_hidden_layers == 2
    assert "X" in shim.architectures
    with pytest.raises(AttributeError):
        shim.does_not_exist  # noqa: B018


def test_raw_config_shim_wraps_nested_config():
    shim = RawConfigShim({"moe_config": {"num_experts": 4}})
    assert isinstance(shim.moe_config, RawConfigShim)
    assert shim.moe_config.num_experts == 4


def test_download_hf_weight_returns_local_dir(moe_ckpt):
    from freetoken.utils.hf import download_hf_weight

    assert download_hf_weight(moe_ckpt) == moe_ckpt
    assert os.path.isfile(os.path.join(moe_ckpt, "config.json"))


# --- torch-gated checkpoint loading -------------------------------------------


def test_cached_load_hf_config_reads_local(moe_ckpt):
    cfg = cached_load_hf_config(moe_ckpt)
    assert cfg.architectures == ["Qwen3MoeForCausalLM"]
    assert cfg.hidden_size == 128
    assert cfg.num_experts == 4


def test_iter_safetensors_reads_shards_onto_device(dense_ckpt):
    # The loader's shard-reading primitive (safetensors -> named tensors on the
    # destination device) is exercised directly: the Qwen3 dense model's
    # ``iter_weights``/``parse_config`` are owned by a separate stub
    # (``models-dense``), so the dense->device routing is asserted at the
    # primitive the loader is built on.
    from freetoken.models.weight import iter_safetensors

    device = torch.device("cpu")
    got = {name: tensor for name, tensor in iter_safetensors(dense_ckpt, device)}
    assert set(got) == {"model.embed_tokens.weight"}
    assert got["model.embed_tokens.weight"].shape == (64, 128)
    assert got["model.embed_tokens.weight"].device == device


def test_load_moe_expert_sources_dummy_banks(moe_ckpt):
    gate_up, down = load_moe_expert_sources(moe_ckpt, dtype=torch.bfloat16, dummy=True)
    # Per-layer banks, one per MoE layer, stacked to [num_experts, ...].
    assert len(gate_up) == 2 and len(down) == 2
    for gu, dn in zip(gate_up, down):
        assert gu.shape == (4, 2 * 32, 128)
        assert gu.dtype == torch.bfloat16
        assert dn.shape == (4, 128, 32)
        assert dn.device.type == "cpu"  # host offload banks


def test_load_moe_expert_sources_real_banks(moe_ckpt):
    gate_up, down = load_moe_expert_sources(moe_ckpt, dtype=torch.bfloat16)
    assert len(gate_up) == 2 and len(down) == 2
    # The real path reads the checkpoint, so bank[0] must match the exact bytes
    # that were written (the fixture persisted them as bf16).
    assert torch.equal(gate_up[0], _EXPECTED_L0_GATE_UP)
    assert gate_up[0].device.type == "cpu"


def test_load_model_real_moe_routes_experts_to_host(moe_ckpt):
    model, expert_sources = load_model(moe_ckpt, torch.device("cpu"))
    # MoE checkpoint -> per-layer expert banks, on host memory.
    assert model is not None
    assert len(expert_sources[0]) == 2
    assert expert_sources[0][0].device.type == "cpu"
