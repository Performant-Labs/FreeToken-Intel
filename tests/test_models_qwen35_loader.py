"""Tests for the Qwen3.5/3.6 (``qwen3_5_moe``) weight path -- torch-gated.

``iter_weights`` (and the ``load_model`` path built on it) need torch to move
tensors to their destination devices, so this module is torch-gated: it
``importorskip("torch")``s at the top and is deselected on a torch-free box (the
CPU ``.venv``). It runs under the XPU venv / nightly.

These tests drive the real loader contract against a fabricated *multimodal*
checkpoint whose language tower sits under ``model.language_model.*`` (the Qwen3.6
layout) and ships a ``model.visual.*`` vision tower:

* ``iter_weights`` drops the vision tower and remaps ``model.language_model.*``
  to ``model.*`` so the loader's MoE-bank plumbing resolves the keys.
* routed experts go to host; everything else (including the always-on shared
  expert and the linear-attention weights) goes to the dense device.
* ``load_moe_expert_sources`` builds the per-layer banks from the remapped keys.
* ``load_model`` runs end-to-end: it places the dense weights, builds the banks,
  and returns the (still-stub) model -- whose ``forward`` fails loud until the
  real hybrid forward lands in a later issue.
"""
from __future__ import annotations

import json
import os

import pytest

torch = pytest.importorskip("torch")

from freetoken.models.loader import load_model
from freetoken.models.qwen3_5_moe import iter_weights
from freetoken.models.weight import load_moe_expert_sources

H, I, E, V, L = 32, 16, 4, 64, 2

# A small hybrid text tower: 2 layers (1 linear-attention + 1 full-attention),
# a 4-way router top-2 + an always-on shared expert, and a vision tower to prove
# it is dropped. The text tower sits under model.language_model.* (the real
# Qwen3.6 layout).
def _qwen35_text_config() -> dict:
    return {
        "hidden_size": H,
        "num_hidden_layers": L,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_experts": E,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": I,
        "shared_expert_intermediate_size": I,
        "vocab_size": V,
        "max_position_embeddings": 128,
        "head_dim": 8,
        "attn_output_gate": True,
        "partial_rotary_factor": 0.5,
        "full_attention_interval": 2,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4,
        "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.5},
    }


def _qwen35_weights() -> dict:
    """The full hybrid text-tower weight set (every layer), plus a vision tower.

    Layer 0 is linear-attention (linear_attn.* + conv1d), layer 1 is full-
    attention (self_attn.*); both carry a MoE mlp (experts.* + shared_expert.*).
    """
    w = {
        "model.visual.blocks.0.attn.q.weight": torch.randn(8, 8),
        "model.language_model.embed_tokens.weight": torch.randn(V, H),
        "model.language_model.norm.weight": torch.randn(H),
        "lm_head.weight": torch.randn(V, H),
    }
    for layer in range(L):
        if layer % 2 == 0:  # linear-attention (Gated DeltaNet) layer
            w[f"model.language_model.layers.{layer}.linear_attn.in_proj_qkv.weight"] = torch.randn(2 * H, H)
            w[f"model.language_model.layers.{layer}.linear_attn.in_proj_z.weight"] = torch.randn(H, H)
            w[f"model.language_model.layers.{layer}.linear_attn.conv1d.weight"] = torch.randn(4, H)
            w[f"model.language_model.layers.{layer}.linear_attn.out_proj.weight"] = torch.randn(H, H)
        else:  # full-GQA layer
            w[f"model.language_model.layers.{layer}.self_attn.q_proj.weight"] = torch.randn(H, H)
            w[f"model.language_model.layers.{layer}.self_attn.o_proj.weight"] = torch.randn(H, H)
        # Every layer has a MoE block: 4 routed experts (packed) + a shared expert.
        w[f"model.language_model.layers.{layer}.mlp.gate.weight"] = torch.randn(E, H)
        w[f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"] = torch.randn(E, 2 * I, H)
        w[f"model.language_model.layers.{layer}.mlp.experts.down_proj"] = torch.randn(E, H, I)
        w[f"model.language_model.layers.{layer}.mlp.shared_expert.gate_proj.weight"] = torch.randn(I, H)
        w[f"model.language_model.layers.{layer}.mlp.shared_expert.up_proj.weight"] = torch.randn(I, H)
        w[f"model.language_model.layers.{layer}.mlp.shared_expert.down_proj.weight"] = torch.randn(H, I)
        w[f"model.language_model.layers.{layer}.mlp.shared_expert_gate.weight"] = torch.randn(1, H)
    return w


@pytest.fixture(scope="module")
def qwen35_ckpt(tmp_path_factory):
    """A fabricated multimodal Qwen3.5/3.6 checkpoint (config.json + one
    safetensors shard + index)."""
    from safetensors.torch import save_file

    path = tmp_path_factory.mktemp("qwen35")
    config = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "tie_word_embeddings": True,
        "vision_config": {"hidden_size": 8, "num_chunks": 2},
        "text_config": _qwen35_text_config(),
    }
    weights = _qwen35_weights()
    (path / "config.json").write_text(json.dumps(config))
    save_file({k: v.contiguous() for k, v in weights.items()}, str(path / "model.safetensors"))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in weights}})
    )
    return str(path)


# --- iter_weights: vision drop + language_model remap + device routing --------


def test_iter_weights_drops_visual_and_remaps_language_model(qwen35_ckpt):
    names = [n for n, _ in iter_weights(qwen35_ckpt, torch.device("cpu"))]
    # No vision tower leaks through.
    assert not any(n.startswith("model.visual") for n in names)
    # The language prefix is remapped to the FreeToken name space.
    assert "model.embed_tokens.weight" in names
    assert "model.layers.0.linear_attn.in_proj_qkv.weight" in names
    assert "model.layers.1.self_attn.q_proj.weight" in names
    assert "model.layers.0.mlp.experts.gate_up_proj" in names
    assert "model.layers.0.mlp.shared_expert.gate_proj.weight" in names
    assert "lm_head.weight" in names
    # The raw checkpoint keys are never yielded un-remapped.
    assert not any(n.startswith("model.language_model") for n in names)


def test_iter_weights_dense_goes_to_device_experts_go_to_host(qwen35_ckpt):
    got = {n: t for n, t in iter_weights(qwen35_ckpt, torch.device("cpu"))}
    # Dense weights (incl. the shared expert + linear-attention) -> the device.
    assert got["model.layers.0.mlp.shared_expert.gate_proj.weight"].device.type == "cpu"
    assert got["model.layers.0.linear_attn.in_proj_qkv.weight"].device.type == "cpu"
    # Routed experts -> host offload banks (cpu here).
    assert got["model.layers.0.mlp.experts.gate_up_proj"].device.type == "cpu"


def test_iter_weights_include_flags_route_experts_only(qwen35_ckpt):
    # include_moe_experts=False: the routed experts are dropped, the shared expert
    # (dense) is kept. This is the loader's dense path.
    no_experts = [n for n, _ in iter_weights(qwen35_ckpt, torch.device("cpu"), include_moe_experts=False)]
    assert not any(".experts." in n for n in no_experts)
    assert "model.layers.0.mlp.shared_expert.gate_proj.weight" in no_experts
    assert "model.embed_tokens.weight" in no_experts
    # include_non_moe=False: only the routed experts remain (the loader's bank path).
    experts_only = [n for n, _ in iter_weights(qwen35_ckpt, torch.device("cpu"), include_non_moe=False)]
    assert all(".experts." in n for n in experts_only)
    assert len(experts_only) == 2 * L  # gate_up_proj + down_proj per layer


# --- the MoE bank path (remapped keys must satisfy the loader) -----------------


def test_load_moe_expert_sources_builds_banks_from_remapped_keys(qwen35_ckpt):
    gate_up, down = load_moe_expert_sources(qwen35_ckpt, dtype=torch.bfloat16)
    assert len(gate_up) == L and len(down) == L
    assert gate_up[0].shape == (E, 2 * I, H)
    assert down[0].shape == (E, H, I)
    assert gate_up[0].device.type == "cpu"
    assert down[0].device.type == "cpu"


# --- the top-level loader path -------------------------------------------------


def test_load_model_end_to_end_on_multimodal_ckpt(qwen35_ckpt):
    model, expert_sources = load_model(qwen35_ckpt, torch.device("cpu"))
    # The loader resolves the multimodal config, builds the model (which the
    # stub's __init__ rebinds to a plain nn.Module instance), and attaches the
    # per-layer host banks for the routed experts. (isinstance(model,
    # Qwen3_5MoEForCausalLM) is False by design -- the instance's class is
    # nn.Module so the loader's named_parameters() resolves; that is what the
    # loader actually consumes.)
    assert isinstance(model, torch.nn.Module)
    assert model.config.architectures == ["Qwen3_5MoeForConditionalGeneration"]
    assert len(expert_sources[0]) == L
    assert expert_sources[0][0].shape == (E, 2 * I, H)
    assert expert_sources[0][0].device.type == "cpu"


def test_load_model_stub_forward_fails_loud(qwen35_ckpt):
    model, _ = load_model(qwen35_ckpt, torch.device("cpu"))
    # Phase 1: the hybrid forward is not implemented yet -- the loaded model must
    # fail loud (NotImplementedError) rather than silently run a wrong forward.
    with pytest.raises(NotImplementedError):
        model.forward()
