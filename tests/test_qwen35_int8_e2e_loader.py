"""End-to-end: load_model() wires a compressed-tensors pack-quantized
INT8 checkpoint's experts into a real "int8_channel" OffloadMoeCache and
runs a forward step (issue `moe-quant-banks-native-multi`, #163's own
noted coverage gap: int8_channel had only SlotWeightAccessor-level XPU
coverage, not a full checkpoint-loader test the way GPTQ's #138 has).

Mirrors test_qwen35_gptq_e2e_loader.py's / test_qwen35_fp8_e2e_loader.py's
/ test_qwen35_mxfp4_e2e_loader.py's own approach -- a small (few-KB)
fabricated INT8 checkpoint, not a real one. Real compressed-tensors INT8
checkpoints (rj1013/gemma-4-26B-A4B-it_q8) use a different model
architecture (gemma-4, not yet ported -- see issue #20) than this
fixture's qwen3_5_moe shell; see test_qwen35_fp8_e2e_loader.py's own
docstring for why that's fine.

Real int8_channel checkpoint keys carry NO ``mlp.`` segment (unlike
GPTQ/FP8's ``...mlp.experts...``): ``...layers.{L}.experts.{e}.{proj}
.{weight_packed|weight_scale|weight_shape}`` -- see
``_parse_int8_expert_key``'s own docstring in weight.py.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from freetoken.models.loader import load_model

from tests.test_qwen35_gptq_e2e_loader import _drive_prefill

H, I, E, V, L = 64, 32, 4, 64, 1
GROUP_SIZE = 32


def _pack_int8(codes: torch.Tensor) -> torch.Tensor:
    """``[N, K]`` int8 codes -> ``[N, K // 4]`` int32, compressed-tensors'
    dense num_bits=8 packing (byte i of each word = element word*4+i,
    stored as ``code + 128``) -- see int8_packed_linear.py's own module
    docstring for the format."""
    n, k = codes.shape
    codes64 = (codes.to(torch.int64) + 128).reshape(n, k // 4, 4)
    word = sum(codes64[:, :, i] << (8 * i) for i in range(4))
    word = torch.where(word >= (1 << 31), word - (1 << 32), word)
    return word.to(torch.int32)


def _int8_projection(n: int, k: int, *, seed: int):
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(-127, 128, (n, k), generator=g, dtype=torch.int8)
    num_groups = k // GROUP_SIZE
    scale = (torch.rand(n, num_groups, generator=g) * 0.1 + 0.01).to(torch.float32)
    weight_shape = torch.tensor([n, k], dtype=torch.int64)
    return _pack_int8(codes), scale, weight_shape


def _int8_expert_weights() -> dict:
    """One full-attention layer's dense weights (unquantized) plus
    compressed-tensors pack-quantized-INT8 routed experts (per-expert raw
    layout, the real checkpoint's own shape) and a plain (unquantized)
    shared expert."""
    w = {
        "model.language_model.embed_tokens.weight": torch.randn(V, H),
        "model.language_model.norm.weight": torch.randn(H),
        "lm_head.weight": torch.randn(V, H),
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(4 * 16 * 2, H),
        "model.language_model.layers.0.self_attn.k_proj.weight": torch.randn(2 * 16, H),
        "model.language_model.layers.0.self_attn.v_proj.weight": torch.randn(2 * 16, H),
        "model.language_model.layers.0.self_attn.o_proj.weight": torch.randn(H, 4 * 16),
        "model.language_model.layers.0.self_attn.q_norm.weight": torch.randn(16),
        "model.language_model.layers.0.self_attn.k_norm.weight": torch.randn(16),
        "model.language_model.layers.0.mlp.gate.weight": torch.randn(E, H),
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": torch.randn(I, H),
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight": torch.randn(I, H),
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight": torch.randn(H, I),
        "model.language_model.layers.0.mlp.shared_expert_gate.weight": torch.randn(1, H),
    }
    for e in range(E):
        base = f"model.language_model.layers.0.experts.{e}"  # no "mlp." -- real checkpoint's own spelling
        for proj, n, k in (("gate_proj", I, H), ("up_proj", I, H), ("down_proj", H, I)):
            weight_packed, weight_scale, weight_shape = _int8_projection(n, k, seed=100 * (e + 1) + hash(proj) % 7)
            w[f"{base}.{proj}.weight_packed"] = weight_packed
            w[f"{base}.{proj}.weight_scale"] = weight_scale
            w[f"{base}.{proj}.weight_shape"] = weight_shape
    return w


@pytest.fixture(scope="module")
def qwen35_int8_ckpt(tmp_path_factory):
    from safetensors.torch import save_file

    path = tmp_path_factory.mktemp("qwen35_int8")
    text_config = {
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
        "head_dim": 16,
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
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {
                    "format": "pack-quantized",
                    "weights": {
                        "num_bits": 8,
                        "type": "int",
                        "strategy": "group",
                        "group_size": GROUP_SIZE,
                        "symmetric": True,
                        "actorder": "static",
                    },
                }
            },
        },
    }
    weights = _int8_expert_weights()
    (path / "config.json").write_text(json.dumps(config))
    save_file({k: v.contiguous() for k, v in weights.items()}, str(path / "model.safetensors"))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in weights}})
    )
    return str(path)


def test_load_model_builds_an_int8_channel_offload_cache(qwen35_int8_ckpt):
    model, _ = load_model(qwen35_int8_ckpt, torch.device("cpu"), dtype=torch.float32, moe_backend="offload")

    cache = model.moe_cache
    assert cache.quant_format == "int8_channel"
    assert cache.int8_k_gate_up == H
    assert cache.int8_k_down == I
    assert set(cache.bank_sources) == {
        "weight_packed_gate_up", "weight_scale_gate_up", "weight_packed_down", "weight_scale_down",
    }


def test_int8_checkpoint_forward_step_produces_finite_logits(qwen35_int8_ckpt):
    """The real point of this test (mirrors #138's own for GPTQ): a
    compressed-tensors INT8-quantized checkpoint's forward pass runs
    end-to-end -- SlotWeightAccessor dequantizes the resident packed
    experts on the fly, reading from the cache
    load_moe_expert_sources/_attach_offload_cache actually built."""
    model, _ = load_model(qwen35_int8_ckpt, torch.device("cpu"), dtype=torch.float32, moe_backend="offload")
    seq = torch.randint(0, V, (5,))
    T = seq.shape[0]
    logits, _ = _drive_prefill(model, seq, T)
    assert logits.shape == (1, V)
    assert torch.isfinite(logits).all()
