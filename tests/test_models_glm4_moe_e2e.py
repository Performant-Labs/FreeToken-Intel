"""End-to-end: a fabricated GLM-4.7 (glm4_moe) checkpoint loads and runs a
real forward pass through the Engine (issue `models-glm`, #22 -- the
glm4_moe half; see issue #187's compat matrix for what's covered).

A small (few-KB) fabricated checkpoint, not a real one -- same established
pattern as every other model port in this project. Exercises the two
features new to this port: a leading dense layer (first_k_dense_replace=1)
and the grouped/sigmoid/bias-corrected top-k router + unweighted-sum shared
expert (n_group=topk_group=1, the same degenerate-to-plain-top-k case
GLM-4.7's own real config uses).
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from freetoken.core import Req, SamplingParams, reset_global_ctx
from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.engine.engine import Engine
from freetoken.models.glm4_moe import iter_weights, parse_config
from freetoken.models.register import get_model_class

DEVICE = "cpu"

H, V, L = 32, 64, 3
NH, NKV, HD = 4, 2, 8
PARTIAL_ROTARY = 0.5
INTER = 48
E, MOE_INTER, TOPK = 4, 24, 2
FIRST_K_DENSE = 1  # layer 0 is dense, layers 1-2 are MoE

TINY_CONFIG = {
    "architectures": ["Glm4MoeForCausalLM"],
    "model_type": "glm4_moe",
    "hidden_size": H,
    "vocab_size": V,
    "num_hidden_layers": L,
    "num_attention_heads": NH,
    "num_key_value_heads": NKV,
    "head_dim": HD,
    "partial_rotary_factor": PARTIAL_ROTARY,
    "use_qk_norm": True,
    "attention_bias": True,
    "intermediate_size": INTER,
    "moe_intermediate_size": MOE_INTER,
    "n_routed_experts": E,
    "n_shared_experts": 1,
    "num_experts_per_tok": TOPK,
    "n_group": 1,
    "topk_group": 1,
    "routed_scaling_factor": 1.0,
    "norm_topk_prob": True,
    "first_k_dense_replace": FIRST_K_DENSE,
    "max_position_embeddings": 128,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-5,
    "tie_word_embeddings": False,
}


@pytest.fixture(autouse=True)
def _clean_global_ctx():
    yield
    reset_global_ctx()


def _write_tiny_checkpoint(tmp_path) -> str:
    from safetensors.torch import save_file

    model_path = tmp_path / "ckpt"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(TINY_CONFIG))

    config = parse_config(type("Hf", (), {"to_dict": lambda self: TINY_CONFIG})())
    state = {}

    def add(name, shape):
        state[name] = torch.randn(shape, dtype=torch.float32) * 0.02

    heads, kv, head_dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    inter, moe_inter = config.intermediate_size, config.moe_intermediate_size
    n_experts = config.num_experts
    n_shared = config.attrs["n_shared_experts"]

    add("model.embed_tokens.weight", (V, H))
    for l in range(L):
        p = f"model.layers.{l}"
        add(f"{p}.input_layernorm.weight", (H,))
        add(f"{p}.self_attn.q_proj.weight", (heads * head_dim, H))
        add(f"{p}.self_attn.q_proj.bias", (heads * head_dim,))
        add(f"{p}.self_attn.k_proj.weight", (kv * head_dim, H))
        add(f"{p}.self_attn.k_proj.bias", (kv * head_dim,))
        add(f"{p}.self_attn.v_proj.weight", (kv * head_dim, H))
        add(f"{p}.self_attn.v_proj.bias", (kv * head_dim,))
        add(f"{p}.self_attn.o_proj.weight", (H, heads * head_dim))
        add(f"{p}.self_attn.q_norm.weight", (head_dim,))
        add(f"{p}.self_attn.k_norm.weight", (head_dim,))
        add(f"{p}.post_attention_layernorm.weight", (H,))
        if l < FIRST_K_DENSE:
            add(f"{p}.mlp.gate_proj.weight", (inter, H))
            add(f"{p}.mlp.up_proj.weight", (inter, H))
            add(f"{p}.mlp.down_proj.weight", (H, inter))
        else:
            add(f"{p}.mlp.gate.weight", (n_experts, H))
            state[f"{p}.mlp.gate.e_score_correction_bias"] = torch.zeros(n_experts, dtype=torch.float32)
            for e in range(n_experts):
                eb = f"{p}.mlp.experts.{e}"
                add(f"{eb}.gate_proj.weight", (moe_inter, H))
                add(f"{eb}.up_proj.weight", (moe_inter, H))
                add(f"{eb}.down_proj.weight", (H, moe_inter))
            add(f"{p}.mlp.shared_experts.gate_proj.weight", (moe_inter * n_shared, H))
            add(f"{p}.mlp.shared_experts.up_proj.weight", (moe_inter * n_shared, H))
            add(f"{p}.mlp.shared_experts.down_proj.weight", (H, moe_inter * n_shared))
    add("model.norm.weight", (H,))
    add("lm_head.weight", (V, H))

    save_file(state, str(model_path / "model.safetensors"))
    return str(model_path)


def _engine_config(model_path: str, *, device: str | None = None) -> EngineConfig:
    return EngineConfig(
        model_path=model_path,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.float32,
        device=device,
        attention_backend="auto",
        max_running_req=2,
        page_size=1,
        max_seq_len_override=32,
    )


def _add_prompt(engine: Engine, output_len: int) -> None:
    engine.add_request(
        Req(
            input_ids=[1, 2, 3],
            table_idx=0,
            cached_len=0,
            output_len=output_len,
            uid=0,
            sampling_params=SamplingParams(temperature=0.0, max_tokens=output_len),
            cache_handle=None,
        )
    )


def test_iter_weights_covers_dense_and_moe_layers(tmp_path):
    model_path = _write_tiny_checkpoint(tmp_path)
    names = {n for n, _ in iter_weights(model_path, torch.device("cpu"))}
    assert "model.layers.0.mlp.gate_proj.weight" in names  # dense layer
    assert "model.layers.1.mlp.gate.weight" in names  # MoE router
    assert "model.layers.1.mlp.gate.e_score_correction_bias" in names
    assert "model.layers.1.mlp.experts.0.gate_proj.weight" in names
    assert "model.layers.1.mlp.shared_experts.gate_proj.weight" in names


def test_model_class_builds_dense_and_moe_layers(tmp_path):
    model_path = _write_tiny_checkpoint(tmp_path)
    config = parse_config(type("Hf", (), {"to_dict": lambda self: TINY_CONFIG})())
    model = get_model_class("Glm4MoeForCausalLM", config, device=torch.device("cpu"))
    from freetoken.models.glm4_moe import _Glm4MLP, _Glm4MoE

    assert isinstance(model.layers[0].mlp, _Glm4MLP)
    assert isinstance(model.layers[1].mlp, _Glm4MoE)
    assert len(model.layers[1].mlp.experts) == E
    assert model.layers[1].mlp.gate.e_score_correction_bias.shape == (E,)


def test_engine_generate_prefill_and_decode(tmp_path):
    model_path = _write_tiny_checkpoint(tmp_path)
    engine = Engine(_engine_config(model_path, device=DEVICE))
    _add_prompt(engine, output_len=4)
    generated = engine.generate()
    vocab = engine.config.model_config.vocab_size
    assert len(generated) == 1
    assert len(generated[0]) == 4
    assert all(0 <= t < vocab for t in generated[0])


def test_engine_greedy_is_deterministic(tmp_path):
    model_path = _write_tiny_checkpoint(tmp_path)

    def build():
        engine = Engine(_engine_config(model_path, device=DEVICE))
        _add_prompt(engine, output_len=3)
        return engine.generate()

    a = build()
    b = build()
    assert a == b
    assert len(a[0]) == 3
