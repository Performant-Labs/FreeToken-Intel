"""End-to-end: a fabricated gpt-oss checkpoint loads and runs a real
forward pass through the Engine (issue `models-gpt-oss`, #23; see issue
#187's compat matrix for what's covered).

A small (few-KB) fabricated checkpoint, not a real one -- same established
pattern as every other model port in this project. Exercises every feature
new to this port: attention sinks, alternating full/sliding-window
attention (real per-layer ``layer_types``), YaRN RoPE, and the clamped-GLU
expert activation (see gpt_oss/__init__.py's own module docstring for the
real, grounded math each of these implements, and its documented
simplification: unbiased split gate_proj/up_proj experts instead of the
real checkpoint's fused+biased ``gate_up_proj``).
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from freetoken.core import Req, SamplingParams, reset_global_ctx
from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.engine.engine import Engine
from freetoken.models.gpt_oss import iter_weights, parse_config
from freetoken.models.register import get_model_class

DEVICE = "cpu"

H, V, L = 32, 64, 4
NH, NKV, HD = 4, 2, 8
E, MOE_INTER, TOPK = 4, 24, 2
SLIDING_WINDOW = 4
SWIGLU_LIMIT = 7.0
LAYER_TYPES = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]

TINY_CONFIG = {
    "architectures": ["GptOssForCausalLM"],
    "model_type": "gpt_oss",
    "hidden_size": H,
    "vocab_size": V,
    "num_hidden_layers": L,
    "num_attention_heads": NH,
    "num_key_value_heads": NKV,
    "head_dim": HD,
    "attention_bias": True,
    "intermediate_size": MOE_INTER,
    "num_local_experts": E,
    "num_experts_per_tok": TOPK,
    "sliding_window": SLIDING_WINDOW,
    "layer_types": LAYER_TYPES,
    "swiglu_limit": SWIGLU_LIMIT,
    "max_position_embeddings": 128,
    "rope_theta": 150000.0,
    "rope_scaling": {
        "rope_type": "yarn",
        "factor": 32.0,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "original_max_position_embeddings": 4096,
        "truncate": False,
    },
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
    moe_inter, n_experts = config.moe_intermediate_size, config.num_experts

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
        add(f"{p}.self_attn.o_proj.bias", (H,))
        add(f"{p}.self_attn.sinks", (heads,))
        add(f"{p}.post_attention_layernorm.weight", (H,))
        add(f"{p}.mlp.gate.weight", (n_experts, H))
        add(f"{p}.mlp.gate.bias", (n_experts,))
        for e in range(n_experts):
            eb = f"{p}.mlp.experts.{e}"
            add(f"{eb}.gate_proj.weight", (moe_inter, H))
            add(f"{eb}.up_proj.weight", (moe_inter, H))
            add(f"{eb}.down_proj.weight", (H, moe_inter))
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
            input_ids=[1, 2, 3, 4, 5],
            table_idx=0,
            cached_len=0,
            output_len=output_len,
            uid=0,
            sampling_params=SamplingParams(temperature=0.0, max_tokens=output_len),
            cache_handle=None,
        )
    )


def test_iter_weights_covers_sinks_and_experts(tmp_path):
    model_path = _write_tiny_checkpoint(tmp_path)
    names = {n for n, _ in iter_weights(model_path, torch.device("cpu"))}
    assert "model.layers.0.self_attn.sinks" in names
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" in names
    assert "model.layers.1.mlp.gate.weight" in names


def test_model_class_builds_alternating_layer_types(tmp_path):
    model_path = _write_tiny_checkpoint(tmp_path)
    config = parse_config(type("Hf", (), {"to_dict": lambda self: TINY_CONFIG})())
    model = get_model_class("GptOssForCausalLM", config, device=torch.device("cpu"))
    assert model.layers[0].self_attn.sliding_window == SLIDING_WINDOW
    assert model.layers[1].self_attn.sliding_window == 0
    assert model.layers[0].self_attn.sinks.shape == (NH,)
    assert len(model.layers[0].mlp.experts) == E


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
