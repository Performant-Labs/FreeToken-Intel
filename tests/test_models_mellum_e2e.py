"""End-to-end: Mellum2-12B-A2.5B full model wiring (issue `models-mellum-e2e`,
#228, final child of the Mellum epic #226).

Small synthetic checkpoint only -- real-checkpoint validation against
`JetBrains/Mellum2-12B-A2.5B-Base` is sequenced separately by the parent
session (deliberately out of scope here, see the issue's own body). Mirrors
the established per-model e2e pattern (`tests/test_models_glm_moe_dsa_e2e.py`,
`tests/test_models_deepseek_v2_lite_mla_e2e.py`): seeded weights, no empty
`model.safetensors.index.json`, a real `Engine` prefill+decode run.

`layer_types` alternates 3 sliding_attention + 1 full_attention (matching the
real checkpoint's own period-4 pattern, per `mellum/__init__.py`'s own
docstring) -- 4 layers here exercises both attention variants and both
per-type RoPE tables (plain for sliding, YaRN for full) in one small model.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from freetoken.core import Req, SamplingParams, reset_global_ctx
from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.engine.engine import Engine
from freetoken.models.register import get_model_class

DEVICE = "cpu"

H, V, L = 32, 64, 4
NH, KVH, HEAD_DIM = 4, 2, 8
INTER, MOE_INTER = 48, 24
E, TOPK = 4, 2
LAYER_TYPES = ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]

TINY_CONFIG = {
    "architectures": ["MellumForCausalLM"],
    "model_type": "mellum",
    "hidden_size": H,
    "vocab_size": V,
    "num_hidden_layers": L,
    "num_attention_heads": NH,
    "num_key_value_heads": KVH,
    "head_dim": HEAD_DIM,
    "intermediate_size": INTER,
    "moe_intermediate_size": MOE_INTER,
    "num_experts": E,
    "num_experts_per_tok": TOPK,
    "norm_topk_prob": True,
    "max_position_embeddings": 128,
    "sliding_window": 8,
    "layer_types": LAYER_TYPES,
    "mlp_layer_types": ["sparse"] * L,
    "rope_parameters": {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": {
            "rope_type": "yarn",
            "rope_theta": 500000.0,
            "factor": 4.0,
            "original_max_position_embeddings": 32,
            "beta_fast": 32,
            "beta_slow": 1,
            "attention_factor": 1.0,
        },
    },
    "rms_norm_eps": 1e-5,
    "tie_word_embeddings": False,
}


@pytest.fixture(autouse=True)
def _clean_global_ctx():
    yield
    reset_global_ctx()


def _config():
    from freetoken.models.mellum import parse_config

    return parse_config(type("Hf", (), {"to_dict": lambda self: TINY_CONFIG})())


def _write_tiny_checkpoint(tmp_path) -> str:
    from safetensors.torch import save_file

    model_path = tmp_path / "ckpt"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(TINY_CONFIG))

    state = {}
    gen = torch.Generator().manual_seed(0)

    def add(name, shape):
        state[name] = torch.randn(shape, generator=gen, dtype=torch.float32) * 0.02

    add("model.embed_tokens.weight", (V, H))
    for l in range(L):
        p = f"model.layers.{l}"
        add(f"{p}.input_layernorm.weight", (H,))
        add(f"{p}.self_attn.q_proj.weight", (NH * HEAD_DIM, H))
        add(f"{p}.self_attn.k_proj.weight", (KVH * HEAD_DIM, H))
        add(f"{p}.self_attn.v_proj.weight", (KVH * HEAD_DIM, H))
        add(f"{p}.self_attn.o_proj.weight", (H, NH * HEAD_DIM))
        add(f"{p}.self_attn.q_norm.weight", (HEAD_DIM,))
        add(f"{p}.self_attn.k_norm.weight", (HEAD_DIM,))
        add(f"{p}.post_attention_layernorm.weight", (H,))
        add(f"{p}.mlp.gate.weight", (E, H))
        for e in range(E):
            eb = f"{p}.mlp.experts.{e}"
            add(f"{eb}.gate_proj.weight", (MOE_INTER, H))
            add(f"{eb}.up_proj.weight", (MOE_INTER, H))
            add(f"{eb}.down_proj.weight", (H, MOE_INTER))
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


def test_model_class_builds_alternating_sliding_and_full_layers():
    config = _config()
    model = get_model_class("MellumForCausalLM", config, device=torch.device("cpu"))
    for i, layer in enumerate(model.layers):
        assert layer.self_attn.layer_type == LAYER_TYPES[i]
        assert layer.self_attn.sliding_window == (8 if LAYER_TYPES[i] == "sliding_attention" else 0)


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
