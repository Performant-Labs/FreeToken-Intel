"""End-to-end: the real DeepSeek-V4 MoE router + full model wiring on top
of MLA (#190) and DSA (#191) (issue `models-deepseek-v4-e2e`, #192, the
final child of the DeepSeek-V4 epic #21).

A small fabricated checkpoint exercising a leading dense layer
(first_k_dense_replace=1) + a real grouped-topk-router MoE layer (with a
shared expert) + DSA -- every feature this epic built, combined, through
the real Engine (prefill + decode, deterministic greedy). See
gpt_oss/__init__.py's own module docstring for the established "small
synthetic checkpoint proves the wiring" pattern this mirrors.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from freetoken.core import Req, SamplingParams, reset_global_ctx
from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.engine.engine import Engine
from freetoken.models.deepseek_v4 import _DeepseekV4MLP, _DeepseekV4MoE, iter_weights, parse_config
from freetoken.models.register import get_model_class

DEVICE = "cpu"

H, V, L = 32, 64, 3
NH = 4
Q_LORA, KV_LORA = 24, 16
QK_ROPE, QK_NOPE, V_HEAD = 4, 6, 8
INTER = 48
E, MOE_INTER, TOPK = 4, 24, 2
N_GROUP, TOPK_GROUP = 2, 1
N_SHARED = 1
FIRST_K_DENSE = 1  # layer 0 dense, layers 1-2 MoE
INDEX_TOPK, INDEX_HEAD_DIM, INDEX_N_HEADS = 3, 8, 2

TINY_CONFIG = {
    "architectures": ["DeepseekV4ForCausalLM"],
    "model_type": "deepseek_v4",
    "hidden_size": H,
    "vocab_size": V,
    "num_hidden_layers": L,
    "num_attention_heads": NH,
    "q_lora_rank": Q_LORA,
    "kv_lora_rank": KV_LORA,
    "qk_rope_head_dim": QK_ROPE,
    "qk_nope_head_dim": QK_NOPE,
    "v_head_dim": V_HEAD,
    "attention_bias": False,
    "intermediate_size": INTER,
    "n_routed_experts": E,
    "moe_intermediate_size": MOE_INTER,
    "num_experts_per_tok": TOPK,
    "n_group": N_GROUP,
    "topk_group": TOPK_GROUP,
    "routed_scaling_factor": 1.5,
    "n_shared_experts": N_SHARED,
    "norm_topk_prob": True,
    "first_k_dense_replace": FIRST_K_DENSE,
    "index_topk": INDEX_TOPK,
    "index_head_dim": INDEX_HEAD_DIM,
    "index_n_heads": INDEX_N_HEADS,
    "max_position_embeddings": 128,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
}


@pytest.fixture(autouse=True)
def _clean_global_ctx():
    yield
    reset_global_ctx()


def _config():
    return parse_config(type("Hf", (), {"to_dict": lambda self: TINY_CONFIG})())


def test_config_is_moe_with_router_fields():
    config = _config()
    assert config.is_moe is True
    assert config.num_experts == E
    assert config.attrs["n_group"] == N_GROUP
    assert config.first_k_dense_replace == FIRST_K_DENSE


def test_model_class_builds_dense_and_moe_layers():
    config = _config()
    model = get_model_class("DeepseekV4ForCausalLM", config, device=torch.device("cpu"))
    assert isinstance(model.layers[0].mlp, _DeepseekV4MLP)  # dense (first_k_dense_replace)
    assert isinstance(model.layers[1].mlp, _DeepseekV4MoE)
    assert len(model.layers[1].mlp.experts) == E
    assert model.layers[1].mlp.shared_experts is not None
    assert model.layers[1].mlp.gate.e_score_correction_bias.shape == (E,)
    # DSA is also active on every layer (independent of dense/MoE split).
    assert model.layers[0].self_attn.index_topk == INDEX_TOPK
    assert model.layers[1].self_attn.index_topk == INDEX_TOPK


def _write_tiny_checkpoint(tmp_path) -> str:
    from safetensors.torch import save_file

    model_path = tmp_path / "ckpt"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(TINY_CONFIG))

    config = _config()
    state = {}
    gen = torch.Generator().manual_seed(0)

    def add(name, shape):
        state[name] = torch.randn(shape, generator=gen, dtype=torch.float32) * 0.02

    qk_head_dim = QK_NOPE + QK_ROPE
    add("model.embed_tokens.weight", (V, H))
    for l in range(L):
        p = f"model.layers.{l}"
        add(f"{p}.input_layernorm.weight", (H,))
        add(f"{p}.self_attn.q_a_proj.weight", (Q_LORA, H))
        add(f"{p}.self_attn.q_a_layernorm.weight", (Q_LORA,))
        add(f"{p}.self_attn.q_b_proj.weight", (NH * qk_head_dim, Q_LORA))
        add(f"{p}.self_attn.kv_a_proj_with_mqa.weight", (KV_LORA + QK_ROPE, H))
        add(f"{p}.self_attn.kv_a_layernorm.weight", (KV_LORA,))
        add(f"{p}.self_attn.kv_b_proj.weight", (NH * (QK_NOPE + V_HEAD), KV_LORA))
        add(f"{p}.self_attn.o_proj.weight", (H, NH * V_HEAD))
        add(f"{p}.self_attn.wq_b.weight", (NH * INDEX_HEAD_DIM, Q_LORA))
        add(f"{p}.self_attn.wk.weight", (INDEX_HEAD_DIM, H))
        add(f"{p}.self_attn.indexer_k_norm.weight", (INDEX_HEAD_DIM,))
        add(f"{p}.self_attn.indexer_k_norm.bias", (INDEX_HEAD_DIM,))
        add(f"{p}.self_attn.weights_proj.weight", (NH, H))
        add(f"{p}.post_attention_layernorm.weight", (H,))
        if l < FIRST_K_DENSE:
            add(f"{p}.mlp.gate_proj.weight", (INTER, H))
            add(f"{p}.mlp.up_proj.weight", (INTER, H))
            add(f"{p}.mlp.down_proj.weight", (H, INTER))
        else:
            add(f"{p}.mlp.gate.weight", (E, H))
            state[f"{p}.mlp.gate.e_score_correction_bias"] = torch.zeros(E, dtype=torch.float32)
            for e in range(E):
                eb = f"{p}.mlp.experts.{e}"
                add(f"{eb}.gate_proj.weight", (MOE_INTER, H))
                add(f"{eb}.up_proj.weight", (MOE_INTER, H))
                add(f"{eb}.down_proj.weight", (H, MOE_INTER))
            add(f"{p}.mlp.shared_experts.gate_proj.weight", (MOE_INTER * N_SHARED, H))
            add(f"{p}.mlp.shared_experts.up_proj.weight", (MOE_INTER * N_SHARED, H))
            add(f"{p}.mlp.shared_experts.down_proj.weight", (H, MOE_INTER * N_SHARED))
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


def test_iter_weights_covers_dense_moe_and_shared_expert(tmp_path):
    model_path = _write_tiny_checkpoint(tmp_path)
    names = {n for n, _ in iter_weights(model_path, torch.device("cpu"))}
    assert "model.layers.0.mlp.gate_proj.weight" in names  # dense layer
    assert "model.layers.1.mlp.gate.weight" in names  # MoE router
    assert "model.layers.1.mlp.experts.0.gate_proj.weight" in names
    assert "model.layers.1.mlp.shared_experts.gate_proj.weight" in names


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
