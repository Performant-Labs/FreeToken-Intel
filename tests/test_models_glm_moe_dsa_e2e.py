"""End-to-end: GLM-5.2 (glm_moe_dsa) -- MLA + DSA with cross-layer top-k
sharing + the real grouped-topk MoE router, the final piece of issue #22
(models-glm). See glm_moe_dsa/__init__.py's own module docstring for the
two real, confirmed differences from deepseek_v4 (#190-#192) this
resolves: interleaved RoPE (not half-split), and indexer_types-driven
cross-layer top-k sharing (the real answer to #191's own flagged open
question).

A small fabricated checkpoint with 4 layers and indexer_types =
["full", "shared", "full", "shared"] -- explicitly exercising BOTH real
indexer modes (not just "full" every layer, which wouldn't prove the
sharing mechanism does anything).
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from freetoken.core import Batch, Context, Req, SamplingParams, reset_global_ctx, set_global_ctx
from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.engine.engine import Engine
from freetoken.kvcache.base import BaseKVCachePool
from freetoken.models.glm_moe_dsa import _GlmMoeDsaMLA, _GlmMoeDsaMLP, _GlmMoeDsaMoE, iter_weights, parse_config
from freetoken.models.register import get_model_class

DEVICE = "cpu"

H, V, L = 32, 64, 4
NH = 4
Q_LORA, KV_LORA = 24, 16
QK_ROPE, QK_NOPE, V_HEAD = 4, 6, 8
INTER = 48
E, MOE_INTER, TOPK = 4, 24, 2
N_GROUP, TOPK_GROUP = 2, 1
N_SHARED = 1
FIRST_K_DENSE = 1
INDEX_TOPK, INDEX_HEAD_DIM, INDEX_N_HEADS = 3, 8, 2
INDEXER_TYPES = ["full", "shared", "full", "shared"]

TINY_CONFIG = {
    "architectures": ["GlmMoeDsaForCausalLM"],
    "model_type": "glm_moe_dsa",
    "hidden_size": H,
    "vocab_size": V,
    "num_hidden_layers": L,
    "num_attention_heads": NH,
    "num_key_value_heads": NH,
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
    "indexer_types": INDEXER_TYPES,
    "max_position_embeddings": 128,
    "rope_parameters": {"rope_theta": 10000.0},
    "rms_norm_eps": 1e-5,
    "tie_word_embeddings": False,
}


@pytest.fixture(autouse=True)
def _clean_global_ctx():
    yield
    reset_global_ctx()


def _config():
    return parse_config(type("Hf", (), {"to_dict": lambda self: TINY_CONFIG})())


def test_config_derives_and_carries_indexer_types_verbatim_when_explicit():
    config = _config()
    assert config.attrs["indexer_types"] == INDEXER_TYPES


def test_config_derives_indexer_types_from_freq_offset_when_absent():
    cfg_dict = dict(TINY_CONFIG)
    del cfg_dict["indexer_types"]
    cfg_dict["index_topk_freq"] = 2
    cfg_dict["index_skip_topk_offset"] = 0
    config = parse_config(type("Hf", (), {"to_dict": lambda self: cfg_dict})())
    # Real formula: "full" if (max(i - offset + 1, 0) % freq) == 0 else "shared"
    expected = ["full" if (max(i - 0 + 1, 0) % 2) == 0 else "shared" for i in range(L)]
    assert config.attrs["indexer_types"] == expected


def test_model_class_builds_shared_layers_without_their_own_indexer():
    config = _config()
    model = get_model_class("GlmMoeDsaForCausalLM", config, device=torch.device("cpu"))
    assert model.layers[0].self_attn.skip_topk is False
    assert model.layers[1].self_attn.skip_topk is True
    assert not hasattr(model.layers[1].self_attn, "wq_b")  # no indexer weights at all
    assert hasattr(model.layers[0].self_attn, "wq_b")
    assert isinstance(model.layers[0].mlp, _GlmMoeDsaMLP)  # dense (first_k_dense_replace)
    assert isinstance(model.layers[1].mlp, _GlmMoeDsaMoE)


def test_shared_layer_reuses_the_previous_full_layers_topk_indices():
    """The real point of cross-layer sharing: a "shared" layer's forward
    call must receive prev_topk_indices and return it UNCHANGED, not
    compute its own."""
    config = _config()
    torch.manual_seed(0)
    full_layer = _GlmMoeDsaMLA(config, torch.device("cpu"), torch.float32, layer_id=0)
    shared_layer = _GlmMoeDsaMLA(config, torch.device("cpu"), torch.float32, layer_id=1)
    assert shared_layer.index_topk and shared_layer.skip_topk

    T = 6
    hidden_states = torch.randn(T, H)
    positions = torch.arange(T)
    ctx = Context(page_size=1)
    ctx.kv_cache = BaseKVCachePool(config, page_size=1, num_pages=32, device=torch.device("cpu"), dtype=torch.float32)
    pt = torch.zeros((2, 32), dtype=torch.int32)
    pt[0, :T] = torch.arange(T)
    ctx.page_table = pt
    ctx.kv_cache.attach_page_table(pt)
    req = Req(
        input_ids=list(range(T)), table_idx=0, cached_len=0, output_len=1, uid=0,
        sampling_params=SamplingParams(), cache_handle=None,
    )
    req.device_len = T
    batch = Batch(reqs=[req], phase="prefill")
    set_global_ctx(ctx)
    ctx._batch = batch

    _, full_topk = full_layer(hidden_states, positions, 0, ctx, batch, prev_topk_indices=None)
    _, shared_topk = shared_layer(hidden_states, positions, 0, ctx, batch, prev_topk_indices=full_topk)
    torch.testing.assert_close(full_topk, shared_topk)

    with pytest.raises(ValueError, match="malformed"):
        shared_layer(hidden_states, positions, 0, ctx, batch, prev_topk_indices=None)


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
        if INDEXER_TYPES[l] == "full":
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


def test_iter_weights_covers_indexer_only_on_full_layers(tmp_path):
    model_path = _write_tiny_checkpoint(tmp_path)
    names = {n for n, _ in iter_weights(model_path, torch.device("cpu"))}
    assert "model.layers.0.self_attn.wq_b.weight" in names  # full
    assert "model.layers.1.self_attn.wq_b.weight" not in names  # shared -- no indexer at all


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
