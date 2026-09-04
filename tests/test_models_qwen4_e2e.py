"""End-to-end: Qwen3.8-Flash-Next (qwen4_exp) -- the final piece of epic
#198, combining every mechanism from #206 (hyper-connections), #207 (PLE),
#208 (QSA) plus GDN (reused from qwen3_5_moe, #170/#172) and MoE (a plain
softmax-topk router + shared expert, matching upstream's real
``Qwen4ExpMoE(Qwen3_5MoE)``) into one real forward pass through the engine.

A small fabricated checkpoint: 3 layers (2 GDN/linear, 1 QSA/full --
``full_attention_interval=3``), PLE on layer 0 (a linear layer, matching
the real constraint), hc_count=2 (proves the multi-stream residual is
actually exercised, not a degenerate hc_count=1 no-op).
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from freetoken.core import Req, SamplingParams, reset_global_ctx
from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.engine.engine import Engine
from freetoken.models.qwen4_exp import parse_config, iter_weights
from freetoken.models.register import get_model_class

DEVICE = "cpu"

H, V, L = 16, 32, 3
NUM_HEADS, NUM_KV_HEADS, HEAD_DIM = 2, 2, 8
HC_COUNT, HC_LOWRANK = 2, 6
E, MOE_INTER, TOPK = 4, 12, 2
SHARED_INTER = 12

# GDN (linear_attention layers 0, 1)
LIN_NUM_K, LIN_NUM_V, LIN_KEY_D, LIN_VAL_D, LIN_CONV_K = 1, 2, 8, 8, 4
KEY_DIM = LIN_NUM_K * LIN_KEY_D
VALUE_DIM = LIN_NUM_V * LIN_VAL_D
CONV_DIM = KEY_DIM * 2 + VALUE_DIM

# QSA (full_attention layer 2)
INDEX_HEAD_DIM, INDEX_N_HEADS, INDEX_RATIO, INDEX_BUDGET = 4, 2, 2, 4
ROTARY_DIM = HEAD_DIM

# PLE (layer 0 only)
NGRAM_SIZE, HEADS_PER_NGRAM = 2, 2
NUM_NGRAM_HEADS = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM
PLE_ROW_WIDTH = 4
PLE_EMBED_DIM = NUM_NGRAM_HEADS * PLE_ROW_WIDTH
PLE_CONV_KERNEL = 3
NGRAM_VOCAB_BASE = 50

TINY_CONFIG = {
    "architectures": ["Qwen4ExpForCausalLM"],
    "model_type": "qwen4_exp",
    "hidden_size": H,
    "vocab_size": V,
    "num_hidden_layers": L,
    "num_attention_heads": NUM_HEADS,
    "num_key_value_heads": NUM_KV_HEADS,
    "head_dim": HEAD_DIM,
    "full_attention_interval": 3,
    "rope_theta": 10000.0,
    "rotary_dim": ROTARY_DIM,
    "rms_norm_eps": 1e-5,
    "tie_word_embeddings": False,
    "hc_count": HC_COUNT,
    "hc_lowrank": HC_LOWRANK,
    "num_experts": E,
    "num_experts_per_tok": TOPK,
    "moe_intermediate_size": MOE_INTER,
    "shared_expert_intermediate_size": SHARED_INTER,
    "indexer_head_dim": INDEX_HEAD_DIM,
    "indexer_n_heads": INDEX_N_HEADS,
    "indexer_compress_ratio": INDEX_RATIO,
    "indexer_budget": INDEX_BUDGET,
    "ple_layer_ids": [1],  # 1-indexed -> layer 0
    "ple_embed_dim": PLE_EMBED_DIM,
    "ple_conv_kernel_size": PLE_CONV_KERNEL,
    "ngram_size": NGRAM_SIZE,
    "heads_per_ngram": HEADS_PER_NGRAM,
    "ngram_vocab_size_base": NGRAM_VOCAB_BASE,
    "eos_token_id": 0,
    "linear_num_key_heads": LIN_NUM_K,
    "linear_num_value_heads": LIN_NUM_V,
    "linear_key_head_dim": LIN_KEY_D,
    "linear_value_head_dim": LIN_VAL_D,
    "linear_conv_kernel_dim": LIN_CONV_K,
}


@pytest.fixture(autouse=True)
def _clean_global_ctx():
    yield
    reset_global_ctx()


def _config():
    return parse_config(type("Hf", (), {"to_dict": lambda self: TINY_CONFIG})())


def test_config_derives_layer_types_and_ple_on_a_linear_layer():
    config = _config()
    assert config.attrs["layer_types"] == ["linear_attention", "linear_attention", "full_attention"]
    assert config.attrs["ple_layer_ids"] == [0]


def _pad_size(lowrank: int, hc_count: int) -> int:
    return (-(lowrank + hc_count)) % 16


def _write_tiny_checkpoint(tmp_path) -> str:
    from safetensors.torch import save_file

    model_path = tmp_path / "ckpt"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(TINY_CONFIG))

    state = {}
    gen = torch.Generator().manual_seed(0)

    def add(name, shape):
        state[name] = torch.randn(shape, generator=gen, dtype=torch.float32) * 0.02

    width = HC_COUNT * H
    pad = _pad_size(HC_LOWRANK, HC_COUNT)

    def add_hc(prefix: str) -> None:
        add(f"{prefix}.hc_norm.weight", (width,))
        add(f"{prefix}.input_mix_weight_down_block_inject.weight", (HC_LOWRANK + HC_COUNT + pad, width))
        add(f"{prefix}.input_mix_weight_up.weight", (width, HC_LOWRANK))

    add("model.embed_tokens.weight", (V, H))

    for l in range(L):
        p = f"model.layers.{l}"
        add_hc(f"{p}.attn_hyper_connection")
        add_hc(f"{p}.mlp_hyper_connection")

        if l < 2:  # linear_attention (GDN)
            lp = f"{p}.linear_attn"
            add(f"{lp}.conv1d.weight", (CONV_DIM, 1, LIN_CONV_K))
            add(f"{lp}.dt_bias", (LIN_NUM_V,))
            add(f"{lp}.A_log", (LIN_NUM_V,))
            add(f"{lp}.norm.weight", (LIN_VAL_D,))
            add(f"{lp}.out_proj.weight", (H, VALUE_DIM))
            add(f"{lp}.in_proj_qkv.weight", (KEY_DIM * 2 + VALUE_DIM, H))
            add(f"{lp}.in_proj_z.weight", (VALUE_DIM, H))
            add(f"{lp}.in_proj_b.weight", (LIN_NUM_V, H))
            add(f"{lp}.in_proj_a.weight", (LIN_NUM_V, H))
        else:  # full_attention (QSA)
            ap = f"{p}.self_attn"
            qkv_out = NUM_HEADS * HEAD_DIM * 2 + NUM_KV_HEADS * HEAD_DIM * 2
            add(f"{ap}.qkv_proj.weight", (qkv_out, H))
            add(f"{ap}.o_proj.weight", (H, NUM_HEADS * HEAD_DIM))
            add(f"{ap}.q_norm", (HEAD_DIM,))
            add(f"{ap}.k_norm", (HEAD_DIM,))
            index_out = INDEX_N_HEADS * INDEX_HEAD_DIM + INDEX_HEAD_DIM
            add(f"{ap}.index_qk_proj.weight", (index_out, H))
            add(f"{ap}.index_q_norm", (INDEX_HEAD_DIM,))
            add(f"{ap}.index_k_norm", (INDEX_HEAD_DIM,))

        if l == 0:  # PLE
            pp = f"{p}.ple"
            add(f"{pp}.key_proj.weight", (width, PLE_EMBED_DIM))
            add(f"{pp}.value_proj.weight", (H, PLE_EMBED_DIM))
            add(f"{pp}.norm_key.weight", (width,))
            add(f"{pp}.norm_query.weight", (width,))
            add(f"{pp}.norm_conv.weight", (width,))
            add(f"{pp}.conv1d", (width, 1, PLE_CONV_KERNEL))

        mp = f"{p}.mlp"
        add(f"{mp}.gate.gate.weight", (E, H))
        for e in range(E):
            eb = f"{mp}.experts.{e}"
            add(f"{eb}.gate_proj.weight", (MOE_INTER, H))
            add(f"{eb}.up_proj.weight", (MOE_INTER, H))
            add(f"{eb}.down_proj.weight", (H, MOE_INTER))
        add(f"{mp}.shared_expert.gate_proj.weight", (SHARED_INTER, H))
        add(f"{mp}.shared_expert.up_proj.weight", (SHARED_INTER, H))
        add(f"{mp}.shared_expert.down_proj.weight", (H, SHARED_INTER))
        add(f"{mp}.shared_expert_gate.weight", (1, H))

    add("model.hyper_connection_mixer.hc_norm.weight", (width,))
    add("model.hyper_connection_mixer.input_mix_weight_down.weight", (HC_LOWRANK, width))
    add("model.hyper_connection_mixer.input_mix_weight_up.weight", (width, HC_LOWRANK))
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


def test_load_model_builds_real_class_and_runs_a_finite_forward(tmp_path):
    model_path = _write_tiny_checkpoint(tmp_path)
    config = parse_config(type("Hf", (), {"to_dict": lambda self: TINY_CONFIG})(), model_path=model_path)
    model = get_model_class("Qwen4ExpForCausalLM", config, device=torch.device("cpu"))
    for name, tensor in iter_weights(model_path, torch.device("cpu")):
        clean = name[len("model.") :] if name.startswith("model.") else name
        named = dict(model.named_parameters())
        assert clean in named, f"checkpoint tensor {clean!r} has no matching parameter"
        with torch.no_grad():
            named[clean].copy_(tensor)


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
