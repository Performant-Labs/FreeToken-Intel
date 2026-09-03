"""End-to-end + numerical-round-trip tests for Multi-head Latent Attention
(issue `models-mla`, #190, child of the DeepSeek-V4 epic #21).

Two things are proven here (see this issue's own "Test strategy"):
1. The compress -> cache -> decompress round trip is LOSSLESS: reading the
   compressed latent back out of the paged KV pool and decompressing it
   reproduces exactly what a direct (no-cache) recomputation from the same
   hidden_states gives -- the real place a subtle write_kv/read_kv/reshape
   bug would hide, per this project's own established test discipline
   (verify the plumbing numerically, not just "doesn't crash").
2. A real forward pass through the actual Engine (prefill + decode,
   deterministic greedy) on a small fabricated DeepSeek-V4-shaped
   checkpoint -- this issue's own MLA-only scope (a plain dense MLP every
   layer; MoE/DSA are separate, later issues, see gpt_oss/__init__.py's own
   module docstring).
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
from freetoken.models.deepseek_v4 import _DeepseekV4MLA, iter_weights, parse_config
from freetoken.models.register import get_model_class

DEVICE = "cpu"

H, V, L = 32, 64, 2
NH = 4
Q_LORA, KV_LORA = 24, 16
QK_ROPE, QK_NOPE, V_HEAD = 4, 6, 8
INTER = 48

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
    "max_position_embeddings": 128,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
}


@pytest.fixture(autouse=True)
def _clean_global_ctx():
    yield
    reset_global_ctx()


def _config() -> "object":
    return parse_config(type("Hf", (), {"to_dict": lambda self: TINY_CONFIG})())


def test_pool_is_sized_for_the_compressed_latent_not_per_head_kv():
    config = _config()
    assert config.num_key_value_heads == 1
    assert config.head_dim == KV_LORA + QK_ROPE


def test_compress_cache_decompress_round_trip_is_lossless():
    """The real point of this issue's own accept bar: reading the cached
    compressed latent back and decompressing it must exactly reproduce a
    direct (no-cache) recomputation from the same hidden_states."""
    config = _config()
    torch.manual_seed(0)
    mla = _DeepseekV4MLA(config, torch.device("cpu"), torch.float32, layer_id=0)

    T = 5
    hidden_states = torch.randn(T, H)
    positions = torch.arange(T)

    ctx = Context(page_size=1)
    ctx.kv_cache = BaseKVCachePool(config, page_size=1, num_pages=32, device=torch.device("cpu"), dtype=torch.float32)
    pt = torch.zeros((1, 32), dtype=torch.int32)
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

    mla.forward(hidden_states, positions, 0, ctx, batch)

    # Direct recomputation, bypassing the cache entirely.
    compressed_kv = mla.kv_a_proj_with_mqa(hidden_states)
    kv_latent_direct, k_rot_direct = compressed_kv.split([mla.kv_lora_rank, mla.qk_rope_head_dim], dim=-1)
    kv_latent_direct = mla.kv_a_layernorm(kv_latent_direct)

    cached_tok, _ = ctx.kv_cache.read_kv(0, torch.arange(T), 0)
    cached = cached_tok.squeeze(1)  # [T, kv_lora_rank + rope]
    cached_latent, cached_k_rot_rotated = cached.split([mla.kv_lora_rank, mla.qk_rope_head_dim], dim=-1)

    torch.testing.assert_close(cached_latent, kv_latent_direct)

    # The cached rope-key slice is stored POST-rotation; recompute the same
    # rotation directly and compare.
    freqs = torch.outer(positions.to(torch.float32), mla.inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos(), emb.sin()
    from freetoken.models.deepseek_v4 import _rotate_half

    k_rot_f = k_rot_direct.to(torch.float32)
    k_rot_direct_rotated = k_rot_f * cos + _rotate_half(k_rot_f) * sin
    torch.testing.assert_close(cached_k_rot_rotated, k_rot_direct_rotated)


def _write_tiny_checkpoint(tmp_path) -> str:
    from safetensors.torch import save_file

    model_path = tmp_path / "ckpt"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(TINY_CONFIG))

    config = _config()
    state = {}

    def add(name, shape):
        state[name] = torch.randn(shape, dtype=torch.float32) * 0.02

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
        add(f"{p}.post_attention_layernorm.weight", (H,))
        add(f"{p}.mlp.gate_proj.weight", (INTER, H))
        add(f"{p}.mlp.up_proj.weight", (INTER, H))
        add(f"{p}.mlp.down_proj.weight", (H, INTER))
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


def test_model_class_builds_real_mla_layers(tmp_path):
    config = _config()
    model = get_model_class("DeepseekV4ForCausalLM", config, device=torch.device("cpu"))
    assert isinstance(model.layers[0].self_attn, _DeepseekV4MLA)
    assert model.layers[0].self_attn.kv_lora_rank == KV_LORA


def test_iter_weights_covers_mla_projections(tmp_path):
    model_path = _write_tiny_checkpoint(tmp_path)
    names = {n for n, _ in iter_weights(model_path, torch.device("cpu"))}
    assert "model.layers.0.self_attn.kv_a_proj_with_mqa.weight" in names
    assert "model.layers.0.self_attn.kv_b_proj.weight" in names
    assert "model.layers.0.self_attn.q_b_proj.weight" in names


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
