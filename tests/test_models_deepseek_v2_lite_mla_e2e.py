"""End-to-end + numerical-round-trip tests for DeepSeek-Coder-V2-Lite's
Multi-head Latent Attention (issue `models-dsv2lite-mla`, #217, first child
of the DeepSeek-Coder-V2-Lite epic #216).

Mirrors `deepseek_v4`'s own MLA-only test file (`test_models_deepseek_v4_mla_e2e.py`)
in shape, plus new coverage for the two real architectural differences this
module's own docstring documents: interleaved-pair RoPE (not `deepseek_v4`'s
half-split) and real YaRN rope scaling (`deepseek_v4`'s own #190 checkpoint
never exercised a non-default rope_type). The YaRN math is tested against
the REAL, installed `transformers.modeling_rope_utils._compute_yarn_parameters`
as an oracle -- the strongest possible grounding, since it's a verbatim port
of that exact function, not a reimplementation from a spec.
"""
from __future__ import annotations

import json
import math

import pytest

torch = pytest.importorskip("torch")

from freetoken.core import Batch, Context, Req, SamplingParams, reset_global_ctx, set_global_ctx
from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.engine.engine import Engine
from freetoken.kvcache.base import BaseKVCachePool
from freetoken.models.deepseek_v2_lite import (
    _DeepseekV2LiteMLA,
    apply_interleaved_rope,
    iter_weights,
    parse_config,
    yarn_rope_params,
    yarn_softmax_scale,
)
from freetoken.models.register import get_model_class

DEVICE = "cpu"

H, V, L = 32, 64, 2
NH = 4
KV_LORA = 16
QK_ROPE, QK_NOPE, V_HEAD = 4, 6, 8
INTER = 48

TINY_CONFIG = {
    "architectures": ["DeepseekV2ForCausalLM"],
    "model_type": "deepseek_v2",
    "hidden_size": H,
    "vocab_size": V,
    "num_hidden_layers": L,
    "num_attention_heads": NH,
    "q_lora_rank": None,  # the real DeepSeek-Coder-V2-Lite-Base checkpoint's own value
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

YARN_CONFIG = dict(
    TINY_CONFIG,
    rope_scaling={
        "type": "yarn",
        "factor": 4,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 0.707,
        "mscale_all_dim": 0.707,
        "original_max_position_embeddings": 32,
    },
)


@pytest.fixture(autouse=True)
def _clean_global_ctx():
    yield
    reset_global_ctx()


def _config(cfg_dict=TINY_CONFIG):
    return parse_config(type("Hf", (), {"to_dict": lambda self: cfg_dict})())


def test_q_lora_rank_null_is_read_from_the_real_checkpoint_not_the_hf_class_default():
    """The real point of this module's own defensive-parsing note: the
    installed transformers' real DeepseekV2Config class has a non-None
    default (1536) for q_lora_rank, but the actual checkpoint sets it to
    null. parse_config must resolve to None here, not 1536."""
    config = _config()
    assert config.q_lora_rank is None


def test_pool_is_sized_for_the_compressed_latent_not_per_head_kv():
    config = _config()
    assert config.num_key_value_heads == 1
    assert config.head_dim == KV_LORA + QK_ROPE


def test_apply_interleaved_rope_matches_real_hf_complex_reference():
    """Confirms the real-valued interleaved-pair rotation this module uses
    is numerically identical to the real HF ``apply_rotary_emb`` (complex
    multiplication via ``torch.polar``/``view_as_complex``)."""
    torch.manual_seed(0)
    T, D = 5, QK_ROPE
    x = torch.randn(T, NH, D)
    positions = torch.arange(T)
    theta = 10000.0
    inv_freq = 1.0 / (theta ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    freqs = torch.outer(positions.to(torch.float32), inv_freq)  # [T, D/2]
    cos, sin = freqs.cos(), freqs.sin()

    got = apply_interleaved_rope(x, cos[:, None, :], sin[:, None, :])

    # Real HF reference: torch.polar + view_as_complex.
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # [T, D/2] complex
    x_complex = torch.view_as_complex(x.float().reshape(T, NH, D // 2, 2))
    expected = torch.view_as_real(x_complex * freqs_cis.unsqueeze(1)).flatten(2).to(x.dtype)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_yarn_rope_params_matches_real_hf_compute_yarn_parameters_oracle():
    """Strongest possible grounding: call the REAL, installed
    ``transformers.modeling_rope_utils._compute_yarn_parameters`` on an
    equivalent config object and compare against this module's own port."""
    from transformers.modeling_rope_utils import _compute_yarn_parameters

    rope_scaling = YARN_CONFIG["rope_scaling"]
    rotary_dim = QK_ROPE
    base = YARN_CONFIG["rope_theta"]
    max_pos = YARN_CONFIG["max_position_embeddings"]

    got_inv_freq, got_factor = yarn_rope_params(rotary_dim, base, max_pos, rope_scaling)

    class _HfConfig:
        head_dim = rotary_dim
        hidden_size = rotary_dim * NH
        num_attention_heads = NH
        max_position_embeddings = max_pos
        rope_parameters = {"rope_theta": base, **rope_scaling}

        def standardize_rope_params(self):
            pass

    expected_inv_freq, expected_factor = _compute_yarn_parameters(_HfConfig())
    torch.testing.assert_close(got_inv_freq, expected_inv_freq, atol=1e-6, rtol=1e-6)
    assert math.isclose(got_factor, expected_factor, rel_tol=1e-6)


def test_yarn_softmax_scale_matches_real_hf_yarn_apply_mscale_oracle():
    from transformers.models.deepseek_v2.modeling_deepseek_v2 import yarn_apply_mscale

    base_scale = (QK_NOPE + QK_ROPE) ** -0.5
    rope_scaling = YARN_CONFIG["rope_scaling"]
    got = yarn_softmax_scale(base_scale, rope_scaling)
    expected = yarn_apply_mscale({"rope_type": "yarn", **rope_scaling}, base_scale)
    assert math.isclose(got, expected, rel_tol=1e-6)


def test_yarn_softmax_scale_is_a_noop_for_default_rope():
    base_scale = (QK_NOPE + QK_ROPE) ** -0.5
    assert yarn_softmax_scale(base_scale, None) == base_scale
    assert yarn_softmax_scale(base_scale, {"type": "default"}) == base_scale


def test_compress_cache_decompress_round_trip_is_lossless():
    config = _config()
    torch.manual_seed(0)
    mla = _DeepseekV2LiteMLA(config, torch.device("cpu"), torch.float32, layer_id=0)

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

    compressed_kv = mla.kv_a_proj_with_mqa(hidden_states)
    kv_latent_direct, k_rot_direct = compressed_kv.split([mla.kv_lora_rank, mla.qk_rope_head_dim], dim=-1)
    kv_latent_direct = mla.kv_a_layernorm(kv_latent_direct)

    cached_tok, _ = ctx.kv_cache.read_kv(0, torch.arange(T), 0)
    cached = cached_tok.squeeze(1)
    cached_latent, cached_k_rot_rotated = cached.split([mla.kv_lora_rank, mla.qk_rope_head_dim], dim=-1)

    torch.testing.assert_close(cached_latent, kv_latent_direct)

    freqs = torch.outer(positions.to(torch.float32), mla.inv_freq) * mla.attention_factor
    cos, sin = freqs.cos(), freqs.sin()
    k_rot_direct_rotated = apply_interleaved_rope(k_rot_direct, cos, sin)
    torch.testing.assert_close(cached_k_rot_rotated, k_rot_direct_rotated)


def _write_tiny_checkpoint(tmp_path, cfg_dict=TINY_CONFIG) -> str:
    from safetensors.torch import save_file

    model_path = tmp_path / "ckpt"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(cfg_dict))

    config = _config(cfg_dict)
    state = {}
    gen = torch.Generator().manual_seed(0)

    def add(name, shape):
        state[name] = torch.randn(shape, generator=gen, dtype=torch.float32) * 0.02

    qk_head_dim = QK_NOPE + QK_ROPE
    add("model.embed_tokens.weight", (V, H))
    for l in range(L):
        p = f"model.layers.{l}"
        add(f"{p}.input_layernorm.weight", (H,))
        if config.q_lora_rank:
            add(f"{p}.self_attn.q_a_proj.weight", (config.q_lora_rank, H))
            add(f"{p}.self_attn.q_a_layernorm.weight", (config.q_lora_rank,))
            add(f"{p}.self_attn.q_b_proj.weight", (NH * qk_head_dim, config.q_lora_rank))
        else:
            add(f"{p}.self_attn.q_proj.weight", (NH * qk_head_dim, H))
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


def test_model_class_builds_real_mla_layers_with_plain_q_proj(tmp_path):
    config = _config()
    model = get_model_class("DeepseekV2ForCausalLM", config, device=torch.device("cpu"))
    assert isinstance(model.layers[0].self_attn, _DeepseekV2LiteMLA)
    assert model.layers[0].self_attn.kv_lora_rank == KV_LORA
    assert not hasattr(model.layers[0].self_attn, "q_a_proj")
    assert hasattr(model.layers[0].self_attn, "q_proj")


def test_iter_weights_covers_mla_projections(tmp_path):
    model_path = _write_tiny_checkpoint(tmp_path)
    names = {n for n, _ in iter_weights(model_path, torch.device("cpu"))}
    assert "model.layers.0.self_attn.kv_a_proj_with_mqa.weight" in names
    assert "model.layers.0.self_attn.kv_b_proj.weight" in names
    assert "model.layers.0.self_attn.q_proj.weight" in names


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


def test_engine_generate_with_yarn_rope_scaling(tmp_path):
    """The real checkpoint's own rope config (YaRN) exercised end to end,
    not just unit-tested in isolation."""
    model_path = _write_tiny_checkpoint(tmp_path, cfg_dict=YARN_CONFIG)
    engine = Engine(_engine_config(model_path, device=DEVICE))
    _add_prompt(engine, output_len=3)
    generated = engine.generate()
    vocab = engine.config.model_config.vocab_size
    assert len(generated[0]) == 3
    assert all(0 <= t < vocab for t in generated[0])
