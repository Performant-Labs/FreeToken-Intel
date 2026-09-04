"""Unit + e2e tests for OLMoE (issue ``models-olmoe-attn``, #224): flat
(non-per-head) q/k RMS-norm attention and the flat top-8/64 MoE router
with no renormalization (``norm_topk_prob: false``, confirmed from the
real downloaded checkpoint's own config.json -- see module docstring in
``freetoken.models.olmoe``).
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
from freetoken.models.olmoe import OlmoeForCausalLM, _OlmoeAttention, _OlmoeMoE, iter_weights, parse_config
from freetoken.models.register import get_model_class

DEVICE = "cpu"

H, V, L = 32, 64, 2
NH = 4
INTER, E, TOPK = 24, 6, 2

TINY_CONFIG = {
    "architectures": ["OlmoeForCausalLM"],
    "model_type": "olmoe",
    "hidden_size": H,
    "vocab_size": V,
    "num_hidden_layers": L,
    "num_attention_heads": NH,
    "num_key_value_heads": NH,
    "intermediate_size": INTER,
    "num_experts": E,
    "num_experts_per_tok": TOPK,
    "norm_topk_prob": False,
    "clip_qkv": None,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "max_position_embeddings": 128,
    "tie_word_embeddings": False,
}


@pytest.fixture(autouse=True)
def _clean_global_ctx():
    yield
    reset_global_ctx()


def _config():
    return parse_config(type("Hf", (), {"to_dict": lambda self: TINY_CONFIG})())


def test_config_confirms_real_checkpoint_shape_not_class_defaults():
    config = _config()
    assert config.num_attention_heads == config.num_key_value_heads == NH  # plain MHA, not MQA
    assert config.attrs["norm_topk_prob"] is False
    assert config.num_experts == E
    assert config.num_experts_per_tok == TOPK


def test_moe_router_matches_hand_computed_reference_without_renormalization():
    """The real, distinctive point of OLMoE's router: norm_topk_prob=False
    means the raw softmax top-k weights are used AS-IS, unlike every
    grouped-topk MoE in this port so far (which always renormalizes)."""
    torch.manual_seed(0)
    config = _config()
    moe = _OlmoeMoE(config, torch.device("cpu"), torch.float32)
    # LinearReplicated leaves its weight uninitialized (torch.empty) --
    # production code always fills it from a real checkpoint via the
    # loader, but an isolated unit test must seed it explicitly, matching
    # this port's own established convention for standalone module tests.
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        moe.gate.weight.copy_(torch.randn(moe.gate.weight.shape, generator=gen) * 0.1)
        for expert in moe.experts:
            expert.gate_proj.weight.copy_(torch.randn(expert.gate_proj.weight.shape, generator=gen) * 0.1)
            expert.up_proj.weight.copy_(torch.randn(expert.up_proj.weight.shape, generator=gen) * 0.1)
            expert.down_proj.weight.copy_(torch.randn(expert.down_proj.weight.shape, generator=gen) * 0.1)
    x = torch.randn(5, H)

    out = moe(x)
    assert out.shape == (5, H)

    # Independent recomputation.
    routing = moe.gate(x)
    gate_log = torch.softmax(routing, dim=-1, dtype=torch.float32)
    top_w, top_idx = torch.topk(gate_log, TOPK, dim=-1)

    expected = torch.zeros_like(x)
    for t in range(5):
        for slot in range(TOPK):
            e = int(top_idx[t, slot])
            w = top_w[t, slot]
            expert = moe.experts[e]
            y = expert.down_proj(torch.nn.functional.silu(expert.gate_proj(x[t])) * expert.up_proj(x[t]))
            expected[t] += w * y
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_moe_router_norm_topk_prob_flag_actually_changes_output():
    """Discriminating test: this specific seed's softmax happens to be
    near-one-hot (top-1 weight ~1.0), so the reference test above cannot by
    itself prove ``norm_topk_prob=False`` is honored (a buggy always-
    renormalize implementation would coincidentally match too). Build two
    identically-weighted MoE blocks differing only in ``norm_topk_prob`` and
    confirm they produce genuinely different output on an input engineered
    to give close top-2 logits (a non-degenerate weight split)."""
    torch.manual_seed(1)
    config_false = _config()
    config_true = _config()
    config_true.attrs["norm_topk_prob"] = True

    moe_false = _OlmoeMoE(config_false, torch.device("cpu"), torch.float32)
    moe_true = _OlmoeMoE(config_true, torch.device("cpu"), torch.float32)

    # LinearReplicated leaves weights uninitialized (torch.empty) -- seed
    # everything explicitly (see the reference test above for why).
    gen = torch.Generator().manual_seed(2)
    with torch.no_grad():
        moe_false.gate.weight.copy_(torch.randn(moe_false.gate.weight.shape, generator=gen) * 0.1)
        for expert in moe_false.experts:
            expert.gate_proj.weight.copy_(torch.randn(expert.gate_proj.weight.shape, generator=gen) * 0.1)
            expert.up_proj.weight.copy_(torch.randn(expert.up_proj.weight.shape, generator=gen) * 0.1)
            expert.down_proj.weight.copy_(torch.randn(expert.down_proj.weight.shape, generator=gen) * 0.1)
        # Engineer close top-2 gate logits: a near-uniform gate weight plus a
        # small perturbation, so top_w's two entries are neither ~[1,0] nor
        # exactly equal (a genuinely non-degenerate split to renormalize).
        moe_false.gate.weight.zero_()
        moe_false.gate.weight[0, 0] = 0.05
        moe_false.gate.weight[1, 0] = 0.04
    moe_true.load_state_dict(moe_false.state_dict())  # identical weights

    x = torch.zeros(1, H)
    x[0, 0] = 1.0

    out_false = moe_false(x)
    out_true = moe_true(x)
    assert not torch.allclose(out_false, out_true, atol=1e-6), (
        "norm_topk_prob=False vs True must produce different output when the "
        "top-k weights are a genuine (non-degenerate) split -- if they match, "
        "the flag is being ignored"
    )


def test_attention_uses_flat_not_per_head_qk_norm():
    """The real, distinctive point of OLMoE's attention: q_norm/k_norm are
    RMSNorm over the FULL flat projected vector (size num_heads*head_dim),
    applied BEFORE the per-head reshape -- not a per-head-sized norm
    applied after reshaping (the qwen3/qwen3_moe convention). Confirms the
    norm weight shapes are flat, not head_dim-sized."""
    config = _config()
    attn = _OlmoeAttention(config, torch.device("cpu"), torch.float32, layer_id=0)
    head_dim = H // NH
    assert attn.q_norm.weight.shape == (NH * head_dim,)
    assert attn.k_norm.weight.shape == (NH * head_dim,)
    assert attn.q_norm.weight.shape != (head_dim,)


def _write_tiny_checkpoint(tmp_path) -> str:
    from safetensors.torch import save_file

    model_path = tmp_path / "ckpt"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(TINY_CONFIG))

    state = {}
    gen = torch.Generator().manual_seed(0)

    def add(name, shape):
        state[name] = torch.randn(shape, generator=gen, dtype=torch.float32) * 0.02

    head_dim = H // NH
    add("model.embed_tokens.weight", (V, H))
    for l in range(L):
        p = f"model.layers.{l}"
        add(f"{p}.input_layernorm.weight", (H,))
        add(f"{p}.self_attn.q_proj.weight", (NH * head_dim, H))
        add(f"{p}.self_attn.k_proj.weight", (NH * head_dim, H))
        add(f"{p}.self_attn.v_proj.weight", (NH * head_dim, H))
        add(f"{p}.self_attn.o_proj.weight", (H, NH * head_dim))
        add(f"{p}.self_attn.q_norm.weight", (NH * head_dim,))
        add(f"{p}.self_attn.k_norm.weight", (NH * head_dim,))
        add(f"{p}.post_attention_layernorm.weight", (H,))
        add(f"{p}.mlp.gate.weight", (E, H))
        for e in range(E):
            eb = f"{p}.mlp.experts.{e}"
            add(f"{eb}.gate_proj.weight", (INTER, H))
            add(f"{eb}.up_proj.weight", (INTER, H))
            add(f"{eb}.down_proj.weight", (H, INTER))
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


def test_iter_weights_covers_router_and_experts(tmp_path):
    model_path = _write_tiny_checkpoint(tmp_path)
    names = {n for n, _ in iter_weights(model_path, torch.device("cpu"))}
    assert "model.layers.0.mlp.gate.weight" in names
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" in names
    assert "model.layers.0.self_attn.q_norm.weight" in names


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
