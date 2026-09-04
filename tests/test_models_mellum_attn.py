"""Unit tests for Mellum2-12B-A2.5B's attention + router primitives (issue
`models-mellum-attn`, #227, first child of the Mellum epic #226).

Standalone -- not yet wired into a decoder layer or the engine (that's
#228's job). Pins: (1) the flat top-8/64 router's hand-computed math
(direct reuse of `qwen3_moe`'s own router shape), (2) a real attention
forward pass through both a `full_attention` (YaRN RoPE) and a
`sliding_attention` (plain RoPE) layer, and (3) PR #234's KV-cache fix --
`write_kv`'s `out_loc` must be the PHYSICAL page-table slot, never the
raw logical position, using a deliberately NON-identity page table (slot
0 reserved, matching the real `MHAKVCache` allocator's own convention)
so an identity-table test could not silently hide the bug.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from freetoken.attention.triton import TritonAttentionBackend
from freetoken.core import Batch, Context, Req, SamplingParams, reset_global_ctx, set_global_ctx
from freetoken.kvcache.base import BaseKVCachePool
from freetoken.models.mellum import MellumAttention, mellum_moe_router, parse_config

H, V, L = 32, 64, 4
NH, KVH, HD = 4, 2, 16
E, TOPK = 8, 2

TINY_CONFIG = {
    "architectures": ["MellumForCausalLM"],
    "model_type": "mellum",
    "hidden_size": H,
    "vocab_size": V,
    "num_hidden_layers": L,
    "num_attention_heads": NH,
    "num_key_value_heads": KVH,
    "head_dim": HD,
    "intermediate_size": 48,
    "moe_intermediate_size": 24,
    "num_experts": E,
    "num_experts_per_tok": TOPK,
    "norm_topk_prob": True,
    "rms_norm_eps": 1e-6,
    "sliding_window": 4,
    "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
    "mlp_layer_types": ["sparse"] * L,
    "rope_parameters": {
        "full_attention": {
            "rope_type": "yarn",
            "rope_theta": 500000.0,
            "factor": 16.0,
            "original_max_position_embeddings": 64,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "attention_factor": 1.2,
        },
        "sliding_attention": {"rope_type": "default", "rope_theta": 500000.0},
    },
    "max_position_embeddings": 4096,
    "tie_word_embeddings": False,
}


@pytest.fixture(autouse=True)
def _clean_global_ctx():
    yield
    reset_global_ctx()


def _config():
    return parse_config(type("Hf", (), {"to_dict": lambda self: TINY_CONFIG})())


def test_config_carries_real_per_layer_type_fields():
    config = _config()
    assert config.attrs["layer_types"] == TINY_CONFIG["layer_types"]
    assert config.attrs["mlp_layer_types"] == ["sparse"] * L
    assert config.attrs["sliding_window"] == 4
    assert config.head_dim == HD  # explicit, not derived (H/NH != HD)


def test_mellum_moe_router_matches_hand_computed_reference():
    torch.manual_seed(0)
    T = 5
    hidden = torch.randn(T, H)
    gate_weight = torch.randn(E, H) * 0.1

    top_w, top_idx = mellum_moe_router(hidden, gate_weight, TOPK)
    assert top_w.shape == (T, TOPK)
    assert top_idx.shape == (T, TOPK)

    # Independent hand-computed reference.
    logits = hidden @ gate_weight.t()
    probs = torch.softmax(logits, dim=-1)
    exp_w, exp_idx = torch.topk(probs, TOPK, dim=-1)
    exp_w = exp_w / exp_w.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(top_idx, exp_idx)
    torch.testing.assert_close(top_w, exp_w, atol=1e-5, rtol=1e-5)
    # norm_topk_prob=True: the selected weights always renormalize to 1.
    torch.testing.assert_close(top_w.sum(dim=-1), torch.ones(T))


def _run_attention_forward(layer_id: int, layer_type: str, num_pages: int = 32):
    config = _config()
    torch.manual_seed(1)
    attn = MellumAttention(config, torch.device("cpu"), torch.float32, layer_id=layer_id, layer_type=layer_type)
    # LinearReplicated/RMSNorm weights are `torch.empty` (loader-filled in
    # production) -- a unit test must seed them itself or every forward is
    # NaN from the first projection, before rope/KV-cache even run.
    gen = torch.Generator().manual_seed(2024)
    with torch.no_grad():
        for p in attn.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.1)

    T = 3
    hidden_states = torch.randn(T, H)
    positions = torch.arange(T)

    ctx = Context(page_size=1)
    ctx.kv_cache = BaseKVCachePool(config, page_size=1, num_pages=num_pages, device=torch.device("cpu"), dtype=torch.float32)
    # Deliberately NON-identity page table (slot 0 reserved, matching the
    # real MHAKVCache allocator's own convention) -- an identity table is
    # exactly what let PR #234's bug hide in earlier tests.
    pt = torch.zeros((1, num_pages), dtype=torch.int32)
    pt[0, :T] = torch.arange(1, T + 1)
    ctx.page_table = pt
    ctx.kv_cache.attach_page_table(pt)
    ctx.attn_backend = TritonAttentionBackend(config)
    req = Req(
        input_ids=list(range(T)), table_idx=0, cached_len=0, output_len=1, uid=0,
        sampling_params=SamplingParams(), cache_handle=None,
    )
    req.device_len = T
    batch = Batch(reqs=[req], phase="prefill", positions=positions.clone(), extend_lens=[T])
    set_global_ctx(ctx)
    ctx._batch = batch

    captured = {}
    orig_write_kv = ctx.kv_cache.write_kv

    def spy_write_kv(k, v, out_loc, layer_id=0):
        captured["out_loc"] = out_loc.clone()
        return orig_write_kv(k, v, out_loc, layer_id)

    ctx.kv_cache.write_kv = spy_write_kv
    try:
        out = attn.forward(hidden_states, positions, 0, ctx, batch)
    finally:
        ctx.kv_cache.write_kv = orig_write_kv
    return out, captured, positions, pt


def test_full_attention_layer_uses_yarn_rope_and_physical_slots():
    out, captured, positions, pt = _run_attention_forward(3, "full_attention")
    assert out.shape == (3, H)
    assert torch.isfinite(out).all()
    expected_slots = pt[0, positions.long()]
    assert 0 not in expected_slots.tolist()
    assert captured["out_loc"].equal(expected_slots)
    assert not captured["out_loc"].equal(positions)


def test_sliding_attention_layer_uses_plain_rope_and_physical_slots():
    out, captured, positions, pt = _run_attention_forward(0, "sliding_attention")
    assert out.shape == (3, H)
    assert torch.isfinite(out).all()
    expected_slots = pt[0, positions.long()]
    assert captured["out_loc"].equal(expected_slots)
    assert not captured["out_loc"].equal(positions)


def test_full_and_sliding_layers_use_different_rope_tables():
    config = _config()
    full = MellumAttention(config, torch.device("cpu"), torch.float32, layer_id=3, layer_type="full_attention")
    sliding = MellumAttention(config, torch.device("cpu"), torch.float32, layer_id=0, layer_type="sliding_attention")
    assert not torch.allclose(full.inv_freq, sliding.inv_freq)
    assert full.attention_scaling != 1.0  # YaRN's attention_factor
    assert sliding.attention_scaling == 1.0  # plain RoPE, no scaling
    assert full.sliding_window == 0
    assert sliding.sliding_window == TINY_CONFIG["sliding_window"]
