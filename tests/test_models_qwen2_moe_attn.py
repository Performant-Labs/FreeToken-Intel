"""Unit tests for Qwen1.5-MoE-A2.7B's bias-term attention + router/shared-
expert primitives (issue `models-qwen2moe-attn`, #221). Standalone -- not
yet wired into a decoder layer or the engine (that's #222's job); these pin
the attention and router/shared-expert math against independently
hand-computed references.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from freetoken.attention.triton import TritonAttentionBackend
from freetoken.core import Batch, Context, Req, SamplingParams, reset_global_ctx, set_global_ctx
from freetoken.kvcache.base import BaseKVCachePool
from freetoken.models.config import ModelConfig
from freetoken.models.qwen2_moe import Qwen2MoeAttention, qwen2moe_router, qwen2moe_shared_expert_output


@pytest.fixture(autouse=True)
def _clean_global_ctx():
    yield
    reset_global_ctx()


def test_qwen2moe_attention_forward_runs_and_bias_terms_are_real():
    """Real forward pass through a Context/BaseKVCachePool (identity page
    table -- the established pattern for a standalone-primitive test, per
    #221's own scope; the real MHAKVCache validation is #222's e2e job).
    Confirms finite output, and that the bias terms are actually wired
    (zeroing them changes the output -- proves has_bias=True is load-bearing,
    not a silently-ignored flag)."""
    torch.manual_seed(0)
    H, NH, NKV, D = 16, 4, 4, 8
    config = ModelConfig(
        hidden_size=H,
        num_attention_heads=NH,
        num_key_value_heads=NKV,
        head_dim=D,
        num_layers=1,
        rope_theta=10000.0,
        attrs={"qkv_bias": True},
    )
    dev = torch.device("cpu")
    attn = Qwen2MoeAttention(config, dev, torch.float32, layer_id=0)
    assert attn.q_proj.has_bias and attn.k_proj.has_bias and attn.v_proj.has_bias
    assert not attn.o_proj.has_bias
    with torch.no_grad():
        for p in attn.parameters():
            p.normal_(0, 0.02)

    T = 5
    hidden_states = torch.randn(T, H)
    positions = torch.arange(T)
    ctx = Context(page_size=1)
    ctx.kv_cache = BaseKVCachePool(config, page_size=1, num_pages=32, device=dev, dtype=torch.float32)
    pt = torch.zeros((2, 32), dtype=torch.int32)
    pt[0, :T] = torch.arange(T)
    ctx.page_table = pt
    ctx.kv_cache.attach_page_table(pt)
    ctx.attn_backend = TritonAttentionBackend(None)
    req = Req(
        input_ids=list(range(T)), table_idx=0, cached_len=0, output_len=1, uid=0,
        sampling_params=SamplingParams(), cache_handle=None,
    )
    req.device_len = T
    batch = Batch(reqs=[req], phase="prefill", positions=positions, extend_lens=[T])
    set_global_ctx(ctx)
    ctx._batch = batch

    out = attn(hidden_states, positions, 0, ctx, batch)
    assert out.shape == (T, H)
    assert torch.isfinite(out).all()

    # Zero every bias term and re-run: a real bias contribution must change
    # the output (proves has_bias isn't silently ignored by the forward).
    with torch.no_grad():
        attn.q_proj.bias.zero_()
        attn.k_proj.bias.zero_()
        attn.v_proj.bias.zero_()
    out_no_bias = attn(hidden_states, positions, 0, ctx, batch)
    assert not torch.allclose(out, out_no_bias)


def test_qwen2moe_router_matches_hand_computed_softmax_topk():
    torch.manual_seed(0)
    T, H, num_experts, top_k = 5, 8, 6, 2
    x = torch.randn(T, H)
    gate_weight = torch.randn(num_experts, H) * 0.1

    weights, experts = qwen2moe_router(x, gate_weight, top_k, norm_topk_prob=False)
    assert weights.shape == (T, top_k)
    assert experts.shape == (T, top_k)

    logits = F.linear(x, gate_weight)
    probs = torch.softmax(logits.float(), dim=-1)
    exp_weights, exp_experts = torch.topk(probs, top_k, dim=-1)
    torch.testing.assert_close(weights.float(), exp_weights, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(experts, exp_experts)


def test_qwen2moe_router_renormalizes_when_norm_topk_prob_true():
    torch.manual_seed(1)
    T, H, num_experts, top_k = 4, 6, 5, 2
    x = torch.randn(T, H)
    gate_weight = torch.randn(num_experts, H) * 0.1

    weights, _ = qwen2moe_router(x, gate_weight, top_k, norm_topk_prob=True)
    row_sums = weights.float().sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones(T), atol=1e-5, rtol=1e-5)


def test_qwen2moe_router_no_renorm_rows_need_not_sum_to_one():
    torch.manual_seed(2)
    T, H, num_experts, top_k = 4, 6, 5, 2
    x = torch.randn(T, H)
    gate_weight = torch.randn(num_experts, H) * 0.1

    weights, _ = qwen2moe_router(x, gate_weight, top_k, norm_topk_prob=False)
    row_sums = weights.float().sum(dim=-1)
    assert not torch.allclose(row_sums, torch.ones(T), atol=1e-3)


def test_qwen2moe_shared_expert_output_matches_hand_computed_sigmoid_gate():
    """Real, gated combination: sigmoid(shared_expert_gate(x)) * shared_expert(x)
    -- a real, different mechanism from deepseek_v4/glm_moe_dsa's unconditional
    add (confirmed against the real HF Qwen2MoeSparseMoeBlock.forward)."""
    torch.manual_seed(3)
    T, H, inter = 4, 8, 12
    x = torch.randn(T, H)
    gate_proj = torch.randn(inter, H) * 0.1
    up_proj = torch.randn(inter, H) * 0.1
    down_proj = torch.randn(H, inter) * 0.1
    shared_gate_weight = torch.randn(1, H) * 0.1

    got = qwen2moe_shared_expert_output(x, gate_proj, up_proj, down_proj, shared_gate_weight)
    assert got.shape == (T, H)

    shared_mlp = F.linear(F.silu(F.linear(x, gate_proj)) * F.linear(x, up_proj), down_proj)
    gate = torch.sigmoid(F.linear(x, shared_gate_weight))
    expected = gate * shared_mlp
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)


def test_qwen2moe_shared_expert_gate_actually_modulates_output():
    """Proof the gate is active: an all-negative gate logit (sigmoid near 0)
    must suppress the shared expert's output near zero."""
    torch.manual_seed(4)
    T, H, inter = 3, 6, 8
    x = torch.rand(T, H) + 0.1  # strictly positive, so a negative-constant gate
    # weight's dot product with x is guaranteed very negative regardless of seed.
    gate_proj = torch.randn(inter, H) * 0.1
    up_proj = torch.randn(inter, H) * 0.1
    down_proj = torch.randn(H, inter) * 0.1
    shared_gate_weight = -50.0 * torch.ones(1, H)

    got = qwen2moe_shared_expert_output(x, gate_proj, up_proj, down_proj, shared_gate_weight)
    assert got.abs().max() < 1e-3
