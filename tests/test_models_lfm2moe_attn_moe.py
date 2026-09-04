"""Unit tests for LFM2-MoE's GQA attention + bias-corrected MoE router/experts
(issue `models-lfm2moe-attn-moe`, #231). Standalone -- not yet wired into a
decoder layer or the engine (that's #232's job); these pin the attention and
router/expert math against independently hand-computed references.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from freetoken.attention.triton import TritonAttentionBackend
from freetoken.core import Batch, Context, Req, SamplingParams, reset_global_ctx, set_global_ctx
from freetoken.kvcache.base import BaseKVCachePool
from freetoken.models.config import ModelConfig
from freetoken.models.lfm2_moe import Lfm2MoeAttention, Lfm2MoeExperts, Lfm2MoeSparseMoeBlock, Lfm2MoeTopKRouter


@pytest.fixture(autouse=True)
def _clean_global_ctx():
    yield
    reset_global_ctx()


def _rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def test_lfm2moe_attention_forward_runs_and_qk_norm_is_real():
    """Real forward pass through a Context/BaseKVCachePool (identity page
    table -- the established pattern for a standalone-primitive test, per
    #231's own scope; the real MHAKVCache validation is #232's e2e job).
    Confirms finite output, and that q/k RMSNorm is actually wired (the
    issue text wrongly assumed no QK-norm exists -- the real modeling code
    has it unconditionally; zeroing the norm weight must change output)."""
    torch.manual_seed(0)
    H, NH, NKV, D = 16, 4, 2, 8
    config = ModelConfig(
        hidden_size=H,
        num_attention_heads=NH,
        num_key_value_heads=NKV,
        head_dim=D,
        num_layers=1,
        rope_theta=10000.0,
        attrs={"norm_eps": 1e-5},
    )
    dev = torch.device("cpu")
    attn = Lfm2MoeAttention(config, dev, torch.float32, layer_id=0)
    with torch.no_grad():
        for p in attn.parameters():
            p.normal_(0, 0.02)
        attn.q_layernorm.weight.fill_(1.0)
        attn.k_layernorm.weight.fill_(1.0)

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

    # A real (non-identity) norm weight must change the output vs identity
    # scale -- proves q_layernorm/k_layernorm are load-bearing, not a no-op.
    with torch.no_grad():
        attn.q_layernorm.weight.normal_(0, 1.0)
        attn.k_layernorm.weight.normal_(0, 1.0)
    out_diff_norm = attn(hidden_states, positions, 0, ctx, batch)
    assert not torch.allclose(out, out_diff_norm)


def test_lfm2moe_attention_rope_matches_hand_computed_rotate_half():
    """Independent recomputation of the half-split RoPE applied to q, isolated
    from attention/KV-cache plumbing -- pins the exact rotation formula."""
    torch.manual_seed(1)
    H, NH, NKV, D = 8, 2, 2, 4
    config = ModelConfig(
        hidden_size=H, num_attention_heads=NH, num_key_value_heads=NKV, head_dim=D,
        num_layers=1, rope_theta=10000.0, attrs={"norm_eps": 1e-5},
    )
    attn = Lfm2MoeAttention(config, torch.device("cpu"), torch.float32, layer_id=0)

    T = 6
    x = torch.randn(NH, T, D)  # head-major, matching attn._rope's expected layout
    positions = torch.arange(T)

    got = attn._rope(x, positions)

    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    freqs = torch.outer(positions.float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos()[None, :, :], emb.sin()[None, :, :]
    expected = x * cos + _rotate_half(x) * sin
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_lfm2moe_router_matches_hand_computed_sigmoid_topk_no_bias():
    """No expert bias: flat top-k directly on sigmoid scores (no grouping --
    LFM2 has no n_group/topk_group at all, unlike DeepSeek-V3/V4)."""
    torch.manual_seed(2)
    T, H, E, K = 5, 8, 6, 2
    config = ModelConfig(
        hidden_size=H, num_experts=E, num_experts_per_tok=K,
        attrs={"norm_topk_prob": False, "routed_scaling_factor": 1.0, "use_expert_bias": False},
    )
    router = Lfm2MoeTopKRouter(config, torch.float32)
    with torch.no_grad():
        router.weight.normal_(0, 0.1)
    assert router.expert_bias is None

    x = torch.randn(T, H)
    selected, weights = router(x)
    assert selected.shape == (T, K)
    assert weights.shape == (T, K)

    logits = F.linear(x, router.weight)
    scores = logits.sigmoid()
    exp_weights, exp_selected = torch.topk(scores, K, dim=-1)
    torch.testing.assert_close(selected, exp_selected)
    torch.testing.assert_close(weights, exp_weights, atol=1e-6, rtol=1e-6)


def test_lfm2moe_router_bias_affects_selection_but_not_gathered_weight_value():
    """With use_expert_bias=True: bias shifts WHICH experts get selected, but
    the returned weight for a selected expert is its RAW (uncorrected)
    sigmoid score -- confirmed against the real HF Lfm2MoeTopKRouter.forward,
    same shape as DeepSeek-V3/V4's own e_score_correction_bias router."""
    torch.manual_seed(3)
    T, H, E, K = 1, 4, 4, 1
    config = ModelConfig(
        hidden_size=H, num_experts=E, num_experts_per_tok=K,
        attrs={"norm_topk_prob": False, "routed_scaling_factor": 1.0, "use_expert_bias": True},
    )
    router = Lfm2MoeTopKRouter(config, torch.float32)
    assert router.expert_bias is not None
    with torch.no_grad():
        router.weight.zero_()  # every expert's raw logit -> sigmoid(0) = 0.5
        # Bias expert 2 heavily so it wins selection despite tied raw scores.
        router.expert_bias.copy_(torch.tensor([0.0, 0.0, 10.0, 0.0]))

    x = torch.randn(T, H)
    selected, weights = router(x)
    assert selected.item() == 2
    # The gathered weight is the RAW sigmoid score (0.5), not score+bias.
    torch.testing.assert_close(weights, torch.full((T, K), 0.5))


def test_lfm2moe_router_renormalizes_when_norm_topk_prob_true():
    torch.manual_seed(4)
    T, H, E, K = 4, 6, 5, 2
    config = ModelConfig(
        hidden_size=H, num_experts=E, num_experts_per_tok=K,
        attrs={"norm_topk_prob": True, "routed_scaling_factor": 1.0, "use_expert_bias": False},
    )
    router = Lfm2MoeTopKRouter(config, torch.float32)
    with torch.no_grad():
        router.weight.normal_(0, 0.1)
    _, weights = router(torch.randn(T, H))
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(T), atol=1e-5, rtol=1e-5)


def test_lfm2moe_experts_matches_hand_computed_weighted_swiglu_sum():
    """Independent per-token recomputation of the selected experts' SwiGLU
    output, weighted and summed -- the one-hot/index_add dispatch inside
    Lfm2MoeExperts must produce the same result as a naive per-token loop."""
    torch.manual_seed(5)
    T, H, E, K, inter = 6, 8, 5, 2, 12
    config = ModelConfig(hidden_size=H, num_experts=E, moe_intermediate_size=inter)
    experts = Lfm2MoeExperts(config, torch.float32)
    with torch.no_grad():
        experts.gate_up_proj.normal_(0, 0.1)
        experts.down_proj.normal_(0, 0.1)

    x = torch.randn(T, H)
    selected = torch.randint(0, E, (T, K))
    weights = torch.rand(T, K)

    got = experts(x, selected, weights)
    assert got.shape == (T, H)

    expected = torch.zeros(T, H)
    for t in range(T):
        for k in range(K):
            e = selected[t, k].item()
            gate, up = F.linear(x[t], experts.gate_up_proj[e]).chunk(2, dim=-1)
            h = F.silu(gate) * up
            h = F.linear(h, experts.down_proj[e])
            expected[t] += h * weights[t, k]
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_lfm2moe_sparse_moe_block_end_to_end_finite():
    torch.manual_seed(6)
    T, H, E, K, inter = 4, 8, 6, 2, 16
    config = ModelConfig(
        hidden_size=H, num_experts=E, num_experts_per_tok=K, moe_intermediate_size=inter,
        attrs={"norm_topk_prob": True, "routed_scaling_factor": 1.0, "use_expert_bias": True},
    )
    block = Lfm2MoeSparseMoeBlock(config, torch.float32)
    with torch.no_grad():
        for p in block.parameters():
            p.normal_(0, 0.1)
    out = block(torch.randn(T, H))
    assert out.shape == (T, H)
    assert torch.isfinite(out).all()
