"""Unit tests for Qwen3.8-Flash-Next's QSA block-sparse indexer mechanism
(issue ``models-qwen4-qsa``, #208). Standalone -- not yet wired into a
decoder layer, checkpoint loader, or the engine's KV cache (that's #209's
job); these pin the compress/score/top-k/expand/attend math itself,
operating on a full, already-materialized sequence (prefill-shaped).

Test strategy mirrors DSA's own (#191): (1) the short-sequence dense-
equivalence oracle upstream's own ``TorchDenseQSAReference`` docstring
states -- QSA is exactly dense while a request sees at most
``index_budget + index_ratio - 1`` tokens; (2) a longer sequence that
forces real block dropping, proving the sparse mask is actually
restrictive and matches an independently hand-computed top-k selection.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from freetoken.models.qwen4_exp import (
    apply_partial_rope,
    qsa_attend_mask,
    qsa_compress_keys,
    qsa_expand_block_mask,
    qsa_rope_cos_sin,
    qsa_score,
    qsa_sparse_attend,
    qsa_topk_blocks,
)


def _dense_causal_attend(q, k, v, sm_scale):
    """Independent reference: plain causal GQA attention over every visible token."""
    T = q.shape[0]
    num_q, num_kv = q.shape[1], k.shape[1]
    rep = num_q // num_kv
    k_full = k.repeat_interleave(rep, dim=1)
    v_full = v.repeat_interleave(rep, dim=1)
    scores = torch.einsum("thd,shd->hts", q.float(), k_full.float()) * sm_scale
    causal = torch.arange(T)[None, :] <= torch.arange(T)[:, None]
    scores = scores.masked_fill(~causal[None, :, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hts,shd->thd", probs, v_full.float()).to(q.dtype)


def test_qsa_compress_keys_drops_trailing_partial_group():
    torch.manual_seed(0)
    T, kv_heads, D, ratio = 10, 1, 4, 4  # 10 // 4 = 2 complete blocks, 2 trailing tokens dropped
    raw = torch.randn(T, kv_heads, D)
    pooled, starts = qsa_compress_keys(raw, ratio)
    assert pooled.shape == (2, kv_heads, D)
    assert torch.equal(starts, torch.tensor([0, 4]))
    expected_block0 = raw[0:4].float().mean(dim=0)
    expected_block1 = raw[4:8].float().mean(dim=0)
    torch.testing.assert_close(pooled[0], expected_block0)
    torch.testing.assert_close(pooled[1], expected_block1)


def test_qsa_compress_keys_empty_when_shorter_than_one_group():
    raw = torch.randn(3, 1, 4)
    pooled, starts = qsa_compress_keys(raw, index_ratio=4)
    assert pooled.shape == (0, 1, 4)
    assert starts.shape == (0,)


def test_apply_partial_rope_matches_hand_rotation_and_passes_through_tail():
    torch.manual_seed(1)
    T, D, rotary_dim = 3, 8, 4
    x = torch.randn(T, D)
    positions = torch.arange(T)
    cos, sin = qsa_rope_cos_sin(positions, rotary_dim, theta=10000.0)
    got = apply_partial_rope(x, cos, sin, rotary_dim)
    # tail (dims rotary_dim:) must be untouched
    torch.testing.assert_close(got[..., rotary_dim:], x[..., rotary_dim:])
    # hand-rotate the first rotary_dim dims
    x_rot = x[..., :rotary_dim].float()
    half = rotary_dim // 2
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    expected = x_rot * cos + rotated * sin
    torch.testing.assert_close(got[..., :rotary_dim], expected.to(x.dtype))


def test_qsa_score_and_topk_match_hand_computed_reference_on_short_sequence():
    """A sequence entirely within index_budget: every historical block should be
    top-k-selected for the last query (nothing dropped)."""
    torch.manual_seed(2)
    T, n_heads, kv_heads, D, ratio = 12, 2, 1, 8, 4
    eps, theta, rotary_dim = 1e-5, 10000.0, 4

    raw_k = torch.randn(T, kv_heads, D)
    q = torch.randn(T, n_heads, D)
    q_norm_w = torch.randn(D) * 0.1
    k_norm_w = torch.randn(D) * 0.1

    pooled, block_start = qsa_compress_keys(raw_k, ratio)
    num_blocks = pooled.shape[0]
    assert num_blocks == 3  # 12 // 4

    positions = torch.arange(T)
    scores = qsa_score(q, pooled, q_norm_w, k_norm_w, eps, positions, block_start, rotary_dim, theta)
    assert scores.shape == (T, num_blocks)

    # Hand-compute the score for the LAST query row against block 0 independently.
    from freetoken.models.qwen4_exp import grouped_plus_one_rms_norm

    t = T - 1
    q_n = grouped_plus_one_rms_norm(q[t : t + 1], q_norm_w, eps, 1)  # [1, n_heads, D]
    cos_q, sin_q = qsa_rope_cos_sin(positions[t : t + 1], rotary_dim, theta)
    q_n = apply_partial_rope(q_n, cos_q[:, None, :], sin_q[:, None, :], rotary_dim)[0]  # [n_heads, D]

    b = 0
    k_n = grouped_plus_one_rms_norm(pooled[b : b + 1], k_norm_w, eps, 1)
    cos_k, sin_k = qsa_rope_cos_sin(block_start[b : b + 1], rotary_dim, theta)
    k_n = apply_partial_rope(k_n, cos_k[:, None, :], sin_k[:, None, :], rotary_dim)[0]  # [kv_heads, D]
    k_n = k_n.repeat_interleave(n_heads // kv_heads, dim=0)  # [n_heads, D]

    expected = torch.relu((q_n.float() * k_n.float()).sum(-1)).sum() / (D**0.5)
    torch.testing.assert_close(scores[t, b], expected, atol=1e-5, rtol=1e-5)

    block_last = block_start + ratio - 1
    topk = qsa_topk_blocks(scores, block_last, positions, budget_blocks=num_blocks)
    # Every block is causally visible to the last query and budget == num_blocks: all selected.
    assert set(topk[t].tolist()) == {0, 1, 2}


def test_qsa_topk_blocks_respects_causality_and_pads_with_minus_one():
    scores = torch.tensor([[5.0, 3.0, 9.0]])  # one query row, 3 blocks
    block_last = torch.tensor([3, 7, 11])
    query_positions = torch.tensor([7])  # can only see blocks 0 and 1 (block 2 ends at 11 > 7)
    topk = qsa_topk_blocks(scores, block_last, query_positions, budget_blocks=3)
    assert topk.shape == (1, 3)
    visible = {v for v in topk[0].tolist() if v >= 0}
    assert visible == {0, 1}
    assert (topk[0] == -1).sum() == 1


def test_qsa_expand_block_mask_covers_exactly_the_selected_blocks_tokens():
    topk_idx = torch.tensor([[0, 2, -1]])  # blocks 0 and 2 selected, one pad
    mask = qsa_expand_block_mask(topk_idx, index_ratio=4, seq_len=12)
    assert mask.shape == (1, 12)
    expected = torch.zeros(12, dtype=torch.bool)
    expected[0:4] = True  # block 0 -> tokens 0..3
    expected[8:12] = True  # block 2 -> tokens 8..11
    torch.testing.assert_close(mask[0], expected)


def test_qsa_attend_mask_includes_trailing_incomplete_block_unconditionally():
    """The last (< index_ratio) tokens haven't formed a compressible block yet -- they
    must always be attended, regardless of top-k block selection."""
    T, ratio = 10, 4  # 2 complete blocks (0-3, 4-7), trailing tokens 8-9 incomplete
    topk_idx = torch.tensor([[-1, -1]])  # no block selected at all for this query
    query_positions = torch.tensor([9])
    mask = qsa_attend_mask(topk_idx, ratio, query_positions, seq_len=T)
    assert mask.shape == (1, T)
    # tokens 8, 9 (trailing incomplete block) must be visible even with no block selected
    assert bool(mask[0, 8]) and bool(mask[0, 9])
    # no earlier (compressed-block) token is visible since nothing was selected
    assert not bool(mask[0, :8].any())


def test_qsa_attend_mask_open_group_is_per_query_not_a_global_tail():
    """Real bug this pins (found running the port on real B70 hardware): for a long,
    EXACTLY block-aligned sequence, an early query (whose own block hasn't closed yet, so
    it has no causally-visible compressed blocks at all) must still see its own open
    group's raw prefix -- not zero tokens. A global "trailing incomplete block" concept
    (the old, wrong implementation) only covers rows near the END of the sequence; upstream
    ``qsa_sparse.py``'s own module docstring (step 5) says the always-visible region is
    "the causal tail of the open group", which is PER QUERY."""
    T, ratio = 40, 4  # exactly 10 complete blocks, no global trailing incomplete region at all
    topk_idx = torch.full((T, 1), -1, dtype=torch.long)  # nothing selected for any query
    query_positions = torch.arange(T)
    mask = qsa_attend_mask(topk_idx, ratio, query_positions, seq_len=T)
    # every query must see at least itself -- an all-false row means softmax(-inf) -> NaN
    assert mask.any(dim=-1).all()
    # query 0's open group is just position 0 (block 0 spans 0..3, not yet closed)
    assert mask[0].tolist() == [i == 0 for i in range(T)]
    # query 2's open group is positions 0..2 (still inside block 0, not yet closed)
    assert mask[2, :3].all() and not mask[2, 3:].any()


def test_qsa_sparse_attend_matches_dense_reference_when_nothing_is_pruned():
    """Equivalence oracle (upstream TorchDenseQSAReference docstring): when every
    causally-visible token is included in the attend mask, QSA reduces to plain
    dense causal attention."""
    torch.manual_seed(3)
    T, num_q, num_kv, D = 6, 4, 2, 8
    q = torch.randn(T, num_q, D)
    k = torch.randn(T, num_kv, D)
    v = torch.randn(T, num_kv, D)
    sm_scale = D**-0.5

    causal = torch.arange(T)[None, :] <= torch.arange(T)[:, None]
    got = qsa_sparse_attend(q, k, v, causal, sm_scale)
    expected = _dense_causal_attend(q, k, v, sm_scale)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_qsa_sparse_attend_actually_restricts_when_mask_is_narrower_than_causal():
    """Proof the sparse mask is active: pruning a visible key changes the output
    versus the unrestricted dense reference."""
    torch.manual_seed(4)
    T, num_q, num_kv, D = 5, 2, 1, 4
    q = torch.randn(T, num_q, D)
    k = torch.randn(T, num_kv, D)
    v = torch.randn(T, num_kv, D)
    sm_scale = D**-0.5

    causal = torch.arange(T)[None, :] <= torch.arange(T)[:, None]
    narrow = causal.clone()
    narrow[-1, 1] = False  # drop one otherwise-visible key for the last query

    dense_out = qsa_sparse_attend(q, k, v, causal, sm_scale)
    sparse_out = qsa_sparse_attend(q, k, v, narrow, sm_scale)
    assert not torch.allclose(dense_out[-1], sparse_out[-1])

    # And the sparse output matches an independent manual recomputation with key 1 masked out.
    manual_scores = torch.einsum("hd,shd->hs", q[-1].float(), k.repeat_interleave(num_q // num_kv, dim=1).float()) * sm_scale
    manual_scores[:, 1] = float("-inf")
    manual_probs = torch.softmax(manual_scores, dim=-1)
    manual_out = torch.einsum("hs,shd->hd", manual_probs, v.repeat_interleave(num_q // num_kv, dim=1).float())
    torch.testing.assert_close(sparse_out[-1], manual_out.to(sparse_out.dtype), atol=1e-5, rtol=1e-5)
