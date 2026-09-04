"""Unit tests for Qwen3.8-Flash-Next's hyper-connection primitive (issue
``models-qwen4-hc``, #206). Standalone -- not yet wired into a decoder
layer or the engine (that's #209's job); these pin the ``mix``/``combine``
math itself against an independently hand-computed reference.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from freetoken.models.qwen4_exp import GatedResidual, GroupedPlusOneRMSNorm, grouped_plus_one_rms_norm


def test_grouped_plus_one_rms_norm_matches_hand_computed_reference():
    torch.manual_seed(0)
    num_groups, group_size = 2, 3
    width = num_groups * group_size
    x = torch.randn(4, width)
    weight = torch.randn(width) * 0.1
    eps = 1e-5

    got = grouped_plus_one_rms_norm(x, weight, eps, num_groups)

    # Independent per-row, per-group recomputation.
    expected = torch.empty_like(x)
    for t in range(x.shape[0]):
        for g in range(num_groups):
            sl = slice(g * group_size, (g + 1) * group_size)
            chunk = x[t, sl].float()
            rms = torch.sqrt(chunk.pow(2).mean() + eps)
            expected[t, sl] = (chunk / rms) * (1.0 + weight[sl].float())
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_grouped_plus_one_rms_norm_module_wraps_the_function():
    norm = GroupedPlusOneRMSNorm(6, 1e-5, 2)
    torch.manual_seed(1)
    norm.weight.data = torch.randn(6) * 0.1
    x = torch.randn(3, 6)
    torch.testing.assert_close(norm(x), grouped_plus_one_rms_norm(x, norm.weight, 1e-5, 2))


def _hand_computed_mix(R, hc_norm_weight, down_weight, up_weight, hc_count, hidden, lowrank, eps):
    """Independent re-derivation of GatedResidual.mix from the module docstring's own formulas."""
    rn = grouped_plus_one_rms_norm(R, hc_norm_weight, eps, hc_count)
    down = rn @ down_weight.t()
    lora, s = down[..., :lowrank], down[..., lowrank : lowrank + hc_count]
    lora = torch.nn.functional.silu(lora.float() / hc_count)
    gate = lora.to(R.dtype) @ up_weight.t()
    mixed = torch.sigmoid(gate.float()).unflatten(-1, (hc_count, hidden))
    mixed = mixed * rn.float().unflatten(-1, (hc_count, hidden))
    x = mixed.mean(-2).to(R.dtype)
    return x, s


def _hand_computed_combine(R, y, s, hc_count, hidden):
    inject = 2.0 * torch.sigmoid(s.float() / hc_count)
    out = R.float().unflatten(-1, (hc_count, hidden))
    out = out + y.float().unsqueeze(-2) * inject.unsqueeze(-1)
    return out.flatten(-2).to(R.dtype)


def test_gated_residual_mix_and_combine_match_hand_computed_reference():
    torch.manual_seed(2)
    hidden, hc_count, lowrank, eps = 4, 3, 5, 1e-5
    T = 6
    width = hc_count * hidden

    gr = GatedResidual(hidden, hc_count, lowrank, eps, use_combine=True)
    R = torch.randn(T, width)

    x, s = gr.mix(R)
    assert x.shape == (T, hidden)
    assert s.shape == (T, hc_count)

    exp_x, exp_s = _hand_computed_mix(
        R,
        gr.hc_norm.weight,
        gr.input_mix_weight_down_block_inject.weight[: lowrank + hc_count],
        gr.input_mix_weight_up.weight,
        hc_count,
        hidden,
        lowrank,
        eps,
    )
    torch.testing.assert_close(x, exp_x, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(s, exp_s, atol=1e-5, rtol=1e-5)

    y = torch.randn(T, hidden)
    R2 = gr.combine(R, y, s)
    assert R2.shape == (T, width)
    exp_R2 = _hand_computed_combine(R, y, s, hc_count, hidden)
    torch.testing.assert_close(R2, exp_R2, atol=1e-5, rtol=1e-5)


def test_gated_residual_without_combine_has_no_inject_logits_and_no_combine_weight():
    hidden, hc_count, lowrank, eps = 4, 2, 3, 1e-5
    gr = GatedResidual(hidden, hc_count, lowrank, eps, use_combine=False)
    assert not hasattr(gr, "input_mix_weight_down_block_inject")
    assert hasattr(gr, "input_mix_weight_down")

    R = torch.randn(5, hc_count * hidden)
    x, s = gr.mix(R)
    assert x.shape == (5, hidden)
    assert s is None


def test_gated_residual_pad_size_rounds_merged_gemm_rows_to_16():
    # lowrank + hc_count = 320 + 4 = 324 -> next multiple of 16 is 336 -> pad 12
    # (the real Qwen3.8-Flash-Next geometry from the merged-GEMM docstring).
    gr = GatedResidual(hidden_size=8, hc_count=4, lowrank=320, eps=1e-5, use_combine=True)
    assert gr.pad_size == 12
    assert gr.input_mix_weight_down_block_inject.out_features == 320 + 4 + 12


def test_gated_residual_combine_pads_are_dropped_not_read():
    """The pad columns of the merged GEMM are sliced off in `_down` and never
    reach `combine`'s inject math -- s always has exactly hc_count columns."""
    hidden, hc_count, lowrank, eps = 4, 4, 320, 1e-5  # pad_size=12, verified above
    gr = GatedResidual(hidden, hc_count, lowrank, eps, use_combine=True)
    R = torch.randn(2, hc_count * hidden)
    _, s = gr.mix(R)
    assert s.shape == (2, hc_count)


def test_gated_residual_is_batch_shape_agnostic():
    """R/x/s carry arbitrary leading dims (unflatten(-1, ...) on the last axis
    only) -- confirms a [B, T, ...]-shaped caller works, not just [T, ...]."""
    hidden, hc_count, lowrank, eps = 4, 2, 3, 1e-5
    gr = GatedResidual(hidden, hc_count, lowrank, eps, use_combine=True)
    R = torch.randn(2, 5, hc_count * hidden)
    x, s = gr.mix(R)
    assert x.shape == (2, 5, hidden)
    assert s.shape == (2, 5, hc_count)
    y = torch.randn(2, 5, hidden)
    R2 = gr.combine(R, y, s)
    assert R2.shape == R.shape
