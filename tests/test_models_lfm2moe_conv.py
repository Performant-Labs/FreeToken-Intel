"""Unit tests for LFM2-MoE's short gated-conv primitive (issue
``models-lfm2moe-conv``, #230). Standalone -- not yet wired into a decoder
layer or the engine (that's #232's job); these pin the conv/gate math
itself against an independently hand-computed reference, and
``parse_config`` against the real checkpoint's own field shapes.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from freetoken.models.lfm2_moe import ShortConv, causal_depthwise_conv1d, parse_config


def _hand_causal_conv(h, weight, bias, kernel_size):
    """Independent re-derivation: for each channel c and position t,
    y[c,t] = bias[c] + sum_{k=0}^{K-1} weight[c,k] * h[c, t-K+1+k] (h[<0]=0)."""
    C, T = h.shape
    out = torch.zeros_like(h)
    for c in range(C):
        for t in range(T):
            acc = bias[c].item() if bias is not None else 0.0
            for k in range(kernel_size):
                src = t - kernel_size + 1 + k
                if src >= 0:
                    acc += (weight[c, k] * h[c, src]).item()
            out[c, t] = acc
    return out


def test_causal_depthwise_conv1d_matches_hand_computed_reference():
    torch.manual_seed(0)
    C, T, K = 4, 6, 3
    h = torch.randn(1, C, T)
    weight = torch.randn(C, K)
    bias = torch.randn(C)
    got = causal_depthwise_conv1d(h, weight, bias, K).squeeze(0)
    expected = _hand_causal_conv(h.squeeze(0), weight, bias, K)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_causal_depthwise_conv1d_no_bias():
    torch.manual_seed(1)
    C, T, K = 3, 5, 2
    h = torch.randn(1, C, T)
    weight = torch.randn(C, K)
    got = causal_depthwise_conv1d(h, weight, None, K).squeeze(0)
    expected = _hand_causal_conv(h.squeeze(0), weight, None, K)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_causal_conv1d_position_zero_only_sees_itself():
    """Position 0 must not read any real (non-zero-padded) prior context --
    the defining property of causal conv."""
    K = 3
    h = torch.zeros(1, 1, 5)
    h[0, 0, 0] = 1.0
    weight = torch.ones(1, K)
    out = causal_depthwise_conv1d(h, weight, None, K).squeeze()
    # y[0] only sums weight[K-1]*h[0] (the rest of the kernel reads left zero-pad)
    assert out[0].item() == pytest.approx(1.0)
    # y[1] sums weight[K-2]*h[0] + weight[K-1]*h[1](=0)
    assert out[1].item() == pytest.approx(1.0)
    # y[K-1] is the first position where the full kernel reads real (or already-seen) data
    assert out[K - 1].item() == pytest.approx(1.0)
    # positions beyond the kernel width no longer see the impulse at 0
    assert out[K].item() == pytest.approx(0.0)


def test_short_conv_module_matches_manual_gate_and_conv_composition():
    torch.manual_seed(2)
    hidden, K, T = 8, 3, 6
    conv = ShortConv(hidden, K, has_bias=True)
    x = torch.randn(T, hidden)

    got = conv(x)
    assert got.shape == (T, hidden)

    # Independent re-derivation of the full forward: in_proj -> gate -> conv -> gate -> out_proj.
    bcx = conv.in_proj(x).transpose(0, 1)
    b, c, gx = bcx.chunk(3, dim=0)
    h = b * gx
    h = _hand_causal_conv(h, conv.conv_weight, conv.conv_bias, K)
    y = c * h
    expected = conv.out_proj(y.transpose(0, 1))
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_short_conv_no_bias_variant():
    torch.manual_seed(3)
    hidden, K, T = 4, 2, 5
    conv = ShortConv(hidden, K, has_bias=False)
    assert conv.conv_bias is None
    x = torch.randn(T, hidden)
    out = conv(x)
    assert out.shape == (T, hidden)
    assert torch.isfinite(out).all()


_REAL_LAYER_TYPES = [
    "conv", "conv", "full_attention", "conv", "conv", "conv", "full_attention",
    "conv", "conv", "conv", "full_attention", "conv", "conv", "conv", "full_attention",
    "conv", "conv", "conv", "full_attention", "conv", "conv", "full_attention", "conv", "conv",
]

_REAL_CONFIG = {
    "architectures": ["Lfm2MoeForCausalLM"],
    "model_type": "lfm2_moe",
    "hidden_size": 2048,
    "vocab_size": 128000,
    "num_hidden_layers": 24,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 7168,
    "moe_intermediate_size": 1792,
    "num_experts": 32,
    "num_experts_per_tok": 4,
    "num_dense_layers": 2,
    "conv_bias": False,
    "conv_L_cache": 3,
    "layer_types": _REAL_LAYER_TYPES,
    "norm_eps": 1e-5,
    "norm_topk_prob": True,
    "use_expert_bias": True,
    "routed_scaling_factor": 1.0,
    "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
    "max_position_embeddings": 128000,
    "tie_word_embeddings": True,
    "dtype": "bfloat16",
}


def _hf(d):
    return type("Hf", (), {"to_dict": lambda self: d})()


def test_parse_config_reads_real_checkpoint_fields_verbatim():
    config = parse_config(_hf(_REAL_CONFIG))
    assert config.hidden_size == 2048
    assert config.num_layers == 24
    assert config.first_k_dense_replace == 2
    assert config.is_moe is True
    assert config.attrs["layer_types"] == _REAL_LAYER_TYPES
    assert config.attrs["conv_L_cache"] == 3
    assert config.attrs["conv_bias"] is False
    assert config.attrs["num_dense_layers"] == 2
    assert config.rope_theta == 1000000.0


def test_parse_config_rejects_missing_layer_types_rather_than_guessing():
    cfg = dict(_REAL_CONFIG)
    del cfg["layer_types"]
    with pytest.raises(ValueError, match="layer_types"):
        parse_config(_hf(cfg))


def test_parse_config_rejects_layer_types_length_mismatch():
    cfg = dict(_REAL_CONFIG)
    cfg["layer_types"] = _REAL_LAYER_TYPES[:-1]  # 23 entries, num_hidden_layers=24
    with pytest.raises(ValueError, match="layer_types"):
        parse_config(_hf(cfg))
