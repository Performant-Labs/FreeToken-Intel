"""int8_channel is excluded from the hybrid CPU/PCIe split (issue
`moe-quant-banks-int8`, #154), the same way gptq_int4 is (issue #137,
tested in test_moe_hybrid_gptq_exclusion.py): _cpu_subset_math hardcodes the
"bf16" bank names (model.moe_cache.bank_sources["gate_up"]/["down"]) and
would KeyError against an "int8_channel" cache's differently-named banks.
_forward_hybrid forces fetch_frac to 1.0 (pure offload, no CPU split)
whenever the cache's quant_format is "int8_channel" too.

Tests both _Qwen3MoE._forward_hybrid (qwen3_moe, takes ``model`` directly)
and _Qwen35MoE._forward_hybrid (qwen3_5_moe, takes a ``ctx`` whose
``.model`` is the model) via a stand-in ``self`` (unittest.mock), not a
real model/engine -- isolates the routing decision from real tensor math,
real weights, or the full loader/engine stack. CPU-only, no XPU.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

from freetoken.models.qwen3_5_moe import _Qwen35MoE
from freetoken.models.qwen3_moe import _Qwen3MoE


def _fake_model(*, cache_quant_format: str, fetch_fraction: float):
    model = MagicMock()
    model.moe_cache = MagicMock()
    model.moe_cache.quant_format = cache_quant_format
    model.moe_hybrid_fetch_fraction = fetch_fraction
    model.moe_hybrid_max_fetch = -1
    return model


def _fake_self_qwen3moe(*, cache_quant_format: str, fetch_fraction: float = 0.5):
    """(self, model) -- qwen3_moe's _forward_hybrid takes model directly."""
    self_ = MagicMock(spec=_Qwen3MoE)
    self_._is_cpu_layer.return_value = False
    self_._forward_offload.return_value = torch.zeros(1)
    model = _fake_model(cache_quant_format=cache_quant_format, fetch_fraction=fetch_fraction)
    return self_, model


def _fake_self_qwen35moe(*, cache_quant_format: str, fetch_fraction: float = 0.5):
    """(self, ctx) -- qwen3_5_moe's _forward_hybrid reads model = ctx.model.
    Unlike qwen3_moe, the bf16 split path here calls _forward_offload_core
    (not _forward_offload) for its XPU half -- see the real body."""
    self_ = MagicMock(spec=_Qwen35MoE)
    self_._is_cpu_layer.return_value = False
    self_._forward_offload.return_value = torch.zeros(1)
    model = _fake_model(cache_quant_format=cache_quant_format, fetch_fraction=fetch_fraction)
    ctx = MagicMock()
    ctx.model = model
    return self_, ctx, model


def test_qwen3moe_int8_channel_never_reaches_cpu_split():
    self_, model = _fake_self_qwen3moe(cache_quant_format="int8_channel", fetch_fraction=0.5)
    top_idx = torch.zeros(1, 1, dtype=torch.long)
    top_w = torch.ones(1, 1)
    flat = torch.zeros(1, 4)

    out = _Qwen3MoE._forward_hybrid(self_, flat, top_idx, top_w, model, batch=None)

    self_._forward_offload.assert_called_once_with(flat, top_idx, top_w, model, None)
    self_._hybrid_cpu_pool.assert_not_called()
    torch.testing.assert_close(out, torch.zeros(1))


def test_qwen3moe_bf16_still_splits_normally_when_fetch_fraction_is_mid_range():
    self_, model = _fake_self_qwen3moe(cache_quant_format="bf16", fetch_fraction=0.5)
    top_idx = torch.tensor([[0, 1]])
    top_w = torch.tensor([[0.5, 0.5]])
    flat = torch.zeros(1, 4)

    self_._forward_offload.return_value = torch.zeros(1, 4)
    self_._hybrid_cpu_pool.return_value.submit.return_value.result.return_value = torch.zeros(1, 4)

    _Qwen3MoE._forward_hybrid(self_, flat, top_idx, top_w, model, batch=None)

    self_._forward_offload.assert_called_once()
    _, kwargs = self_._forward_offload.call_args
    assert kwargs.get("exclude"), "expected a real, non-empty CPU-expert split (exclude=...)"


def test_qwen35moe_int8_channel_never_reaches_cpu_split():
    self_, ctx, model = _fake_self_qwen35moe(cache_quant_format="int8_channel", fetch_fraction=0.5)
    top_idx = torch.zeros(1, 1, dtype=torch.long)
    top_w = torch.ones(1, 1)
    flat = torch.zeros(1, 4)

    out = _Qwen35MoE._forward_hybrid(self_, flat, top_idx, top_w, ctx, batch=None)

    self_._forward_offload.assert_called_once_with(flat, top_idx, top_w, ctx, None)
    self_._hybrid_cpu_pool.assert_not_called()
    torch.testing.assert_close(out, torch.zeros(1))


def test_qwen35moe_bf16_still_splits_normally_when_fetch_fraction_is_mid_range():
    """Sanity check that the new guard does not accidentally change bf16
    behavior. qwen3_5_moe's split path calls _forward_offload_core (not
    _forward_offload) for its XPU half -- see _fake_self_qwen35moe."""
    self_, ctx, model = _fake_self_qwen35moe(cache_quant_format="bf16", fetch_fraction=0.5)
    top_idx = torch.tensor([[0, 1]])
    top_w = torch.tensor([[0.5, 0.5]])
    flat = torch.zeros(1, 4)

    self_._forward_offload_core.return_value = torch.zeros(1, 4)
    self_._hybrid_cpu_pool.return_value.submit.return_value.result.return_value = torch.zeros(1, 4)

    _Qwen35MoE._forward_hybrid(self_, flat, top_idx, top_w, ctx, batch=None)

    self_._forward_offload_core.assert_called_once()
    _, kwargs = self_._forward_offload_core.call_args
    assert kwargs.get("exclude"), "expected a real, non-empty CPU-expert split (exclude=...)"
