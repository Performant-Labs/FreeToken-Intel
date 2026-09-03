"""gptq_int4 is excluded from the hybrid CPU/PCIe split (issue
moe-quant-banks-compute, #137): _cpu_subset_math hardcodes the "bf16" bank
names (model.moe_cache.bank_sources["gate_up"]/["down"]) and would KeyError
against a "gptq_int4" cache's differently-named banks. _forward_hybrid now
forces fetch_frac to 1.0 (pure offload, no CPU split) whenever the cache's
quant_format is "gptq_int4", documented as a deliberate first-cut tradeoff
rather than teaching the CPU half to dequantize too.

Tests _Qwen3MoE._forward_hybrid (qwen3_moe) directly via a stand-in ``self``
(unittest.mock), not a real model/engine -- isolates the routing decision
(does it call _forward_offload or attempt the CPU split?) from real tensor
math, real weights, or the full loader/engine stack. CPU-only, no XPU.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

from freetoken.models.qwen3_moe import _Qwen3MoE


def _fake_self(*, cache_quant_format: str, fetch_fraction: float = 0.5):
    """A MagicMock standing in for `self` in _Qwen3MoE._forward_hybrid,
    with just enough surface for the method body to run without touching
    real tensors: _is_cpu_layer -> False (not a whole-layer CPU carve-out),
    _forward_offload / _cpu_subset_math / _hybrid_cpu_pool stubbed."""
    self_ = MagicMock(spec=_Qwen3MoE)
    self_._is_cpu_layer.return_value = False
    self_._forward_offload.return_value = torch.zeros(1)
    model = MagicMock()
    model.moe_cache = MagicMock()
    model.moe_cache.quant_format = cache_quant_format
    model.moe_hybrid_fetch_fraction = fetch_fraction
    model.moe_hybrid_max_fetch = -1
    return self_, model


def test_gptq_int4_never_reaches_cpu_split():
    self_, model = _fake_self(cache_quant_format="gptq_int4", fetch_fraction=0.5)
    top_idx = torch.zeros(1, 1, dtype=torch.long)
    top_w = torch.ones(1, 1)
    flat = torch.zeros(1, 4)

    out = _Qwen3MoE._forward_hybrid(self_, flat, top_idx, top_w, model, batch=None)

    self_._forward_offload.assert_called_once_with(flat, top_idx, top_w, model, None)
    self_._hybrid_cpu_pool.assert_not_called()
    torch.testing.assert_close(out, torch.zeros(1))


def test_bf16_still_splits_normally_when_fetch_fraction_is_mid_range():
    """Sanity check that the new guard does not accidentally change bf16
    behavior: with a mid-range fetch_fraction and quant_format="bf16", the
    method must proceed into the real split logic (calling _forward_offload
    with a non-empty `exclude` set -- the CPU-computed experts), not take
    the fetch_frac<=0 / >=1.0 early-return shortcuts a forced-1.0 gptq_int4
    call would take instead."""
    self_, model = _fake_self(cache_quant_format="bf16", fetch_fraction=0.5)
    top_idx = torch.tensor([[0, 1]])
    top_w = torch.tensor([[0.5, 0.5]])
    flat = torch.zeros(1, 4)

    self_._forward_offload.return_value = torch.zeros(1, 4)
    self_._hybrid_cpu_pool.return_value.submit.return_value.result.return_value = torch.zeros(1, 4)

    _Qwen3MoE._forward_hybrid(self_, flat, top_idx, top_w, model, batch=None)

    self_._forward_offload.assert_called_once()
    _, kwargs = self_._forward_offload.call_args
    assert kwargs.get("exclude"), "expected a real, non-empty CPU-expert split (exclude=...)"
