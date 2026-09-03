"""mxfp4 is excluded from the hybrid CPU/PCIe split (issue
moe-quant-banks-mxfp4, #153), mirroring gptq_int4's own exclusion
(test_moe_hybrid_gptq_exclusion.py): _cpu_subset_math hardcodes the "bf16"
bank names (model.moe_cache.bank_sources["gate_up"]/["down"]) and would
KeyError against a "mxfp4" cache's differently-named banks. _forward_hybrid
now forces fetch_frac to 1.0 (pure offload, no CPU split) whenever the
cache's quant_format is "mxfp4" too, documented as the same deliberate
first-cut tradeoff gptq_int4 already carries.

Tests both ``_Qwen3MoE._forward_hybrid`` (qwen3_moe) and
``_Qwen35MoE._forward_hybrid`` (qwen3_5_moe) directly via a stand-in
``self`` (unittest.mock), not a real model/engine -- isolates the routing
decision (does it call _forward_offload or attempt the CPU split?) from real
tensor math, real weights, or the full loader/engine stack. CPU-only, no XPU.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

from freetoken.models.qwen3_5_moe import _Qwen35MoE
from freetoken.models.qwen3_moe import _Qwen3MoE


def _fake_self(cls, *, cache_quant_format: str, fetch_fraction: float = 0.5):
    """A MagicMock standing in for `self`, plus the 4th positional arg
    ``_forward_hybrid`` expects it as. qwen3_moe's ``_forward_hybrid`` reads
    its bank-cache/fetch-fraction state straight off that 4th arg (named
    ``model`` there); qwen3_5_moe's reads it off ``ctx.model`` instead (its
    4th arg is named ``ctx`` and the method's first line is ``model =
    ctx.model``) -- so for ``_Qwen35MoE`` the 4th arg must be a ctx-shaped
    mock whose ``.model`` is the configured model mock, not the model mock
    itself (an unset ``ctx.model`` would be a *fresh* auto-mock with none of
    the configured attributes, defeating the whole test)."""
    self_ = MagicMock(spec=cls)
    self_._is_cpu_layer.return_value = False
    self_._forward_offload.return_value = torch.zeros(1)
    model = MagicMock()
    model.moe_cache = MagicMock()
    model.moe_cache.quant_format = cache_quant_format
    model.moe_hybrid_fetch_fraction = fetch_fraction
    model.moe_hybrid_max_fetch = -1
    if cls is _Qwen35MoE:
        ctx = MagicMock()
        ctx.model = model
        return self_, ctx
    return self_, model


@pytest.mark.parametrize("cls", [_Qwen3MoE, _Qwen35MoE])
def test_mxfp4_never_reaches_cpu_split(cls):
    self_, ctx = _fake_self(cls, cache_quant_format="mxfp4", fetch_fraction=0.5)
    top_idx = torch.zeros(1, 1, dtype=torch.long)
    top_w = torch.ones(1, 1)
    flat = torch.zeros(1, 4)

    out = cls._forward_hybrid(self_, flat, top_idx, top_w, ctx, batch=None)

    self_._forward_offload.assert_called_once_with(flat, top_idx, top_w, ctx, None)
    self_._hybrid_cpu_pool.assert_not_called()
    torch.testing.assert_close(out, torch.zeros(1))


@pytest.mark.parametrize("cls", [_Qwen3MoE, _Qwen35MoE])
def test_bf16_still_splits_normally_when_fetch_fraction_is_mid_range(cls):
    """Sanity check that the new guard does not accidentally change bf16
    behavior. qwen3_moe's split calls ``_forward_offload(..., exclude=...)``
    directly; qwen3_5_moe's overlapped split calls
    ``_forward_offload_core(..., exclude=...)`` instead (see
    ``_Qwen35MoE._forward_hybrid``'s own docstring) -- both are checked
    generically via whichever of the two the class actually defines."""
    self_, ctx = _fake_self(cls, cache_quant_format="bf16", fetch_fraction=0.5)
    top_idx = torch.tensor([[0, 1]])
    top_w = torch.tensor([[0.5, 0.5]])
    flat = torch.zeros(1, 4)

    offload_mock = self_._forward_offload_core if cls is _Qwen35MoE else self_._forward_offload
    offload_mock.return_value = torch.zeros(1, 4)
    self_._hybrid_cpu_pool.return_value.submit.return_value.result.return_value = torch.zeros(1, 4)

    cls._forward_hybrid(self_, flat, top_idx, top_w, ctx, batch=None)

    offload_mock.assert_called_once()
    _, kwargs = offload_mock.call_args
    assert kwargs.get("exclude"), "expected a real, non-empty CPU-expert split (exclude=...)"
