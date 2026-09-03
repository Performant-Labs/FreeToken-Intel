"""SlotWeightAccessor.expert_forward: the offload forward's mxfp4 path
actually dispatches to the native fused-GEMM kernel on real XPU hardware,
and matches the dequant-then-matmul fallback (issue `moe-quant-banks-
native-multi`, #163).

Reuses test_moe_slot_weight_accessor_mxfp4.py's own fixture pattern
(_make_packed_projection / _cache_with_mxfp4_bank), moved to a real XPU
device -- expert_forward's native branch only engages when
``cache.is_xpu``, so the existing CPU-only accessor tests never exercise
it. ``xpu``-marked: deselected on a torch-free / no-XPU box (see
``conftest.py``).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

XPU = pytest.mark.skipif(not torch.xpu.is_available(), reason="no XPU available")

from freetoken.kernel.triton import fused_mxfp4_linear as fused_mxfp4_linear_mod
from freetoken.kernel.triton.mxfp4_linear import quantize_mxfp4_blocks
from freetoken.moe.offload_cache import OffloadMoeCache, SlotWeightAccessor

DEVICE = torch.device("xpu")


def _make_packed_projection(out_features: int, k: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    dense = (torch.rand(out_features, k, generator=g) - 0.5) * 4.0
    blocks, scales = quantize_mxfp4_blocks(dense)
    return blocks.to(DEVICE), scales.to(DEVICE)


def _cache_with_mxfp4_bank(hidden, inter, *, num_experts, cache_size):
    cache = OffloadMoeCache(1, num_experts, cache_size, DEVICE, quant_format="mxfp4")
    sources = {name: [] for name in ("blocks_gate_up", "scales_gate_up", "blocks_down", "scales_down")}
    for e in range(num_experts):
        gu_blocks, gu_scales = _make_packed_projection(2 * inter, hidden, seed=100 + e)
        dn_blocks, dn_scales = _make_packed_projection(hidden, inter, seed=200 + e)
        for name, t in (
            ("blocks_gate_up", gu_blocks), ("scales_gate_up", gu_scales),
            ("blocks_down", dn_blocks), ("scales_down", dn_scales),
        ):
            sources[name].append(t)
    sources = {name: [torch.stack(v)] for name, v in sources.items()}
    cache.set_bank_sources(sources)
    cache.materialize_layer(0)
    cache.copy_missing()
    return cache


@XPU
@pytest.mark.xpu
def test_expert_forward_dispatches_to_native_fused_path():
    hidden, inter = 64, 32  # multiples of the 32-element MXFP4 block
    cache = _cache_with_mxfp4_bank(hidden, inter, num_experts=2, cache_size=2)
    assert cache.is_xpu
    accessor = SlotWeightAccessor(cache, intermediate=inter, dtype=torch.float32)

    with patch(
        "freetoken.kernel.triton.fused_mxfp4_linear.fused_mxfp4_expert_forward",
        wraps=fused_mxfp4_linear_mod.fused_mxfp4_expert_forward,
    ) as spy:
        for e in range(2):
            slot = int(cache.slot_for_id[0, e].item())
            x = torch.randn(1, hidden, device=DEVICE, dtype=torch.float32)  # M=1: real call-site batch size
            accessor.expert_forward(slot, x)

    assert spy.call_count == 2


@XPU
@pytest.mark.xpu
def test_expert_forward_matches_dequant_fallback():
    hidden, inter = 64, 32
    cache = _cache_with_mxfp4_bank(hidden, inter, num_experts=1, cache_size=1)
    accessor = SlotWeightAccessor(cache, intermediate=inter, dtype=torch.float32)
    slot = int(cache.slot_for_id[0, 0].item())
    x = torch.randn(1, hidden, device=DEVICE, dtype=torch.float32)

    native_out = accessor.expert_forward(slot, x)

    gate_w, up_w, down_w = accessor.get(slot)
    fallback_out = (torch.nn.functional.silu(x @ gate_w.t()) * (x @ up_w.t())) @ down_w.t()

    torch.testing.assert_close(native_out, fallback_out, atol=2e-3, rtol=2e-2)
