"""SlotWeightAccessor.expert_forward: the offload forward's fp8_block path
actually dispatches to the native fused-GEMM kernel on real XPU hardware,
and matches the dequant-then-matmul fallback (issue `moe-quant-banks-
native-multi`, #163).

Reuses test_moe_slot_weight_accessor_fp8.py's own fixture pattern
(_make_packed_projection / _cache_with_fp8_bank), moved to a real XPU
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

from freetoken.kernel.triton import fused_fp8_linear as fused_fp8_linear_mod
from freetoken.kernel.triton.fp8_block_linear import quantize_block_fp8
from freetoken.moe.offload_cache import OffloadMoeCache, SlotWeightAccessor

DEVICE = torch.device("xpu")


def _make_packed_projection(n: int, k: int, block: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    dense = torch.randn(n, k, generator=g)
    w, s = quantize_block_fp8(dense, block=block)
    return w.to(DEVICE), s.to(DEVICE)


def _cache_with_fp8_bank(n_gu, k_gu, n_dn, k_dn, block, *, num_experts, cache_size):
    cache = OffloadMoeCache(1, num_experts, cache_size, DEVICE, quant_format="fp8_block")
    cache.fp8_block_size = block
    sources = {name: [] for name in ("weight_gate_up", "scale_gate_up", "weight_down", "scale_down")}
    for e in range(num_experts):
        gu_w, gu_s = _make_packed_projection(n_gu, k_gu, block, seed=100 + e)
        dn_w, dn_s = _make_packed_projection(n_dn, k_dn, block, seed=200 + e)
        for name, t in (
            ("weight_gate_up", gu_w), ("scale_gate_up", gu_s),
            ("weight_down", dn_w), ("scale_down", dn_s),
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
    hidden, inter, block = 256, 128, 128  # multiples of block, real checkpoint's convention
    cache = _cache_with_fp8_bank(2 * inter, hidden, hidden, inter, block, num_experts=2, cache_size=2)
    assert cache.is_xpu
    accessor = SlotWeightAccessor(cache, intermediate=inter, dtype=torch.float32)

    with patch(
        "freetoken.kernel.triton.fused_fp8_linear.fused_fp8_expert_forward",
        wraps=fused_fp8_linear_mod.fused_fp8_expert_forward,
    ) as spy:
        for e in range(2):
            slot = int(cache.slot_for_id[0, e].item())
            x = torch.randn(1, hidden, device=DEVICE, dtype=torch.float32)  # M=1: real call-site batch size
            accessor.expert_forward(slot, x)

    assert spy.call_count == 2


@XPU
@pytest.mark.xpu
def test_expert_forward_matches_dequant_fallback():
    hidden, inter, block = 256, 128, 128
    cache = _cache_with_fp8_bank(2 * inter, hidden, hidden, inter, block, num_experts=1, cache_size=1)
    accessor = SlotWeightAccessor(cache, intermediate=inter, dtype=torch.float32)
    slot = int(cache.slot_for_id[0, 0].item())
    x = torch.randn(1, hidden, device=DEVICE, dtype=torch.float32)

    native_out = accessor.expert_forward(slot, x)

    gate_w, up_w, down_w = accessor.get(slot)
    fallback_out = (torch.nn.functional.silu(x @ gate_w.t()) * (x @ up_w.t())) @ down_w.t()

    torch.testing.assert_close(native_out, fallback_out, atol=2e-3, rtol=2e-2)


@XPU
@pytest.mark.xpu
def test_expert_forward_falls_back_above_the_measured_crossover():
    hidden, inter, block = 256, 128, 128
    cache = _cache_with_fp8_bank(2 * inter, hidden, hidden, inter, block, num_experts=1, cache_size=1)
    accessor = SlotWeightAccessor(cache, intermediate=inter, dtype=torch.float32)
    slot = int(cache.slot_for_id[0, 0].item())
    x = torch.randn(64, hidden, device=DEVICE, dtype=torch.float32)  # well above _FUSED_MAX_M

    with patch(
        "freetoken.kernel.triton.fused_fp8_linear.fused_fp8_expert_forward",
        wraps=fused_fp8_linear_mod.fused_fp8_expert_forward,
    ) as spy:
        accessor.expert_forward(slot, x)

    assert spy.call_count == 0
