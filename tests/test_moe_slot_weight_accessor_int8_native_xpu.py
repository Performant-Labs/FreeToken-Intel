"""SlotWeightAccessor.expert_forward: the offload forward's int8_channel
path actually dispatches to the native fused-GEMM kernel on real XPU
hardware, and matches the dequant-then-matmul fallback (issue `moe-quant-
banks-native-multi`, #163).

Reuses test_moe_slot_weight_accessor_int8.py's own fixture pattern
(_pack_int8 / _make_packed_projection / _cache_with_int8_bank), moved to a
real XPU device -- expert_forward's native branch only engages when
``cache.is_xpu``, so the existing CPU-only accessor tests never exercise
it. ``xpu``-marked: deselected on a torch-free / no-XPU box (see
``conftest.py``).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

XPU = pytest.mark.skipif(not torch.xpu.is_available(), reason="no XPU available")

from freetoken.kernel.triton import fused_int8_linear as fused_int8_linear_mod
from freetoken.moe.offload_cache import OffloadMoeCache, SlotWeightAccessor

DEVICE = torch.device("xpu")
GROUP_SIZE = 8  # >= tl.dot's own hardware minimum contraction dim (see fused_int8_linear.py)


def _pack_int8(codes: torch.Tensor) -> torch.Tensor:
    n, k = codes.shape
    codes64 = (codes.to(torch.int64) + 128).reshape(n, k // 4, 4)
    word = sum(codes64[:, :, i] << (8 * i) for i in range(4))
    word = torch.where(word >= (1 << 31), word - (1 << 32), word)
    return word.to(torch.int32)


def _make_packed_projection(n: int, k: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(-127, 128, (n, k), generator=g, dtype=torch.int8)
    num_groups = k // GROUP_SIZE
    scale = torch.rand(n, num_groups, generator=g) * 0.1 + 0.01
    return _pack_int8(codes).to(DEVICE), scale.to(DEVICE)


def _cache_with_int8_bank(k_gu, n_gu, k_dn, n_dn, *, num_experts, cache_size):
    cache = OffloadMoeCache(1, num_experts, cache_size, DEVICE, quant_format="int8_channel")
    sources = {name: [] for name in ("weight_packed_gate_up", "weight_scale_gate_up", "weight_packed_down", "weight_scale_down")}
    for e in range(num_experts):
        gu_w, gu_s = _make_packed_projection(n_gu, k_gu, seed=100 + e)
        dn_w, dn_s = _make_packed_projection(n_dn, k_dn, seed=200 + e)
        for name, t in (
            ("weight_packed_gate_up", gu_w), ("weight_scale_gate_up", gu_s),
            ("weight_packed_down", dn_w), ("weight_scale_down", dn_s),
        ):
            sources[name].append(t)
    sources = {name: [torch.stack(v)] for name, v in sources.items()}
    cache.set_bank_sources(sources)
    cache.int8_k_gate_up = k_gu
    cache.int8_k_down = k_dn
    cache.materialize_layer(0)
    cache.copy_missing()
    return cache


@XPU
@pytest.mark.xpu
def test_expert_forward_dispatches_to_native_fused_path():
    hidden, inter = 16, 8
    cache = _cache_with_int8_bank(hidden, 2 * inter, inter, hidden, num_experts=2, cache_size=2)
    assert cache.is_xpu
    accessor = SlotWeightAccessor(cache, intermediate=inter, dtype=torch.float32)

    with patch(
        "freetoken.kernel.triton.fused_int8_linear.fused_int8_expert_forward",
        wraps=fused_int8_linear_mod.fused_int8_expert_forward,
    ) as spy:
        for e in range(2):
            slot = int(cache.slot_for_id[0, e].item())
            x = torch.randn(1, hidden, device=DEVICE, dtype=torch.float32)  # M=1: real call-site batch size
            accessor.expert_forward(slot, x)

    assert spy.call_count == 2


@XPU
@pytest.mark.xpu
def test_expert_forward_matches_dequant_fallback():
    hidden, inter = 16, 8
    cache = _cache_with_int8_bank(hidden, 2 * inter, inter, hidden, num_experts=1, cache_size=1)
    accessor = SlotWeightAccessor(cache, intermediate=inter, dtype=torch.float32)
    slot = int(cache.slot_for_id[0, 0].item())
    x = torch.randn(1, hidden, device=DEVICE, dtype=torch.float32)

    native_out = accessor.expert_forward(slot, x)

    gate_w, up_w, down_w = accessor.get(slot)
    fallback_out = (torch.nn.functional.silu(x @ gate_w.t()) * (x @ up_w.t())) @ down_w.t()

    torch.testing.assert_close(native_out, fallback_out, atol=2e-3, rtol=2e-2)
