"""SlotWeightAccessor: the offload forward's per-slot weight lookup,
abstracted over bf16 (unchanged) and gptq_int4 (dequantize-at-compute,
issue moe-quant-banks-compute, #137).

Builds against the shape/bank-name contract #136 registered in
_BANK_SCHEMAS["gptq_int4"] (qweight/qzeros/scales x {gate_up,down}, no
g_idx bank -- reconstructed implicitly from group_size). CPU-only, small
synthetic fixtures, no real checkpoint, no XPU.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from freetoken.kernel.triton.gptq_linear import dequantize_gptq_int4_sequential_groups
from freetoken.moe.offload_cache import OffloadMoeCache, SlotWeightAccessor

DEVICE = torch.device("cpu")


def _pack_nibbles(codes: list[int]) -> int:
    word = 0
    for i, c in enumerate(codes):
        word |= c << (4 * i)
    return word


def _make_packed_projection(k: int, n: int, group_size: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A real (non-constant) small GPTQ-packed [K, N] projection: qweight
    codes and zero-points vary by position (derived from `seed`), so a test
    can't accidentally pass by everything happening to be uniform."""
    g = torch.Generator().manual_seed(seed)
    n_groups = k // group_size
    codes = torch.randint(0, 16, (k, n), generator=g)
    qweight = torch.zeros(k // 8, n, dtype=torch.int32)
    for row8 in range(k // 8):
        for col in range(n):
            word = 0
            for r in range(8):
                word |= int(codes[row8 * 8 + r, col]) << (4 * r)
            qweight[row8, col] = word if word < (1 << 31) else word - (1 << 32)
    zero_codes = torch.randint(1, 15, (n_groups, n), generator=g)  # stored value (post -1)
    qzeros = torch.zeros(n_groups, n // 8, dtype=torch.int32)
    for grp in range(n_groups):
        for col8 in range(n // 8):
            word = 0
            for c in range(8):
                word |= int(zero_codes[grp, col8 * 8 + c] - 1) << (4 * c)
            qzeros[grp, col8] = word if word < (1 << 31) else word - (1 << 32)
    scales = torch.rand(n_groups, n, generator=g) * 0.1 + 0.01
    return qweight, qzeros, scales


def _cache_with_gptq_bank(k_gu: int, n_gu: int, k_dn: int, n_dn: int, group_size: int, *, num_experts: int, cache_size: int):
    cache = OffloadMoeCache(1, num_experts, cache_size, DEVICE, quant_format="gptq_int4")
    cache.gptq_group_size = group_size  # SlotWeightAccessor reads this (default 128 otherwise)
    sources = {name: [] for name in ("qweight_gate_up", "qzeros_gate_up", "scales_gate_up", "qweight_down", "qzeros_down", "scales_down")}
    projections = []  # keep the raw per-expert projections to check against
    for e in range(num_experts):
        gu_qw, gu_qz, gu_sc = _make_packed_projection(k_gu, n_gu, group_size, seed=100 + e)
        dn_qw, dn_qz, dn_sc = _make_packed_projection(k_dn, n_dn, group_size, seed=200 + e)
        projections.append((gu_qw, gu_qz, gu_sc, dn_qw, dn_qz, dn_sc))
        for name, t in (
            ("qweight_gate_up", gu_qw), ("qzeros_gate_up", gu_qz), ("scales_gate_up", gu_sc),
            ("qweight_down", dn_qw), ("qzeros_down", dn_qz), ("scales_down", dn_sc),
        ):
            sources[name].append(t)
    sources = {name: [torch.stack(v)] for name, v in sources.items()}  # -> [1 layer][E, ...]
    cache.set_bank_sources(sources)
    cache.materialize_layer(0)
    cache.copy_missing()
    return cache, projections


def test_gptq_get_matches_direct_dequant_for_each_expert():
    hidden, inter, group_size, num_experts = 16, 8, 8, 3
    cache, projections = _cache_with_gptq_bank(
        k_gu=hidden, n_gu=2 * inter, k_dn=inter, n_dn=hidden, group_size=group_size,
        num_experts=num_experts, cache_size=num_experts,
    )
    accessor = SlotWeightAccessor(cache, intermediate=inter)

    for e in range(num_experts):
        slot = int(cache.slot_for_id[0, e].item())
        gate_w, up_w, down_w = accessor.get(slot)
        gu_qw, gu_qz, gu_sc, dn_qw, dn_qz, dn_sc = projections[e]

        # SlotWeightAccessor dequantizes to the bank's own scales dtype (matching
        # the real checkpoint's fp16/bf16 scales, not a hardcoded dtype); the
        # fixture's scales are float32, so the expected value must match that.
        expected_gu = dequantize_gptq_int4_sequential_groups(gu_qw, gu_qz, gu_sc, group_size=group_size, out_dtype=gu_sc.dtype).T
        expected_dn = dequantize_gptq_int4_sequential_groups(dn_qw, dn_qz, dn_sc, group_size=group_size, out_dtype=dn_sc.dtype).T

        torch.testing.assert_close(gate_w, expected_gu[0:inter])
        torch.testing.assert_close(up_w, expected_gu[inter : 2 * inter])
        torch.testing.assert_close(down_w, expected_dn)


def test_gptq_get_shapes_match_out_in_convention():
    hidden, inter, group_size = 16, 8, 8
    cache, _ = _cache_with_gptq_bank(
        k_gu=hidden, n_gu=2 * inter, k_dn=inter, n_dn=hidden, group_size=group_size,
        num_experts=1, cache_size=1,
    )
    accessor = SlotWeightAccessor(cache, intermediate=inter)
    gate_w, up_w, down_w = accessor.get(0)
    assert gate_w.shape == (inter, hidden)
    assert up_w.shape == (inter, hidden)
    assert down_w.shape == (hidden, inter)


def test_gptq_get_caches_per_slot_within_one_instance():
    """A distinct slot is dequantized once, not once per .get() call -- the
    whole point of this issue (dequantize only the resident working set,
    at most once per step, never re-derive redundantly)."""
    hidden, inter, group_size = 16, 8, 8
    cache, _ = _cache_with_gptq_bank(
        k_gu=hidden, n_gu=2 * inter, k_dn=inter, n_dn=hidden, group_size=group_size,
        num_experts=1, cache_size=1,
    )
    accessor = SlotWeightAccessor(cache, intermediate=inter)
    first = accessor.get(0)
    second = accessor.get(0)
    for a, b in zip(first, second):
        assert a.data_ptr() == b.data_ptr()  # identical cached tensor object, not recomputed


def test_gptq_accessor_refuses_to_guess_group_size():
    """A missing cache.gptq_group_size must raise, not silently default to
    some group size -- a wrong-but-plausible default would dequantize with
    the wrong group boundaries and produce silently-wrong-but-finite numbers
    (found the hard way while writing this test file's own fixtures)."""
    E, hidden, inter = 1, 16, 8
    cache = OffloadMoeCache(1, E, E, DEVICE, quant_format="gptq_int4")
    # Deliberately do NOT set cache.gptq_group_size.
    gu_qw, gu_qz, gu_sc = _make_packed_projection(hidden, 2 * inter, 8, seed=1)
    dn_qw, dn_qz, dn_sc = _make_packed_projection(inter, hidden, 8, seed=2)
    cache.set_bank_sources({
        "qweight_gate_up": [gu_qw.unsqueeze(0)], "qzeros_gate_up": [gu_qz.unsqueeze(0)], "scales_gate_up": [gu_sc.unsqueeze(0)],
        "qweight_down": [dn_qw.unsqueeze(0)], "qzeros_down": [dn_qz.unsqueeze(0)], "scales_down": [dn_sc.unsqueeze(0)],
    })
    with pytest.raises(ValueError, match="gptq_group_size"):
        SlotWeightAccessor(cache, intermediate=inter)


def test_bf16_accessor_behavior_unchanged():
    """The bf16 path must be a pure passthrough to bank_views() indexing --
    no dequant, no extra allocation, identical values to reading gu/dn directly."""
    E, hidden, inter = 2, 16, 8
    cache = OffloadMoeCache(1, E, E, DEVICE, quant_format="bf16")
    gu_src = [torch.randn(E, 2 * inter, hidden)]
    dn_src = [torch.randn(E, hidden, inter)]
    cache.set_bank_sources({"gate_up": gu_src, "down": dn_src})
    cache.materialize_layer(0)
    cache.copy_missing()

    accessor = SlotWeightAccessor(cache, intermediate=inter)
    gu, dn = cache.bank_views()
    for e in range(E):
        slot = int(cache.slot_for_id[0, e].item())
        gate_w, up_w, down_w = accessor.get(slot)
        torch.testing.assert_close(gate_w, gu[slot, 0:inter])
        torch.testing.assert_close(up_w, gu[slot, inter : 2 * inter])
        torch.testing.assert_close(down_w, dn[slot])
