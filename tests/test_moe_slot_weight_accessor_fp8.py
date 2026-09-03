"""SlotWeightAccessor: the offload forward's per-slot weight lookup,
abstracted over bf16 (unchanged), gptq_int4, and fp8_block (issue
`moe-quant-banks-fp8`, #152) -- dequantize-at-compute.

Builds against the shape/bank-name contract registered in
_BANK_SCHEMAS["fp8_block"] (weight/scale x {gate_up,down}, no shared side
tensor -- unlike gptq_int4's g_idx, block-FP8's scale is genuinely per
expert). CPU-only, small synthetic fixtures, no real checkpoint, no XPU.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from freetoken.kernel.triton.fp8_block_linear import dequantize_block_fp8, quantize_block_fp8
from freetoken.moe.offload_cache import OffloadMoeCache, SlotWeightAccessor

DEVICE = torch.device("cpu")


def _make_packed_projection(n: int, k: int, block: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A real (non-constant) small block-FP8-packed [N, K] projection: values
    vary by position (derived from `seed`), so a test can't accidentally pass
    by everything happening to be uniform."""
    g = torch.Generator().manual_seed(seed)
    dense = torch.randn(n, k, generator=g)
    return quantize_block_fp8(dense, block=block)


def _cache_with_fp8_bank(n_gu: int, k_gu: int, n_dn: int, k_dn: int, block: int, *, num_experts: int, cache_size: int):
    cache = OffloadMoeCache(1, num_experts, cache_size, DEVICE, quant_format="fp8_block")
    cache.fp8_block_size = block  # SlotWeightAccessor reads this (default 128 otherwise)
    sources = {name: [] for name in ("weight_gate_up", "scale_gate_up", "weight_down", "scale_down")}
    projections = []  # keep the raw per-expert projections to check against
    for e in range(num_experts):
        gu_w, gu_s = _make_packed_projection(n_gu, k_gu, block, seed=100 + e)
        dn_w, dn_s = _make_packed_projection(n_dn, k_dn, block, seed=200 + e)
        projections.append((gu_w, gu_s, dn_w, dn_s))
        for name, t in (
            ("weight_gate_up", gu_w), ("scale_gate_up", gu_s),
            ("weight_down", dn_w), ("scale_down", dn_s),
        ):
            sources[name].append(t)
    sources = {name: [torch.stack(v)] for name, v in sources.items()}  # -> [1 layer][E, ...]
    cache.set_bank_sources(sources)
    cache.materialize_layer(0)
    cache.copy_missing()
    return cache, projections


def test_fp8_get_matches_direct_dequant_for_each_expert():
    hidden, inter, block, num_experts = 16, 8, 8, 3
    cache, projections = _cache_with_fp8_bank(
        n_gu=2 * inter, k_gu=hidden, n_dn=hidden, k_dn=inter, block=block,
        num_experts=num_experts, cache_size=num_experts,
    )
    accessor = SlotWeightAccessor(cache, intermediate=inter, dtype=torch.float32)

    for e in range(num_experts):
        slot = int(cache.slot_for_id[0, e].item())
        gate_w, up_w, down_w = accessor.get(slot)
        gu_w, gu_s, dn_w, dn_s = projections[e]

        # SlotWeightAccessor dequantizes to the dtype it was constructed with
        # (torch.float32 here) -- NOT the bank's own weight_scale_inv dtype (the
        # same dtype bug class issue #138 found for gptq_int4: the loaded
        # checkpoint's stored scale dtype must never leak into the dequantized
        # output, which must match the model's activation dtype instead).
        expected_gu = dequantize_block_fp8(gu_w, gu_s, block=block, out_dtype=torch.float32)
        expected_dn = dequantize_block_fp8(dn_w, dn_s, block=block, out_dtype=torch.float32)

        torch.testing.assert_close(gate_w, expected_gu[0:inter])
        torch.testing.assert_close(up_w, expected_gu[inter : 2 * inter])
        torch.testing.assert_close(down_w, expected_dn)


def test_fp8_get_dtype_matches_requested_dtype_not_scale_dtype():
    """The gptq_int4 bug class (#138) applied to fp8_block: dequantizing must
    always target the requested dtype, never whatever dtype the checkpoint
    happens to store weight_scale_inv as. Builds a fixture whose scale bank is
    deliberately float16 -- a different dtype than the bfloat16 requested at
    construction -- and asserts the *output* matches the requested dtype."""
    hidden, inter, block = 16, 8, 8
    cache, _ = _cache_with_fp8_bank(
        n_gu=2 * inter, k_gu=hidden, n_dn=hidden, k_dn=inter, block=block,
        num_experts=1, cache_size=1,
    )
    cache.bank_caches["scale_gate_up"] = cache.bank_caches["scale_gate_up"].to(torch.float16)
    cache.bank_caches["scale_down"] = cache.bank_caches["scale_down"].to(torch.float16)

    accessor = SlotWeightAccessor(cache, intermediate=inter, dtype=torch.bfloat16)
    gate_w, up_w, down_w = accessor.get(0)
    assert gate_w.dtype == torch.bfloat16
    assert up_w.dtype == torch.bfloat16
    assert down_w.dtype == torch.bfloat16


def test_fp8_get_shapes_match_out_in_convention():
    hidden, inter, block = 16, 8, 8
    cache, _ = _cache_with_fp8_bank(
        n_gu=2 * inter, k_gu=hidden, n_dn=hidden, k_dn=inter, block=block,
        num_experts=1, cache_size=1,
    )
    accessor = SlotWeightAccessor(cache, intermediate=inter, dtype=torch.float32)
    gate_w, up_w, down_w = accessor.get(0)
    assert gate_w.shape == (inter, hidden)
    assert up_w.shape == (inter, hidden)
    assert down_w.shape == (hidden, inter)


def test_fp8_get_caches_per_slot_within_one_instance():
    """A distinct slot is dequantized once, not once per .get() call -- the
    whole point of this issue (dequantize only the resident working set, at
    most once per step, never re-derive redundantly)."""
    hidden, inter, block = 16, 8, 8
    cache, _ = _cache_with_fp8_bank(
        n_gu=2 * inter, k_gu=hidden, n_dn=hidden, k_dn=inter, block=block,
        num_experts=1, cache_size=1,
    )
    accessor = SlotWeightAccessor(cache, intermediate=inter, dtype=torch.float32)
    first = accessor.get(0)
    second = accessor.get(0)
    for a, b in zip(first, second):
        assert a.data_ptr() == b.data_ptr()  # identical cached tensor object, not recomputed


def test_fp8_accessor_defaults_block_size_to_128():
    """Unlike gptq_int4's group_size (a genuine per-checkpoint choice, so
    SlotWeightAccessor refuses to guess it), block-FP8's block size is a
    fixed, universal convention (128) that every real checkpoint found so
    far uses -- so a cache with no fp8_block_size override must not raise,
    and must dequantize using the module's own default block."""
    hidden, inter = 256, 128  # multiples of the real default block (128)
    cache, projections = _cache_with_fp8_bank(
        n_gu=2 * inter, k_gu=hidden, n_dn=hidden, k_dn=inter, block=128,
        num_experts=1, cache_size=1,
    )
    del cache.fp8_block_size  # exercise the no-override default, not this fixture's explicit set
    accessor = SlotWeightAccessor(cache, intermediate=inter, dtype=torch.float32)
    gate_w, up_w, down_w = accessor.get(0)
    gu_w, gu_s, dn_w, dn_s = projections[0]
    expected_gu = dequantize_block_fp8(gu_w, gu_s, block=128, out_dtype=torch.float32)
    expected_dn = dequantize_block_fp8(dn_w, dn_s, block=128, out_dtype=torch.float32)
    torch.testing.assert_close(gate_w, expected_gu[0:inter])
    torch.testing.assert_close(up_w, expected_gu[inter : 2 * inter])
    torch.testing.assert_close(down_w, expected_dn)


def test_bf16_accessor_behavior_unchanged():
    """The bf16 path must be a pure passthrough to bank_views() indexing --
    no dequant, no extra allocation, identical values to reading gu/dn
    directly. Re-asserted here (mirrors test_moe_slot_weight_accessor.py) as
    a strictly-additive check against this issue's fp8_block branch."""
    E, hidden, inter = 2, 16, 8
    cache = OffloadMoeCache(1, E, E, DEVICE, quant_format="bf16")
    gu_src = [torch.randn(E, 2 * inter, hidden)]
    dn_src = [torch.randn(E, hidden, inter)]
    cache.set_bank_sources({"gate_up": gu_src, "down": dn_src})
    cache.materialize_layer(0)
    cache.copy_missing()

    accessor = SlotWeightAccessor(cache, intermediate=inter, dtype=torch.float32)
    gu, dn = cache.bank_views()
    for e in range(E):
        slot = int(cache.slot_for_id[0, e].item())
        gate_w, up_w, down_w = accessor.get(slot)
        torch.testing.assert_close(gate_w, gu[slot, 0:inter])
        torch.testing.assert_close(up_w, gu[slot, inter : 2 * inter])
        torch.testing.assert_close(down_w, dn[slot])
