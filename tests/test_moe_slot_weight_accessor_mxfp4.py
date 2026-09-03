"""SlotWeightAccessor's ``"mxfp4"`` branch: dequantize-at-compute for MXFP4
packed banks (issue moe-quant-banks-mxfp4, #153).

Companion to test_moe_slot_weight_accessor.py (the ``gptq_int4`` branch's own
tests). Builds against the shape/bank-name contract #153 registered in
_BANK_SCHEMAS["mxfp4"] (blocks/scales x {gate_up,down}, no g_idx-equivalent
side table -- MXFP4's scale is fully local to its own 32-element block).
CPU-only, small synthetic fixtures, no real checkpoint, no XPU.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from freetoken.kernel.triton.mxfp4_linear import dequantize_mxfp4_blocks, quantize_mxfp4_blocks
from freetoken.moe.offload_cache import OffloadMoeCache, SlotWeightAccessor

DEVICE = torch.device("cpu")


def _make_packed_projection(out_features: int, k: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A real (non-constant) small MXFP4-packed ``[out_features, K]``
    projection: values vary by position (derived from ``seed``), so a test
    can't accidentally pass by everything happening to be uniform."""
    g = torch.Generator().manual_seed(seed)
    dense = (torch.rand(out_features, k, generator=g) - 0.5) * 4.0  # within E2M1's +-6.0 range
    return quantize_mxfp4_blocks(dense)


def _cache_with_mxfp4_bank(hidden: int, inter: int, *, num_experts: int, cache_size: int):
    cache = OffloadMoeCache(1, num_experts, cache_size, DEVICE, quant_format="mxfp4")
    sources = {name: [] for name in ("blocks_gate_up", "scales_gate_up", "blocks_down", "scales_down")}
    projections = []  # keep the raw per-expert projections to check against
    for e in range(num_experts):
        gu_blocks, gu_scales = _make_packed_projection(2 * inter, hidden, seed=100 + e)
        dn_blocks, dn_scales = _make_packed_projection(hidden, inter, seed=200 + e)
        projections.append((gu_blocks, gu_scales, dn_blocks, dn_scales))
        for name, t in (
            ("blocks_gate_up", gu_blocks), ("scales_gate_up", gu_scales),
            ("blocks_down", dn_blocks), ("scales_down", dn_scales),
        ):
            sources[name].append(t)
    sources = {name: [torch.stack(v)] for name, v in sources.items()}  # -> [1 layer][E, ...]
    cache.set_bank_sources(sources)
    cache.materialize_layer(0)
    cache.copy_missing()
    return cache, projections


def test_mxfp4_get_matches_direct_dequant_for_each_expert():
    hidden, inter, num_experts = 64, 32, 3
    cache, projections = _cache_with_mxfp4_bank(hidden, inter, num_experts=num_experts, cache_size=num_experts)
    accessor = SlotWeightAccessor(cache, intermediate=inter, dtype=torch.float32)

    for e in range(num_experts):
        slot = int(cache.slot_for_id[0, e].item())
        gate_w, up_w, down_w = accessor.get(slot)
        gu_blocks, gu_scales, dn_blocks, dn_scales = projections[e]

        expected_gu = dequantize_mxfp4_blocks(gu_blocks, gu_scales, out_dtype=torch.float32)
        expected_dn = dequantize_mxfp4_blocks(dn_blocks, dn_scales, out_dtype=torch.float32)

        torch.testing.assert_close(gate_w, expected_gu[0:inter])
        torch.testing.assert_close(up_w, expected_gu[inter : 2 * inter])
        torch.testing.assert_close(down_w, expected_dn)


def test_mxfp4_get_dtype_matches_requested_dtype_not_scales_dtype():
    """Mirrors the real bug #138 found for GPTQ: SlotWeightAccessor must
    dequantize to the dtype it was constructed with (the model's activation
    dtype), never anything checkpoint-stored -- here MXFP4's ``scales`` bank
    is itself uint8 (an E8M0 exponent byte, not a float dtype at all), so
    this is an even sharper version of the same invariant: the *output*
    dtype must be the requested one regardless of the underlying storage."""
    hidden, inter = 64, 32
    cache, _ = _cache_with_mxfp4_bank(hidden, inter, num_experts=1, cache_size=1)

    accessor = SlotWeightAccessor(cache, intermediate=inter, dtype=torch.bfloat16)
    gate_w, up_w, down_w = accessor.get(0)
    assert gate_w.dtype == torch.bfloat16
    assert up_w.dtype == torch.bfloat16
    assert down_w.dtype == torch.bfloat16


def test_mxfp4_get_shapes_match_out_in_convention():
    hidden, inter = 64, 32
    cache, _ = _cache_with_mxfp4_bank(hidden, inter, num_experts=1, cache_size=1)
    accessor = SlotWeightAccessor(cache, intermediate=inter, dtype=torch.float32)
    gate_w, up_w, down_w = accessor.get(0)
    assert gate_w.shape == (inter, hidden)
    assert up_w.shape == (inter, hidden)
    assert down_w.shape == (hidden, inter)


def test_mxfp4_get_caches_per_slot_within_one_instance():
    """A distinct slot is dequantized once, not once per .get() call -- the
    whole point of this issue (dequantize only the resident working set, at
    most once per step, never re-derive redundantly)."""
    hidden, inter = 64, 32
    cache, _ = _cache_with_mxfp4_bank(hidden, inter, num_experts=1, cache_size=1)
    accessor = SlotWeightAccessor(cache, intermediate=inter, dtype=torch.float32)
    first = accessor.get(0)
    second = accessor.get(0)
    for a, b in zip(first, second):
        assert a.data_ptr() == b.data_ptr()  # identical cached tensor object, not recomputed
