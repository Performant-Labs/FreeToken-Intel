"""SlotWeightAccessor: the offload forward's per-slot weight lookup,
abstracted over int8_channel (dequantize-at-compute, issue
`moe-quant-banks-int8`, #154). Companion to
test_moe_slot_weight_accessor.py (gptq_int4's own version, #137).

Builds against the shape/bank-name contract registered in
_BANK_SCHEMAS["int8_channel"] (weight/scale x {gate_up,down}, no extra
side tensor -- unlike gptq_int4's g_idx, a per-channel scale is already one
row per expert). CPU-only, small synthetic fixtures, no real checkpoint, no
XPU.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from freetoken.kernel.triton.int8_linear import dequantize_int8_channel
from freetoken.moe.offload_cache import OffloadMoeCache, SlotWeightAccessor

DEVICE = torch.device("cpu")


def _make_packed_projection(n: int, k: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A real (non-constant) small per-channel-INT8-packed ``[N, K]``
    projection: weight codes and scales vary by row (derived from `seed`),
    so a test can't accidentally pass by everything happening to be
    uniform."""
    g = torch.Generator().manual_seed(seed)
    weight = torch.randint(-127, 128, (n, k), generator=g, dtype=torch.int8)
    scale = torch.rand(n, generator=g) * 0.1 + 0.01
    return weight, scale


def _cache_with_int8_bank(k_gu: int, n_gu: int, k_dn: int, n_dn: int, *, num_experts: int, cache_size: int):
    cache = OffloadMoeCache(1, num_experts, cache_size, DEVICE, quant_format="int8_channel")
    sources = {name: [] for name in ("weight_gate_up", "scale_gate_up", "weight_down", "scale_down")}
    projections = []  # keep the raw per-expert projections to check against
    for e in range(num_experts):
        gu_w, gu_s = _make_packed_projection(n_gu, k_gu, seed=100 + e)
        dn_w, dn_s = _make_packed_projection(n_dn, k_dn, seed=200 + e)
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


def test_int8_get_matches_direct_dequant_for_each_expert():
    hidden, inter, num_experts = 16, 8, 3
    cache, projections = _cache_with_int8_bank(
        k_gu=hidden, n_gu=2 * inter, k_dn=inter, n_dn=hidden,
        num_experts=num_experts, cache_size=num_experts,
    )
    accessor = SlotWeightAccessor(cache, intermediate=inter, dtype=torch.float32)

    for e in range(num_experts):
        slot = int(cache.slot_for_id[0, e].item())
        gate_w, up_w, down_w = accessor.get(slot)
        gu_w, gu_s, dn_w, dn_s = projections[e]

        expected_gu = dequantize_int8_channel(gu_w, gu_s, out_dtype=torch.float32)
        expected_dn = dequantize_int8_channel(dn_w, dn_s, out_dtype=torch.float32)

        torch.testing.assert_close(gate_w, expected_gu[0:inter])
        torch.testing.assert_close(up_w, expected_gu[inter : 2 * inter])
        torch.testing.assert_close(down_w, expected_dn)


def test_int8_get_dtype_matches_requested_dtype_not_scale_dtype():
    """Same bug class issue #138 found for gptq_int4: dequantizing to the
    checkpoint's own scale dtype (here forced to float16) rather than the
    activation dtype requested at construction would crash matmul-ing
    against bf16 activations elsewhere in the model."""
    hidden, inter = 16, 8
    cache, _ = _cache_with_int8_bank(
        k_gu=hidden, n_gu=2 * inter, k_dn=inter, n_dn=hidden,
        num_experts=1, cache_size=1,
    )
    # Force the scale bank to a dtype that must NOT leak into the output.
    cache.bank_caches["scale_gate_up"] = cache.bank_caches["scale_gate_up"].to(torch.float16)
    cache.bank_caches["scale_down"] = cache.bank_caches["scale_down"].to(torch.float16)

    accessor = SlotWeightAccessor(cache, intermediate=inter, dtype=torch.bfloat16)
    gate_w, up_w, down_w = accessor.get(0)
    assert gate_w.dtype == torch.bfloat16
    assert up_w.dtype == torch.bfloat16
    assert down_w.dtype == torch.bfloat16


def test_int8_get_shapes_match_out_in_convention():
    hidden, inter = 16, 8
    cache, _ = _cache_with_int8_bank(
        k_gu=hidden, n_gu=2 * inter, k_dn=inter, n_dn=hidden,
        num_experts=1, cache_size=1,
    )
    accessor = SlotWeightAccessor(cache, intermediate=inter, dtype=torch.float32)
    gate_w, up_w, down_w = accessor.get(0)
    assert gate_w.shape == (inter, hidden)
    assert up_w.shape == (inter, hidden)
    assert down_w.shape == (hidden, inter)


def test_int8_get_caches_per_slot_within_one_instance():
    """A distinct slot is dequantized once, not once per .get() call."""
    hidden, inter = 16, 8
    cache, _ = _cache_with_int8_bank(
        k_gu=hidden, n_gu=2 * inter, k_dn=inter, n_dn=hidden,
        num_experts=1, cache_size=1,
    )
    accessor = SlotWeightAccessor(cache, intermediate=inter, dtype=torch.float32)
    first = accessor.get(0)
    second = accessor.get(0)
    for a, b in zip(first, second):
        assert a.data_ptr() == b.data_ptr()  # identical cached tensor object, not recomputed
