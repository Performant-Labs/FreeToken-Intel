"""SlotWeightAccessor: the offload forward's per-slot weight lookup,
abstracted over int8_channel (dequantize-at-compute, issue
`moe-quant-banks-int8`, #154). Companion to
test_moe_slot_weight_accessor.py (gptq_int4's own version, #137).

Corrected from an earlier, unverified draft (plain unpacked int8 tensors) --
the real bank shapes are ``weight_packed_*`` (int32, packed via
compressed-tensors' pack-quantized scheme) + ``weight_scale_*`` (one value
per (output channel, group) pair), matching the real bit-exact-verified
format in ``freetoken.kernel.triton.int8_packed_linear``. CPU-only, small
synthetic fixtures, no real checkpoint, no XPU.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from freetoken.kernel.triton.int8_packed_linear import dequantize_int8_packed
from freetoken.moe.offload_cache import OffloadMoeCache, SlotWeightAccessor

DEVICE = torch.device("cpu")
GROUP_SIZE = 4  # small group size so group-vs-per-channel dequant is actually exercised


def _pack_int8(codes: torch.Tensor) -> torch.Tensor:
    """``[N, K]`` int8 codes -> ``[N, K // 4]`` int32, compressed-tensors'
    dense num_bits=8 packing (byte i of each word = element word*4+i,
    stored as ``code + 128``)."""
    n, k = codes.shape
    assert k % 4 == 0
    codes32 = (codes.to(torch.int64) + 128).reshape(n, k // 4, 4)
    word = sum(codes32[:, :, i] << (8 * i) for i in range(4))
    word = torch.where(word >= (1 << 31), word - (1 << 32), word)
    return word.to(torch.int32)


def _make_packed_projection(n: int, k: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A real (non-constant) small packed-INT8 ``[N, K]`` projection: weight
    codes and per-group scales vary by row/group (derived from `seed`), so a
    test can't accidentally pass by everything happening to be uniform."""
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(-127, 128, (n, k), generator=g, dtype=torch.int8)
    num_groups = k // GROUP_SIZE
    scale = torch.rand(n, num_groups, generator=g) * 0.1 + 0.01
    return _pack_int8(codes), scale


def _cache_with_int8_bank(k_gu: int, n_gu: int, k_dn: int, n_dn: int, *, num_experts: int, cache_size: int):
    cache = OffloadMoeCache(1, num_experts, cache_size, DEVICE, quant_format="int8_channel")
    sources = {name: [] for name in ("weight_packed_gate_up", "weight_scale_gate_up", "weight_packed_down", "weight_scale_down")}
    projections = []  # keep the raw per-expert packed projections to check against
    for e in range(num_experts):
        gu_w, gu_s = _make_packed_projection(n_gu, k_gu, seed=100 + e)
        dn_w, dn_s = _make_packed_projection(n_dn, k_dn, seed=200 + e)
        projections.append((gu_w, gu_s, dn_w, dn_s))
        for name, t in (
            ("weight_packed_gate_up", gu_w), ("weight_scale_gate_up", gu_s),
            ("weight_packed_down", dn_w), ("weight_scale_down", dn_s),
        ):
            sources[name].append(t)
    sources = {name: [torch.stack(v)] for name, v in sources.items()}  # -> [1 layer][E, ...]
    cache.set_bank_sources(sources)
    cache.int8_k_gate_up = k_gu
    cache.int8_k_down = k_dn
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

        expected_gu = dequantize_int8_packed(gu_w, gu_s, k=hidden, out_dtype=torch.float32)
        expected_dn = dequantize_int8_packed(dn_w, dn_s, k=inter, out_dtype=torch.float32)

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
    cache.bank_caches["weight_scale_gate_up"] = cache.bank_caches["weight_scale_gate_up"].to(torch.float16)
    cache.bank_caches["weight_scale_down"] = cache.bank_caches["weight_scale_down"].to(torch.float16)

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


def test_missing_int8_k_raises():
    """SlotWeightAccessor refuses to guess -- mirrors gptq_int4's own
    gptq_group_size "refuse to guess" test."""
    cache = OffloadMoeCache(1, 1, 1, DEVICE, quant_format="int8_channel")
    sources = {name: [torch.zeros(1, *shape)] for name, shape in (
        ("weight_packed_gate_up", (4, 4)), ("weight_scale_gate_up", (4, 4)),
        ("weight_packed_down", (4, 4)), ("weight_scale_down", (4, 4)),
    )}
    cache.set_bank_sources(sources)
    with pytest.raises(ValueError, match="int8_k_gate_up"):
        SlotWeightAccessor(cache, intermediate=2, dtype=torch.float32)
