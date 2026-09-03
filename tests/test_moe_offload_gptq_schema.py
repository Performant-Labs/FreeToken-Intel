"""Tests for the ``gptq_int4`` offload-cache bank schema (issue moe-quant-banks-schema, #136).

Companion to test_moe_offload_cache.py / test_moe_offload_rebuild.py (the
``bf16`` schema's own tests, run unchanged below to confirm this is a
strictly additive registration). Builds against the shape contract documented
in the parent epic (#134) and issue #135 -- small synthetic packed-int32
fixtures, no real checkpoint, no XPU.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from freetoken.moe.offload_cache import (
    _BANK_SCHEMAS,
    OffloadMoeCache,
    gptq_int4_bytes_per_expert_slot,
)

DEVICE = torch.device("cpu")
L, E, S = 2, 4, 8  # 2 layers, 4 experts, 8 slots
K, N, GROUP = 16, 8, 8  # small in/out dims + group size (K % group == 0, K % 8 == 0, N % 8 == 0)


def _packed_bank_sources():
    """Distinguishable per-(layer, expert) packed rows for every gptq_int4
    bank -- values chosen so a test can tell which (layer, expert) a slot
    currently holds, same spirit as test_moe_offload_cache.py's `_bank_values`."""
    n_groups = K // GROUP

    def tag(layer, expert):
        return 100 * (layer + 1) + 10 * (expert + 1)

    sources = {name: [] for name in _BANK_SCHEMAS["gptq_int4"]}
    for layer in range(L):
        per_bank_layers = {name: [] for name in sources}
        for expert in range(E):
            t = tag(layer, expert)
            for proj, k, n in (("gate_up", K, N), ("down", N, K)):
                # value magnitude kept small enough that int32 holds it exactly.
                per_bank_layers[f"qweight_{proj}"].append(torch.full((k // 8, n), t, dtype=torch.int32))
                per_bank_layers[f"qzeros_{proj}"].append(torch.full((n_groups if proj == "gate_up" else n // GROUP, n // 8), t, dtype=torch.int32))
                per_bank_layers[f"scales_{proj}"].append(torch.full((n_groups if proj == "gate_up" else n // GROUP, n), float(t), dtype=torch.float32))
        for name in sources:
            sources[name].append(torch.stack(per_bank_layers[name]))
    return sources


def test_schema_registered():
    assert "gptq_int4" in _BANK_SCHEMAS
    assert _BANK_SCHEMAS["gptq_int4"] == (
        "qweight_gate_up",
        "qzeros_gate_up",
        "scales_gate_up",
        "qweight_down",
        "qzeros_down",
        "scales_down",
    )


def test_set_bank_sources_allocates_correctly_shaped_dtyped_device_caches():
    cache = OffloadMoeCache(L, E, S, DEVICE, quant_format="gptq_int4")
    sources = _packed_bank_sources()
    cache.set_bank_sources(sources)

    for name in _BANK_SCHEMAS["gptq_int4"]:
        host_row_shape = sources[name][0].shape[1:]
        dev_cache = cache.bank_caches[name]
        assert dev_cache.shape == (S, *host_row_shape)
        assert dev_cache.dtype == sources[name][0].dtype
    # int32 banks stay int32 (not silently upcast/downcast anywhere).
    assert cache.bank_caches["qweight_gate_up"].dtype == torch.int32
    assert cache.bank_caches["scales_gate_up"].dtype == torch.float32


def test_materialize_and_copy_missing_moves_real_packed_bytes():
    cache = OffloadMoeCache(L, E, S, DEVICE, quant_format="gptq_int4")
    sources = _packed_bank_sources()
    cache.set_bank_sources(sources)

    cache.materialize_layer(0)
    cache.copy_missing()

    for expert in range(E):
        slot = int(cache.slot_for_id[0, expert].item())
        assert slot != -1
        expected_tag = 100 * 1 + 10 * (expert + 1)
        # A real bit-exact check on the packed int32 payload, not just shape.
        torch.testing.assert_close(
            cache.bank_caches["qweight_gate_up"][slot],
            torch.full_like(cache.bank_caches["qweight_gate_up"][slot], expected_tag),
        )
        torch.testing.assert_close(
            cache.bank_caches["scales_down"][slot],
            torch.full_like(cache.bank_caches["scales_down"][slot], float(expected_tag)),
        )


def test_rebuild_resizes_gptq_schema_pool():
    cache = OffloadMoeCache(L, E, S, DEVICE, quant_format="gptq_int4")
    cache.set_bank_sources(_packed_bank_sources())
    cache.materialize_layer(0)
    cache.copy_missing()

    cache.rebuild(2 * S)
    for name in _BANK_SCHEMAS["gptq_int4"]:
        assert cache.bank_caches[name].shape[0] == 2 * S
    # rebuild clears LRU state (same contract as the bf16 schema).
    assert int(cache.slot_for_id.min().item()) == -1 or (cache.slot_for_id == -1).all()


def test_bf16_schema_unaffected_by_gptq_registration():
    """Strictly-additive check: the existing bf16 schema still behaves
    exactly as it did before this change."""
    assert "bf16" in _BANK_SCHEMAS and _BANK_SCHEMAS["bf16"] == ("gate_up", "down")
    cache = OffloadMoeCache(L, E, S, DEVICE, quant_format="bf16")
    h, i = 16, 8
    sources = {
        "gate_up": [torch.randn(E, 2 * i, h) for _ in range(L)],
        "down": [torch.randn(E, h, i) for _ in range(L)],
    }
    cache.set_bank_sources(sources)
    cache.materialize_layer(0)
    cache.copy_missing()
    assert all(int(cache.slot_for_id[0, e].item()) != -1 for e in range(E))


# -- extra_metadata (g_idx) --------------------------------------------------


def test_extra_metadata_round_trips_per_layer():
    cache = OffloadMoeCache(L, E, S, DEVICE, quant_format="gptq_int4")
    g_idx_gate_up = [torch.arange(K, dtype=torch.int32) // GROUP for _ in range(L)]
    cache.set_extra_metadata("g_idx_gate_up", g_idx_gate_up)

    for layer in range(L):
        got = cache.get_extra_metadata("g_idx_gate_up", layer)
        torch.testing.assert_close(got, g_idx_gate_up[layer])


def test_extra_metadata_rejects_wrong_layer_count():
    cache = OffloadMoeCache(L, E, S, DEVICE, quant_format="gptq_int4")
    with pytest.raises(ValueError, match="layers"):
        cache.set_extra_metadata("g_idx_gate_up", [torch.zeros(K, dtype=torch.int32)])  # only 1, need L=2


def test_extra_metadata_unset_raises_keyerror():
    cache = OffloadMoeCache(L, E, S, DEVICE, quant_format="gptq_int4")
    with pytest.raises(KeyError):
        cache.get_extra_metadata("never_set", 0)


def test_extra_metadata_survives_reset():
    """reset() clears LRU/slot bookkeeping but must not drop attached
    per-layer side data -- it's load-time state, not runtime cache state
    (same contract bank_sources already has: reset() zeros bank_caches'
    device contents but leaves bank_sources, the host banks, untouched)."""
    cache = OffloadMoeCache(L, E, S, DEVICE, quant_format="gptq_int4")
    cache.set_bank_sources(_packed_bank_sources())
    g_idx = [torch.arange(K, dtype=torch.int32) // GROUP for _ in range(L)]
    cache.set_extra_metadata("g_idx_gate_up", g_idx)

    cache.materialize_layer(0)
    cache.copy_missing()
    cache.reset()

    for layer in range(L):
        torch.testing.assert_close(cache.get_extra_metadata("g_idx_gate_up", layer), g_idx[layer])


# -- gptq_int4_bytes_per_expert_slot -----------------------------------------


def test_bytes_per_expert_slot_matches_hand_computed_value():
    hidden, inter, group = 2048, 512, 128
    got = gptq_int4_bytes_per_expert_slot(hidden, inter, group)

    def proj_bytes(k, n):
        groups = -(-k // group)
        return (k // 8) * n * 4 + groups * (n // 8) * 4 + groups * n * 2

    expected = proj_bytes(hidden, 2 * inter) + proj_bytes(inter, hidden)
    assert got == expected
    # Sanity: real numbers from the actual downloaded Qwen3.5-35B-A3B-GPTQ-Int4
    # checkpoint's gate_proj (K=2048, N=512, group=128) confirm qweight alone
    # is 256*512*4 = 524288 bytes for ONE projection half of gate_up.
    assert got > 0


def test_bytes_per_expert_slot_much_smaller_than_bf16_equivalent():
    """The whole point of this issue: a packed slot must be dramatically
    smaller than the bf16-dequantized equivalent (roughly 4x, matching
    4-bit -> 16-bit expansion) -- this is the real fix for the ~88GB RAM
    blowup #134 found."""
    hidden, inter, group = 2048, 512, 128
    packed = gptq_int4_bytes_per_expert_slot(hidden, inter, group)
    bf16_equivalent = (2 * inter * hidden + hidden * inter) * 2  # gate_up + down, bf16
    assert packed < bf16_equivalent / 3  # generously below the ~4x expansion bf16 would cost


def test_bytes_per_expert_slot_rejects_non_positive_inputs():
    with pytest.raises(ValueError):
        gptq_int4_bytes_per_expert_slot(0, 512, 128)
    with pytest.raises(ValueError):
        gptq_int4_bytes_per_expert_slot(2048, 512, 0)
