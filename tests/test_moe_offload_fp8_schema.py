"""Tests for the ``fp8_block`` offload-cache bank schema (issue
`moe-quant-banks-fp8`, #152).

Companion to test_moe_offload_gptq_schema.py (the ``gptq_int4`` schema's own
tests) and test_moe_offload_cache.py / test_moe_offload_rebuild.py (the
``bf16`` schema's). Small synthetic packed-fp8 fixtures, no real checkpoint,
no XPU.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from freetoken.moe.offload_cache import _BANK_SCHEMAS, OffloadMoeCache

DEVICE = torch.device("cpu")
L, E, S = 2, 4, 8  # 2 layers, 4 experts, 8 slots
N, K, BLOCK = 8, 16, 8  # small in/out dims + block size (both multiples of BLOCK)


def _packed_bank_sources():
    """Distinguishable per-(layer, expert) packed rows for every fp8_block
    bank -- values chosen so a test can tell which (layer, expert) a slot
    currently holds, same spirit as the gptq_int4 schema's own fixture."""
    n_blocks_row, n_blocks_col = N // BLOCK, K // BLOCK

    def tag(layer, expert):
        return 100 * (layer + 1) + 10 * (expert + 1)

    sources = {name: [] for name in _BANK_SCHEMAS["fp8_block"]}
    for layer in range(L):
        per_bank_layers = {name: [] for name in sources}
        for expert in range(E):
            t = tag(layer, expert)
            for proj, n, k in (("gate_up", N, K), ("down", N, K)):
                per_bank_layers[f"weight_{proj}"].append(torch.full((n, k), float(t), dtype=torch.float8_e4m3fn))
                per_bank_layers[f"scale_{proj}"].append(torch.full((n_blocks_row, n_blocks_col), float(t), dtype=torch.float32))
        for name in sources:
            sources[name].append(torch.stack(per_bank_layers[name]))
    return sources


def test_schema_registered():
    assert "fp8_block" in _BANK_SCHEMAS
    assert _BANK_SCHEMAS["fp8_block"] == (
        "weight_gate_up",
        "scale_gate_up",
        "weight_down",
        "scale_down",
    )


def test_set_bank_sources_allocates_correctly_shaped_dtyped_device_caches():
    cache = OffloadMoeCache(L, E, S, DEVICE, quant_format="fp8_block")
    sources = _packed_bank_sources()
    cache.set_bank_sources(sources)

    for name in _BANK_SCHEMAS["fp8_block"]:
        host_row_shape = sources[name][0].shape[1:]
        dev_cache = cache.bank_caches[name]
        assert dev_cache.shape == (S, *host_row_shape)
        assert dev_cache.dtype == sources[name][0].dtype
    assert cache.bank_caches["weight_gate_up"].dtype == torch.float8_e4m3fn
    assert cache.bank_caches["scale_gate_up"].dtype == torch.float32


def test_materialize_and_copy_missing_moves_real_packed_bytes():
    cache = OffloadMoeCache(L, E, S, DEVICE, quant_format="fp8_block")
    sources = _packed_bank_sources()
    cache.set_bank_sources(sources)

    cache.materialize_layer(0)
    cache.copy_missing()

    for expert in range(E):
        slot = int(cache.slot_for_id[0, expert].item())
        assert slot != -1
        expected_tag = 100 * 1 + 10 * (expert + 1)
        # fp8_e4m3 has coarse precision at this magnitude (tag 110 rounds to
        # 112, e.g.) -- round the expected value through the same dtype
        # rather than asserting against the un-rounded tag.
        expected_fp8 = torch.tensor(float(expected_tag), dtype=torch.float8_e4m3fn).to(torch.float32)
        torch.testing.assert_close(
            cache.bank_caches["weight_gate_up"][slot].to(torch.float32),
            torch.full_like(cache.bank_caches["weight_gate_up"][slot].to(torch.float32), expected_fp8.item()),
        )
        torch.testing.assert_close(
            cache.bank_caches["scale_down"][slot],
            torch.full_like(cache.bank_caches["scale_down"][slot], float(expected_tag)),
        )


def test_rebuild_resizes_fp8_schema_pool():
    cache = OffloadMoeCache(L, E, S, DEVICE, quant_format="fp8_block")
    cache.set_bank_sources(_packed_bank_sources())
    cache.materialize_layer(0)
    cache.copy_missing()

    cache.rebuild(2 * S)
    for name in _BANK_SCHEMAS["fp8_block"]:
        assert cache.bank_caches[name].shape[0] == 2 * S
    assert int(cache.slot_for_id.min().item()) == -1 or (cache.slot_for_id == -1).all()


def test_bf16_and_gptq_schemas_unaffected_by_fp8_registration():
    """Strictly-additive check: the existing schemas still behave exactly as
    they did before this change."""
    assert "bf16" in _BANK_SCHEMAS and _BANK_SCHEMAS["bf16"] == ("gate_up", "down")
    assert "gptq_int4" in _BANK_SCHEMAS
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
