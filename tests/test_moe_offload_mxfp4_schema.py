"""Tests for the ``mxfp4`` offload-cache bank schema (issue
moe-quant-banks-mxfp4, #153).

Companion to test_moe_offload_gptq_schema.py (the ``gptq_int4`` schema's own
tests) -- small synthetic packed-uint8 fixtures, no real checkpoint, no XPU.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from freetoken.moe.offload_cache import _BANK_SCHEMAS, OffloadMoeCache

DEVICE = torch.device("cpu")
L, E, S = 2, 4, 8  # 2 layers, 4 experts, 8 slots
HIDDEN, INTER = 64, 32  # both multiples of 32, the MXFP4 block size


def _packed_bank_sources():
    """Distinguishable per-(layer, expert) packed rows for every mxfp4 bank
    -- values chosen so a test can tell which (layer, expert) a slot
    currently holds, same spirit as test_moe_offload_gptq_schema.py's own
    ``_packed_bank_sources``."""

    def tag(layer, expert):
        return 100 * (layer + 1) + 10 * (expert + 1)

    sources = {name: [] for name in _BANK_SCHEMAS["mxfp4"]}
    for layer in range(L):
        per_bank_layers = {name: [] for name in sources}
        for expert in range(E):
            t = tag(layer, expert)
            for proj, out_features, k in (("gate_up", 2 * INTER, HIDDEN), ("down", HIDDEN, INTER)):
                n_blocks = k // 32
                per_bank_layers[f"blocks_{proj}"].append(
                    torch.full((out_features, n_blocks, 16), t % 256, dtype=torch.uint8)
                )
                per_bank_layers[f"scales_{proj}"].append(
                    torch.full((out_features, n_blocks), t % 256, dtype=torch.uint8)
                )
        for name in sources:
            sources[name].append(torch.stack(per_bank_layers[name]))
    return sources


def test_schema_registered():
    assert "mxfp4" in _BANK_SCHEMAS
    assert _BANK_SCHEMAS["mxfp4"] == (
        "blocks_gate_up",
        "scales_gate_up",
        "blocks_down",
        "scales_down",
    )


def test_set_bank_sources_allocates_correctly_shaped_dtyped_device_caches():
    cache = OffloadMoeCache(L, E, S, DEVICE, quant_format="mxfp4")
    sources = _packed_bank_sources()
    cache.set_bank_sources(sources)

    for name in _BANK_SCHEMAS["mxfp4"]:
        host_row_shape = sources[name][0].shape[1:]
        dev_cache = cache.bank_caches[name]
        assert dev_cache.shape == (S, *host_row_shape)
        assert dev_cache.dtype == sources[name][0].dtype == torch.uint8


def test_materialize_and_copy_missing_moves_real_packed_bytes():
    cache = OffloadMoeCache(L, E, S, DEVICE, quant_format="mxfp4")
    sources = _packed_bank_sources()
    cache.set_bank_sources(sources)

    cache.materialize_layer(0)
    cache.copy_missing()

    for expert in range(E):
        slot = int(cache.slot_for_id[0, expert].item())
        assert slot != -1
        expected_tag = (100 * 1 + 10 * (expert + 1)) % 256
        torch.testing.assert_close(
            cache.bank_caches["blocks_gate_up"][slot],
            torch.full_like(cache.bank_caches["blocks_gate_up"][slot], expected_tag),
        )
        torch.testing.assert_close(
            cache.bank_caches["scales_down"][slot],
            torch.full_like(cache.bank_caches["scales_down"][slot], expected_tag),
        )


def test_rebuild_resizes_mxfp4_schema_pool():
    cache = OffloadMoeCache(L, E, S, DEVICE, quant_format="mxfp4")
    cache.set_bank_sources(_packed_bank_sources())
    cache.materialize_layer(0)
    cache.copy_missing()

    cache.rebuild(2 * S)
    for name in _BANK_SCHEMAS["mxfp4"]:
        assert cache.bank_caches[name].shape[0] == 2 * S
    assert (cache.slot_for_id == -1).all()


def test_bf16_schema_unaffected_by_mxfp4_registration():
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
