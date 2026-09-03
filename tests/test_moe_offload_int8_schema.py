"""Tests for the ``int8_channel`` offload-cache bank schema (issue
`moe-quant-banks-int8`, #154).

Companion to test_moe_offload_cache.py / test_moe_offload_gptq_schema.py --
small synthetic per-channel-INT8 fixtures, no real checkpoint, no XPU.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from freetoken.moe.offload_cache import _BANK_SCHEMAS, OffloadMoeCache

DEVICE = torch.device("cpu")
L, E, S = 2, 4, 8  # 2 layers, 4 experts, 8 slots
K, N = 16, 8  # small in/out dims


def _packed_bank_sources():
    """Distinguishable per-(layer, expert) packed rows for every
    int8_channel bank -- values chosen so a test can tell which
    (layer, expert) a slot currently holds."""

    def tag(layer, expert):
        return 100 * (layer + 1) + 10 * (expert + 1)

    sources = {name: [] for name in _BANK_SCHEMAS["int8_channel"]}
    for layer in range(L):
        per_bank_layers = {name: [] for name in sources}
        for expert in range(E):
            t = tag(layer, expert)
            for proj, n, k in (("gate_up", 2 * N, K), ("down", N, K)):
                per_bank_layers[f"weight_{proj}"].append(torch.full((n, k), t % 127, dtype=torch.int8))
                per_bank_layers[f"scale_{proj}"].append(torch.full((n,), float(t), dtype=torch.float32))
        for name in sources:
            sources[name].append(torch.stack(per_bank_layers[name]))
    return sources


def test_schema_registered():
    assert "int8_channel" in _BANK_SCHEMAS
    assert _BANK_SCHEMAS["int8_channel"] == (
        "weight_gate_up",
        "scale_gate_up",
        "weight_down",
        "scale_down",
    )


def test_set_bank_sources_allocates_correctly_shaped_dtyped_device_caches():
    cache = OffloadMoeCache(L, E, S, DEVICE, quant_format="int8_channel")
    sources = _packed_bank_sources()
    cache.set_bank_sources(sources)

    for name in _BANK_SCHEMAS["int8_channel"]:
        host_row_shape = sources[name][0].shape[1:]
        dev_cache = cache.bank_caches[name]
        assert dev_cache.shape == (S, *host_row_shape)
        assert dev_cache.dtype == sources[name][0].dtype
    assert cache.bank_caches["weight_gate_up"].dtype == torch.int8
    assert cache.bank_caches["scale_gate_up"].dtype == torch.float32


def test_materialize_and_copy_missing_moves_real_packed_bytes():
    cache = OffloadMoeCache(L, E, S, DEVICE, quant_format="int8_channel")
    sources = _packed_bank_sources()
    cache.set_bank_sources(sources)

    cache.materialize_layer(0)
    cache.copy_missing()

    for expert in range(E):
        slot = int(cache.slot_for_id[0, expert].item())
        assert slot != -1
        expected_tag = 100 * 1 + 10 * (expert + 1)
        torch.testing.assert_close(
            cache.bank_caches["weight_gate_up"][slot],
            torch.full_like(cache.bank_caches["weight_gate_up"][slot], expected_tag % 127),
        )
        torch.testing.assert_close(
            cache.bank_caches["scale_down"][slot],
            torch.full_like(cache.bank_caches["scale_down"][slot], float(expected_tag)),
        )


def test_rebuild_resizes_int8_schema_pool():
    cache = OffloadMoeCache(L, E, S, DEVICE, quant_format="int8_channel")
    cache.set_bank_sources(_packed_bank_sources())
    cache.materialize_layer(0)
    cache.copy_missing()

    cache.rebuild(2 * S)
    for name in _BANK_SCHEMAS["int8_channel"]:
        assert cache.bank_caches[name].shape[0] == 2 * S
    assert int(cache.slot_for_id.min().item()) == -1 or (cache.slot_for_id == -1).all()


def test_bf16_schema_unaffected_by_int8_registration():
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
