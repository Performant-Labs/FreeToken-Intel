"""Tests for lazy, packed GGUF K-quant MoE expert dequantization (issue
`models-gguf-lazy-packed-dequant`, #282).

The problem this issue fixes: ``freetoken.models.gguf.iter_weights``
dequantizes every MoE expert tensor EAGERLY to full bf16 at load time --
for the real target checkpoint (256 experts x 41 layers) that needs ~66GB
of host RAM, defeating the entire point of using a 22.7GB GGUF quant
instead of the 72GB bf16 original (see #199's own "Why" and #282's own
issue body). The fix mirrors the packed-bank precedent this port already
has for GPTQ/FP8/MXFP4/INT8 (#134/#140): keep the raw, packed, quantized
bytes resident in host banks, and dequantize only the currently-resident
LRU slots, lazily, at compute time.

Covers, mirroring #152/#154's own established test pattern for the other
packed formats:

* :func:`freetoken.models.gguf.iter_moe_expert_raw_banks` reads exactly the
  real per-expert packed bytes (byte-identical to a real dequant of the
  same raw bytes, when fed back through :func:`dequantize` directly).
* :func:`freetoken.models.weight.stream_moe_expert_sources_gguf_kquant`
  builds ``[E, ...]`` packed banks -- including the real, verified
  complication #282's own issue body flags: the GGML quant TYPE (hence
  the packed row's byte length) can differ per LAYER (not per expert
  within one tensor -- GGUF's tensor-info section only ever stores one
  ggml_type per tensor), which this port's shared, single-row-shape
  ``OffloadMoeCache`` bank machinery cannot hold directly without padding.
* :class:`freetoken.moe.offload_cache.SlotWeightAccessor`'s
  ``"gguf_kquant"`` branch dequantizes a resident slot correctly (round-trip
  against :func:`dequantize` called directly on the same raw bytes) and to
  the requested (model activation) dtype, not the dequant kernel's own
  bf16 default.

CPU-only, small synthetic fixtures built with the reference ``gguf`` pip
package's own ``gguf.quants.quantize`` (an independent oracle, already a
real runtime dependency of this port -- see ``dequant.py``'s own module
docstring for the established "verify against a real/independent
implementation" convention) -- never the real 22GB checkpoint (that is
issue #282's own real-checkpoint validation step, run manually with careful
RSS monitoring, not part of the CPU test suite).
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gguf_ref = pytest.importorskip("gguf")

from freetoken.models.gguf import GGUFValueType, gguf_expert_row_bytes, iter_moe_expert_raw_banks
from freetoken.models.gguf.dequant import dequantize
from freetoken.models.weight import (
    GgufKQuantDownBank,
    GgufKQuantGateUpBank,
    stream_moe_expert_sources_gguf_kquant,
)
from freetoken.moe.offload_cache import OffloadMoeCache, SlotWeightAccessor

# --------------------------------------------------------------------------
# A tiny synthetic MoE GGUF file builder (mirrors tests/test_gguf_loader_
# wiring.py's own _build_minimal_gguf / _gguf_string helpers -- duplicated
# here rather than imported, matching that file's own precedent of copying
# tests/test_gguf_reader.py's helper locally).
# --------------------------------------------------------------------------
HIDDEN = 8
INTER = 8  # HIDDEN * INTER == 64, a multiple of Q4_0/Q8_0's 32-element block
EXPERTS = 3
LAYERS = 2


def _gguf_string(s: str) -> bytes:
    data = s.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _kv_scalar(key: str, value_type: int, fmt: str, value) -> tuple:
    return (key, value_type, struct.pack("<" + fmt, value))


def _kv_string(key: str, value: str) -> tuple:
    return (key, GGUFValueType.STRING, _gguf_string(value))


def _build_minimal_gguf(*, version: int = 3, kv: list, tensors: list) -> bytes:
    count_fmt = "<I" if version == 1 else "<Q"
    out = bytearray()
    out += b"GGUF"
    out += struct.pack("<I", version)
    out += struct.pack(count_fmt, len(tensors))
    out += struct.pack(count_fmt, len(kv))
    for key, value_type, raw in kv:
        out += _gguf_string(key)
        out += struct.pack("<I", value_type)
        out += raw
    for name, shape, ggml_type, rel_offset in tensors:
        out += _gguf_string(name)
        out += struct.pack("<I", len(shape))
        for d in shape:
            out += struct.pack("<Q", d)
        out += struct.pack("<I", ggml_type)
        out += struct.pack("<Q", rel_offset)
    pad = (-len(out)) % 32
    out += b"\x00" * pad
    return bytes(out)


def _quantize_expert(dense_out_in: np.ndarray, ggml_type: int) -> bytes:
    """Real packed GGML bytes for one expert's ``[out, in]`` dense weight,
    via the reference ``gguf`` package's own quantizer -- ``.flatten()``
    (C-order, last axis fastest) IS the tensor's ne-order flat element
    sequence for any 2-D ``[out, in]`` array (ne-order-reversed == the
    ``[out, in]`` shape itself, so ne[0], the fastest dim, is exactly the
    array's last axis) -- no separate reordering needed.
    """
    flat = dense_out_in.astype(np.float32).flatten()
    qtype = gguf_ref.GGMLQuantizationType(ggml_type)
    packed = gguf_ref.quants.quantize(flat, qtype)
    return bytes(packed.tobytes())


def _build_synthetic_moe_gguf(
    tmp_path,
    *,
    down_ggml_types: list[int],
    gate_ggml_type: int = 2,  # Q4_0
    up_ggml_type: int = 2,  # Q4_0
) -> tuple[str, dict]:
    """A tiny 2-layer, 3-expert synthetic ``qwen3moe`` GGUF file whose expert
    tensors are REAL Q4_0/Q8_0-quantized bytes (not F32) -- ``down_ggml_types``
    is one ggml_type per layer, letting a test deliberately mix quant types
    across layers (the real complication #282's own issue body flags: the
    real target checkpoint's own ``down`` projection mixes Q5_K/Q6_K across
    layers). Returns ``(path, dense_by_layer)`` where ``dense_by_layer`` is
    ``{layer: {"gate": [E arrays], "up": [...], "down": [...]}}``, the exact
    pre-quantization dense values, for a test to build its own expected
    values against.
    """
    assert len(down_ggml_types) == LAYERS

    kv = [
        _kv_string("general.architecture", "qwen3moe"),
        _kv_string("general.name", "tiny-qwen3moe-kquant"),
        _kv_scalar("qwen3moe.context_length", GGUFValueType.UINT32, "I", 128),
        _kv_scalar("qwen3moe.embedding_length", GGUFValueType.UINT32, "I", HIDDEN),
        _kv_scalar("qwen3moe.block_count", GGUFValueType.UINT32, "I", LAYERS),
        _kv_scalar("qwen3moe.feed_forward_length", GGUFValueType.UINT32, "I", HIDDEN),
        _kv_scalar("qwen3moe.attention.head_count", GGUFValueType.UINT32, "I", 2),
        _kv_scalar("qwen3moe.attention.head_count_kv", GGUFValueType.UINT32, "I", 2),
        _kv_scalar("qwen3moe.expert_count", GGUFValueType.UINT32, "I", EXPERTS),
        _kv_scalar("qwen3moe.expert_used_count", GGUFValueType.UINT32, "I", 2),
        _kv_scalar("qwen3moe.expert_feed_forward_length", GGUFValueType.UINT32, "I", INTER),
        _kv_scalar("qwen3moe.vocab_size", GGUFValueType.UINT32, "I", 16),
    ]

    tensors: list[tuple] = []
    data_chunks: list[bytes] = []
    offset = 0
    dense_by_layer: dict = {}

    def add(name: str, shape: tuple, ggml_type: int, payload: bytes):
        nonlocal offset
        tensors.append((name, shape, ggml_type, offset))
        data_chunks.append(payload)
        offset += len(payload)

    rng = np.random.default_rng(1234)
    for layer in range(LAYERS):
        down_type = down_ggml_types[layer]
        gate_dense = [rng.standard_normal((INTER, HIDDEN)).astype(np.float32) for _ in range(EXPERTS)]
        up_dense = [rng.standard_normal((INTER, HIDDEN)).astype(np.float32) for _ in range(EXPERTS)]
        down_dense = [rng.standard_normal((HIDDEN, INTER)).astype(np.float32) for _ in range(EXPERTS)]
        dense_by_layer[layer] = {"gate": gate_dense, "up": up_dense, "down": down_dense}

        gate_raw = b"".join(_quantize_expert(d, gate_ggml_type) for d in gate_dense)
        up_raw = b"".join(_quantize_expert(d, up_ggml_type) for d in up_dense)
        down_raw = b"".join(_quantize_expert(d, down_type) for d in down_dense)

        add(f"blk.{layer}.ffn_gate_exps.weight", (HIDDEN, INTER, EXPERTS), gate_ggml_type, gate_raw)
        add(f"blk.{layer}.ffn_up_exps.weight", (HIDDEN, INTER, EXPERTS), up_ggml_type, up_raw)
        add(f"blk.{layer}.ffn_down_exps.weight", (INTER, HIDDEN, EXPERTS), down_type, down_raw)

    header = _build_minimal_gguf(kv=kv, tensors=tensors)
    path = tmp_path / "tiny-moe-kquant.gguf"
    path.write_bytes(header + b"".join(data_chunks))
    return str(path), dense_by_layer


_Q4_0 = 2
_Q8_0 = 8


# --------------------------------------------------------------------------
# iter_moe_expert_raw_banks: reads exact raw bytes, no dequant.
# --------------------------------------------------------------------------


def test_iter_moe_expert_raw_banks_reads_byte_identical_rows(tmp_path):
    path, dense_by_layer = _build_synthetic_moe_gguf(tmp_path, down_ggml_types=[_Q4_0, _Q4_0])

    seen = {(layer, bank): (raw, ggml_type) for layer, bank, raw, ggml_type in iter_moe_expert_raw_banks(path)}
    assert set(seen) == {(l, b) for l in range(LAYERS) for b in ("gate", "up", "down")}

    for layer in range(LAYERS):
        for bank, dense_list, ggml_type in (
            ("gate", dense_by_layer[layer]["gate"], _Q4_0),
            ("up", dense_by_layer[layer]["up"], _Q4_0),
            ("down", dense_by_layer[layer]["down"], _Q4_0),
        ):
            raw, seen_type = seen[(layer, bank)]
            assert seen_type == ggml_type
            assert raw.shape[0] == EXPERTS
            assert raw.dtype == torch.uint8
            out_shape = dense_list[0].shape  # [out, in]
            for e in range(EXPERTS):
                expected_bytes = _quantize_expert(dense_list[e], ggml_type)
                assert bytes(raw[e].numpy()) == expected_bytes
                # And dequantizing that row directly reproduces the same
                # dequantized tensor dequantize() would produce from the
                # original raw bytes -- a real round-trip through the read
                # path, not just a byte-copy check.
                row_bytes = gguf_expert_row_bytes(ggml_type, out_shape[0] * out_shape[1])
                got = dequantize(ggml_type, bytes(raw[e, :row_bytes].numpy()), out_shape)
                expected = dequantize(ggml_type, expected_bytes, out_shape)
                torch.testing.assert_close(got, expected)


# --------------------------------------------------------------------------
# stream_moe_expert_sources_gguf_kquant: packed bank building + padding.
# --------------------------------------------------------------------------


def test_stream_builds_packed_banks_never_dequantized(tmp_path):
    path, _ = _build_synthetic_moe_gguf(tmp_path, down_ggml_types=[_Q4_0, _Q4_0])
    from types import SimpleNamespace

    config = SimpleNamespace(num_layers=LAYERS, num_experts=EXPERTS)

    gate_up_banks, down_banks = stream_moe_expert_sources_gguf_kquant(path, config)

    assert len(gate_up_banks) == LAYERS
    assert len(down_banks) == LAYERS
    for bank in gate_up_banks:
        assert isinstance(bank, GgufKQuantGateUpBank)
        assert bank.raw_gate.dtype == torch.uint8
        assert bank.raw_up.dtype == torch.uint8
        assert bank.raw_gate.shape[0] == EXPERTS
        assert bank.ggml_type_gate == _Q4_0
        assert bank.ggml_type_up == _Q4_0
    for bank in down_banks:
        assert isinstance(bank, GgufKQuantDownBank)
        assert bank.raw.dtype == torch.uint8
        assert bank.ggml_type == _Q4_0


def test_stream_pads_bank_rows_when_layers_disagree_on_ggml_type(tmp_path):
    """The real, verified complication #282's own issue body flags: the real
    target checkpoint's own ``down`` projection mixes Q5_K (176 bytes/block)
    and Q6_K (210 bytes/block) across layers -- different row byte lengths.
    Mirrored here with Q4_0 (18 bytes/block) vs Q8_0 (34 bytes/block, same
    32-element block size) so the row-byte-length mismatch is real and
    checkable without needing the much larger K-quant super-blocks.
    """
    path, dense_by_layer = _build_synthetic_moe_gguf(tmp_path, down_ggml_types=[_Q4_0, _Q8_0])
    from types import SimpleNamespace

    config = SimpleNamespace(num_layers=LAYERS, num_experts=EXPERTS)

    gate_up_banks, down_banks = stream_moe_expert_sources_gguf_kquant(path, config)

    assert down_banks[0].ggml_type == _Q4_0
    assert down_banks[1].ggml_type == _Q8_0
    # Both layers' raw bank rows must share ONE row-byte length (the padded,
    # uniform shape OffloadMoeCache's single device slot-cache pool needs) --
    # the max of the two real (unpadded) lengths, i.e. Q8_0's (the bigger).
    n_elem = HIDDEN * INTER
    q4_0_bytes = gguf_expert_row_bytes(_Q4_0, n_elem)
    q8_0_bytes = gguf_expert_row_bytes(_Q8_0, n_elem)
    assert q8_0_bytes > q4_0_bytes  # sanity: the mismatch this test exercises is real
    assert down_banks[0].raw.shape[1] == q8_0_bytes
    assert down_banks[1].raw.shape[1] == q8_0_bytes

    # Layer 0's real bytes are the first q4_0_bytes of its (padded) row, with
    # zero padding after -- recoverable exactly via gguf_expert_row_bytes.
    for e in range(EXPERTS):
        real = down_banks[0].raw[e, :q4_0_bytes]
        pad = down_banks[0].raw[e, q4_0_bytes:]
        expected = _quantize_expert(dense_by_layer[0]["down"][e], _Q4_0)
        assert bytes(real.numpy()) == expected
        assert torch.count_nonzero(pad).item() == 0
    # Layer 1 (Q8_0, the max) needs no padding at all.
    for e in range(EXPERTS):
        expected = _quantize_expert(dense_by_layer[1]["down"][e], _Q8_0)
        assert bytes(down_banks[1].raw[e].numpy()) == expected


def test_stream_missing_layer_raises():
    """A checkpoint missing a whole layer's expert tensors must fail loudly,
    matching every other packed-bank streamer's own missing-layer check."""
    import unittest.mock as mock
    from types import SimpleNamespace

    import freetoken.models.weight as weight_mod

    config = SimpleNamespace(num_layers=1, num_experts=2)
    # A GGUF path with no real MoE tensors at all (num_layers=1 expects
    # layer 0's gate/up/down; nothing is yielded) -- exercised directly
    # against a monkeypatched empty generator so this test needs no real
    # GGUF file on disk.
    with mock.patch("freetoken.models.gguf.iter_moe_expert_raw_banks", return_value=iter(())):
        with pytest.raises(ValueError, match="Missing/incomplete GGUF MoE expert bank"):
            weight_mod.stream_moe_expert_sources_gguf_kquant("unused-path", config)


# --------------------------------------------------------------------------
# SlotWeightAccessor: gguf_kquant dequant-at-compute round trip.
# --------------------------------------------------------------------------


def _cache_from_synthetic_gguf(tmp_path, *, down_ggml_types: list[int]):
    from types import SimpleNamespace

    path, dense_by_layer = _build_synthetic_moe_gguf(tmp_path, down_ggml_types=down_ggml_types)
    config = SimpleNamespace(num_layers=LAYERS, num_experts=EXPERTS)
    gate_up_banks, down_banks = stream_moe_expert_sources_gguf_kquant(path, config)

    # cache_size = LAYERS * EXPERTS (not just EXPERTS): the pool is a single
    # GLOBAL LRU shared by every layer, so materializing layer 1 would evict
    # layer 0's just-materialized slots if the pool only fit one layer at a
    # time -- size it to keep every layer resident simultaneously so this
    # test can check every layer after materializing all of them.
    cache = OffloadMoeCache(LAYERS, EXPERTS, LAYERS * EXPERTS, torch.device("cpu"), quant_format="gguf_kquant")
    cache.set_bank_sources(
        {
            "raw_gate": [b.raw_gate for b in gate_up_banks],
            "raw_up": [b.raw_up for b in gate_up_banks],
            "raw_down": [b.raw for b in down_banks],
        }
    )
    cache.set_extra_metadata("gguf_ggml_type_gate", [b.ggml_type_gate for b in gate_up_banks])
    cache.set_extra_metadata("gguf_ggml_type_up", [b.ggml_type_up for b in gate_up_banks])
    cache.set_extra_metadata("gguf_ggml_type_down", [b.ggml_type for b in down_banks])
    cache.gguf_hidden_size = HIDDEN
    cache.gguf_moe_intermediate_size = INTER
    for layer in range(LAYERS):
        cache.materialize_layer(layer)
        cache.copy_missing()
    return cache, dense_by_layer


def test_slot_weight_accessor_get_matches_direct_dequant(tmp_path):
    cache, _ = _cache_from_synthetic_gguf(tmp_path, down_ggml_types=[_Q4_0, _Q8_0])

    for layer in range(LAYERS):
        gate_type = cache.get_extra_metadata("gguf_ggml_type_gate", layer)
        up_type = cache.get_extra_metadata("gguf_ggml_type_up", layer)
        down_type = cache.get_extra_metadata("gguf_ggml_type_down", layer)
        accessor = SlotWeightAccessor(cache, intermediate=INTER, dtype=torch.float32, layer_id=layer)
        for e in range(EXPERTS):
            slot = int(cache.slot_for_id[layer, e].item())
            gate_w, up_w, down_w = accessor.get(slot)

            gate_raw = cache.bank_caches["raw_gate"][slot]
            up_raw = cache.bank_caches["raw_up"][slot]
            down_raw = cache.bank_caches["raw_down"][slot]
            gate_bytes = gguf_expert_row_bytes(gate_type, INTER * HIDDEN)
            up_bytes = gguf_expert_row_bytes(up_type, INTER * HIDDEN)
            down_bytes = gguf_expert_row_bytes(down_type, HIDDEN * INTER)
            expected_gate = dequantize(gate_type, bytes(gate_raw[:gate_bytes].numpy()), (INTER, HIDDEN), out_dtype=torch.float32)
            expected_up = dequantize(up_type, bytes(up_raw[:up_bytes].numpy()), (INTER, HIDDEN), out_dtype=torch.float32)
            expected_down = dequantize(down_type, bytes(down_raw[:down_bytes].numpy()), (HIDDEN, INTER), out_dtype=torch.float32)

            torch.testing.assert_close(gate_w, expected_gate)
            torch.testing.assert_close(up_w, expected_up)
            torch.testing.assert_close(down_w, expected_down)


def test_slot_weight_accessor_dtype_matches_requested_not_kernel_default(tmp_path):
    """The gptq_int4/fp8_block dtype bug class (#138): the accessor must
    dequantize to the REQUESTED dtype, never dequant.py's own bf16 default."""
    cache, _ = _cache_from_synthetic_gguf(tmp_path, down_ggml_types=[_Q4_0, _Q4_0])
    accessor = SlotWeightAccessor(cache, intermediate=INTER, dtype=torch.float32, layer_id=0)
    slot = int(cache.slot_for_id[0, 0].item())
    gate_w, up_w, down_w = accessor.get(slot)
    assert gate_w.dtype == torch.float32
    assert up_w.dtype == torch.float32
    assert down_w.dtype == torch.float32


def test_slot_weight_accessor_caches_per_slot_within_one_instance(tmp_path):
    cache, _ = _cache_from_synthetic_gguf(tmp_path, down_ggml_types=[_Q4_0, _Q4_0])
    accessor = SlotWeightAccessor(cache, intermediate=INTER, dtype=torch.float32, layer_id=0)
    slot = int(cache.slot_for_id[0, 0].item())
    first = accessor.get(slot)
    second = accessor.get(slot)
    for a, b in zip(first, second):
        assert a.data_ptr() == b.data_ptr()


def test_slot_weight_accessor_requires_layer_id():
    cache = OffloadMoeCache(1, 2, 2, torch.device("cpu"), quant_format="gguf_kquant")
    cache.set_bank_sources(
        {
            "raw_gate": [torch.zeros(2, 4, dtype=torch.uint8)],
            "raw_up": [torch.zeros(2, 4, dtype=torch.uint8)],
            "raw_down": [torch.zeros(2, 4, dtype=torch.uint8)],
        }
    )
    cache.set_extra_metadata("gguf_ggml_type_gate", [0])
    cache.set_extra_metadata("gguf_ggml_type_up", [0])
    cache.set_extra_metadata("gguf_ggml_type_down", [0])
    cache.gguf_hidden_size = HIDDEN
    cache.gguf_moe_intermediate_size = INTER
    with pytest.raises(ValueError, match="layer_id"):
        SlotWeightAccessor(cache, intermediate=INTER, dtype=torch.float32)


def test_slot_weight_accessor_requires_hidden_and_intermediate_size():
    cache = OffloadMoeCache(1, 2, 2, torch.device("cpu"), quant_format="gguf_kquant")
    cache.set_bank_sources(
        {
            "raw_gate": [torch.zeros(2, 4, dtype=torch.uint8)],
            "raw_up": [torch.zeros(2, 4, dtype=torch.uint8)],
            "raw_down": [torch.zeros(2, 4, dtype=torch.uint8)],
        }
    )
    cache.set_extra_metadata("gguf_ggml_type_gate", [0])
    cache.set_extra_metadata("gguf_ggml_type_up", [0])
    cache.set_extra_metadata("gguf_ggml_type_down", [0])
    with pytest.raises(ValueError, match="gguf_hidden_size"):
        SlotWeightAccessor(cache, intermediate=INTER, dtype=torch.float32, layer_id=0)
