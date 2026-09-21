"""GGML quant-type dequantization kernels (issue `models-gguf-dequant`, #271).

Upstream NVIDIA path: python/freetoken/models/gguf/
Parent epic: `models-gguf` (#199, see docs/architecture.md). Depends on
`models-gguf-reader` (#270): :class:`~freetoken.models.gguf.reader.GGUFTensorInfo`
gives the ``ggml_type``/``shape``/``offset`` this module needs to read and
dequantize a tensor's raw bytes.

GGUF's own quant format zoo (``Q4_0``/``Q8_0`` and the ``Q4_K``/``Q5_K``/``Q6_K``
"K-quant" super-block formats) is a real, different format from every quant
this port already supports (GPTQ/FP8-block/MXFP4/INT8-channel, all
safetensors-based, see ``kernel/triton/int8_linear.py``'s own docstring).
Scope here is narrowed to the block formats the real target checkpoint
(``Qwen3.6-35B-A3B`` ``q4_k_m``, see #199's "Why") actually uses: llama.cpp's
``q4_k_m`` naming mixes a dominant ``Q4_K`` with ``Q6_K`` for a few
sensitive tensors (e.g. the output head) -- plus the simpler legacy
``Q4_0``/``Q8_0`` block formats real GGUF checkpoints (and this module's own
test fixtures) also use for small/non-K-quantizable tensors.

Every block layout below (super-block sizes, the packed 6-bit K-quant scale
encoding, nibble/bit-plane packing order) was read from ``llama.cpp``'s own
``ggml/src/ggml-quants.c`` and cross-checked -- not just "does it run," but
"does it produce the same floats" (#271's own accept bar) -- against the
reference ``gguf`` pip package's ``gguf.quants.dequantize`` (already a real
runtime dependency of this port, see ``pyproject.toml``; used here only as
an independent oracle in tests, never as this module's own implementation,
matching ``reader.py``'s "verify, don't assume, but don't import the
reference's own parser either" discipline) on real ``Q4_K``/``Q6_K``/``Q8_0``
bytes read from a real small GGUF checkpoint file -- see
``tests/test_gguf_dequant.py``.

Dequantizes to ``bf16`` (not the checkpoint's own stored ``f16`` scale
dtype): matches this port's existing dequant-to-activation-dtype convention
(the ``dtype`` bug class issue #138 found for GPTQ -- dequant to the
model's activation dtype, never the checkpoint's own stored scale dtype).

CPU-side (plain numpy for the bit-unpacking, then a single ``torch.tensor``
conversion at the end) -- the floor per #271's own scope; a SYCL/XPU kernel
is a separable follow-up once #274 measures whether CPU dequant is actually
a bottleneck on real hardware. This module is torch-*required* (unlike
``reader.py``): it must be imported after torch is available, but no torch
import happens at ``freetoken.models.gguf`` package-import time (guarded by
the caller only invoking :func:`dequantize`), so the CPU-CLI-smoke /
torch-free ``.venv`` import path for the rest of the ``gguf`` package is
unaffected.
"""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch

from .reader import GGML_TYPE_NAMES

# ggml_type int -> human name, reverse of GGML_TYPE_NAMES (reader.py owns the
# canonical table; this module just needs the reverse lookup for the
# convenience of accepting either an int or a name string in `dequantize`).
GGML_NAME_TO_TYPE = {name: ggml_type for ggml_type, name in GGML_TYPE_NAMES.items()}

# GGUF's own K-quant super-block size (elements per super-block): every
# Q*_K format packs QK_K values into one block, unlike the legacy
# Q4_0/Q8_0 formats' 32-value blocks.
_QK_K = 256


class DequantError(ValueError):
    """Raised for an unsupported ``ggml_type`` or malformed raw bytes."""


def _resolve_ggml_type(quant_type: Union[int, str]) -> int:
    if isinstance(quant_type, str):
        try:
            return GGML_NAME_TO_TYPE[quant_type]
        except KeyError:
            raise DequantError(f"unknown GGML quant-type name: {quant_type!r}") from None
    return quant_type


def _as_block_bytes(raw_bytes: bytes, type_size: int, ggml_type_name: str) -> np.ndarray:
    if len(raw_bytes) % type_size != 0:
        raise DequantError(
            f"{ggml_type_name}: raw byte length ({len(raw_bytes)}) is not a "
            f"multiple of its block size in bytes ({type_size})"
        )
    n_blocks = len(raw_bytes) // type_size
    return np.frombuffer(raw_bytes, dtype=np.uint8).reshape(n_blocks, type_size)


# --------------------------------------------------------------------------
# Per-format dequantization: raw block bytes (n_blocks, type_size) -> float32
# (n_blocks, block_size). Bit layouts per llama.cpp's ggml-quants.c.
# --------------------------------------------------------------------------


def _dequantize_f32(raw_bytes: bytes) -> np.ndarray:
    return np.frombuffer(raw_bytes, dtype="<f4").astype(np.float32)


def _dequantize_f16(raw_bytes: bytes) -> np.ndarray:
    return np.frombuffer(raw_bytes, dtype="<f2").astype(np.float32)


def _dequantize_q4_0(raw_bytes: bytes) -> np.ndarray:
    # Block: 2-byte f16 scale `d` + 16 bytes of 32 packed 4-bit values
    # (nibble-low = elements [0:16], nibble-high = elements [16:32]).
    # value = (nibble - 8) * d
    block_size, type_size = 32, 18
    blocks = _as_block_bytes(raw_bytes, type_size, "Q4_0")
    n_blocks = blocks.shape[0]

    d = blocks[:, :2].view("<f2").astype(np.float32)  # (n_blocks, 1)
    qs = blocks[:, 2:]  # (n_blocks, 16)

    lo = (qs & 0x0F).astype(np.int8) - 8
    hi = ((qs >> 4) & 0x0F).astype(np.int8) - 8
    q = np.concatenate([lo, hi], axis=1).astype(np.float32)  # (n_blocks, 32)

    return (d * q).reshape(n_blocks, block_size)


def _dequantize_q8_0(raw_bytes: bytes) -> np.ndarray:
    # Block: 2-byte f16 scale `d` + 32 raw int8 values. value = qs * d
    block_size, type_size = 32, 34
    blocks = _as_block_bytes(raw_bytes, type_size, "Q8_0")
    n_blocks = blocks.shape[0]

    d = blocks[:, :2].view("<f2").astype(np.float32)  # (n_blocks, 1)
    qs = blocks[:, 2:].view(np.int8).astype(np.float32)  # (n_blocks, 32)

    return (d * qs).reshape(n_blocks, block_size)


def _unpack_q4_k_scale_min(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpack Q4_K/Q5_K's 12-byte packed 6-bit scale+min pair table into
    (sc, m), each ``(n_blocks, 8)`` uint8 arrays of 8 sub-block 6-bit values.

    The 12 bytes hold 8 six-bit scales and 8 six-bit mins (96 bits total),
    packed as: byte[0:4] = low 6 bits of scale[0:4] in bits [5:0], with
    bits [7:6] borrowed from the *high* 2 bits of min[0:4]; byte[4:8] = low
    6 bits of min[0:4] (bits [5:0]) with bits [7:6] = high 2 bits of
    scale[4:8]; byte[8:12] = low 4 bits of scale[4:8] in [3:0] and low 4
    bits of min[4:8] in [7:4]. (Matches ggml-quants.c's `get_scale_min_k4`.)
    """
    n_blocks = scales.shape[0]
    sc = np.empty((n_blocks, 8), dtype=np.uint8)
    m = np.empty((n_blocks, 8), dtype=np.uint8)

    b0_3 = scales[:, 0:4]
    b4_7 = scales[:, 4:8]
    b8_11 = scales[:, 8:12]

    sc[:, 0:4] = b0_3 & 0x3F
    m[:, 0:4] = b4_7 & 0x3F
    sc[:, 4:8] = (b8_11 & 0x0F) | ((b0_3 >> 6) << 4)
    m[:, 4:8] = (b8_11 >> 4) | ((b4_7 >> 6) << 4)
    return sc, m


def _dequantize_q4_k(raw_bytes: bytes) -> np.ndarray:
    # Super-block (256 values): 2B f16 `d` + 2B f16 `dmin` + 12B packed
    # 6-bit (scale, min) pairs for 8 32-value sub-blocks + 128B of 256
    # packed 4-bit values, laid out as 4 byte-groups of 32 bytes each --
    # byte-group g's low nibbles are sub-block 2g, its high nibbles are
    # sub-block 2g+1 (each sub-block 32 values wide).
    # value = d*sc[j]*q - dmin*m[j] for sub-block j.
    block_size, type_size = _QK_K, 144
    blocks = _as_block_bytes(raw_bytes, type_size, "Q4_K")
    n_blocks = blocks.shape[0]

    d = blocks[:, 0:2].view("<f2").astype(np.float32)  # (n_blocks, 1)
    dmin = blocks[:, 2:4].view("<f2").astype(np.float32)  # (n_blocks, 1)
    scales = blocks[:, 4:16]  # (n_blocks, 12)
    qs = blocks[:, 16:144].reshape(n_blocks, 4, 32)  # 4 byte-groups of 32

    sc, m = _unpack_q4_k_scale_min(scales)  # each (n_blocks, 8)
    dl = (d * sc.astype(np.float32)).reshape(n_blocks, 8, 1)
    ml = (dmin * m.astype(np.float32)).reshape(n_blocks, 8, 1)

    lo = (qs & 0x0F).astype(np.float32)  # (n_blocks, 4, 32): sub-blocks 0,2,4,6
    hi = ((qs >> 4) & 0x0F).astype(np.float32)  # sub-blocks 1,3,5,7
    q = np.stack([lo, hi], axis=2).reshape(n_blocks, 8, 32)

    out = dl * q - ml  # (n_blocks, 8, 32)
    return out.reshape(n_blocks, block_size)


def _dequantize_q6_k(raw_bytes: bytes) -> np.ndarray:
    # Super-block (256 values): 128B `ql` (low 4 bits, 2 vals/byte) + 64B
    # `qh` (high 2 bits, 4 vals/byte) + 16B int8 per-16-value scales + 2B
    # f16 `d`. value = d * scale[j] * ((ql | (qh << 4)) - 32) for group j
    # (16 groups of 16 values each).
    block_size, type_size = _QK_K, 210
    blocks = _as_block_bytes(raw_bytes, type_size, "Q6_K")
    n_blocks = blocks.shape[0]

    ql = blocks[:, 0:128]
    qh = blocks[:, 128:192]
    scales = blocks[:, 192:208].view(np.int8).astype(np.float32)  # (n_blocks, 16)
    d = blocks[:, 208:210].view("<f2").astype(np.float32)  # (n_blocks, 1)

    ql = ql.reshape(n_blocks, 2, 1, 64) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)
    ql = (ql & 0x0F).reshape(n_blocks, 8, 32)

    qh = qh.reshape(n_blocks, 2, 1, 32) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)
    qh = (qh & 0x03).reshape(n_blocks, 8, 32)

    q = (ql | (qh << 4)).astype(np.int16) - 32  # (n_blocks, 8, 32)
    q = q.reshape(n_blocks, 16, 16).astype(np.float32)

    dl = d.reshape(n_blocks, 1, 1) * scales.reshape(n_blocks, 16, 1)  # (n_blocks, 16, 1)
    out = dl * q
    return out.reshape(n_blocks, block_size)


_DEQUANTIZERS = {
    "F32": _dequantize_f32,
    "F16": _dequantize_f16,
    "Q4_0": _dequantize_q4_0,
    "Q8_0": _dequantize_q8_0,
    "Q4_K": _dequantize_q4_k,
    "Q6_K": _dequantize_q6_k,
}


def dequantize(
    quant_type: Union[int, str], raw_bytes: bytes, shape: Sequence[int]
) -> torch.Tensor:
    """Dequantize one tensor's raw GGUF bytes to a dense ``bf16`` tensor.

    ``quant_type`` is either the raw ``ggml_type`` int (as stored in a
    ``GGUFTensorInfo.ggml_type``) or its name (e.g. ``"Q4_K"``, matching
    ``GGML_TYPE_NAMES``/``GGUFTensorInfo.ggml_type_name``). ``shape`` is the
    tensor's element shape (``GGUFTensorInfo.shape``) -- GGUF stores shapes
    fastest-varying-dimension-first (same axis order the reader already
    hands back), so the caller passes it through unchanged; this function
    only reshapes the flat dequantized values into it.

    Dequantizes to ``bf16`` regardless of the checkpoint's own stored scale
    dtype (always ``f16`` for the quant types here), matching this port's
    dequant-to-activation-dtype convention (see this module's docstring).
    """
    ggml_type = _resolve_ggml_type(quant_type)
    type_name = GGML_TYPE_NAMES.get(ggml_type, f"UNKNOWN_{ggml_type}")
    fn = _DEQUANTIZERS.get(type_name)
    if fn is None:
        raise DequantError(
            f"dequantize: unsupported ggml_type {ggml_type} ({type_name}); "
            f"supported types are {sorted(_DEQUANTIZERS)}"
        )

    flat = fn(raw_bytes)

    n_elements = 1
    for d in shape:
        n_elements *= d
    if flat.size < n_elements:
        raise DequantError(
            f"{type_name}: dequantized {flat.size} values, but shape {tuple(shape)} "
            f"needs at least {n_elements}"
        )
    flat = flat[:n_elements]

    tensor = torch.from_numpy(np.ascontiguousarray(flat))
    return tensor.reshape(tuple(shape)).to(torch.bfloat16)
