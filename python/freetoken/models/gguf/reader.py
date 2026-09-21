"""GGUF binary format reader + metadata parser.

Upstream NVIDIA path: python/freetoken/models/gguf/
Issue: `models-gguf-reader` (#270, see docs/architecture.md). Parent epic:
`models-gguf` (#199).

GGUF is a single self-describing binary file (llama.cpp's own checkpoint
format), laid out as -- in order --

    magic ("GGUF") + version                          (header)
    tensor_count + metadata_kv_count                   (header, cont'd)
    metadata_kv_count typed key/value pairs             (KV-metadata section)
    tensor_count tensor descriptors                     (tensor-info section)
    padding up to ``general.alignment`` (default 32)
    raw tensor bytes, one after another                 (tensor-data section)

-- nothing here goes through ``config.json`` / ``safetensors.index.json`` the
way every other checkpoint format this port loads does. This module parses
everything up to (but not including) the tensor-data bytes themselves:
reading/dequantizing the tensor data is `models-gguf-dequant` (#271)'s job,
and mapping the parsed KV-metadata into this port's own ``ModelConfig`` is
`models-gguf-config-tokenizer` (#272)'s job. Both depend on this module's
:class:`GGUFFile` / :class:`GGUFTensorInfo` for tensor byte offsets +
quant types + shapes, so they never need to re-parse the file header.

The exact byte layout below (little-endian throughout; the KV-metadata value
types, the STRING/ARRAY encodings, the tensor-info field order, the alignment
padding) was verified against real GGUF files, not just the spec doc -- see
``tests/test_gguf_reader.py``, which parses real llama.cpp CI fixtures
(``ggml-org/models`` on the Hugging Face Hub: ``tinyllamas/stories260K.gguf``,
an F32 checkpoint, and ``tinyllamas/stories15M-q8_0.gguf``, a Q8_0-quantized
one) and cross-checks the parsed KV-metadata and tensor info against both
hand-computed values and the reference ``gguf`` pip package's own
``GGUFReader``.

Pure Python (``struct`` only -- no ``mmap`` needed: the header + KV-metadata
+ tensor-info section this module parses is always small, even for a
many-GB checkpoint, so plain sequential ``file.read()`` calls suffice and
the multi-GB tensor-data section is never touched here). Torch-free at
import time, matching every other module on the CPU-CLI-smoke import path
(see ``checkpoint/convert.py``'s own docstring for the same constraint).
"""
from __future__ import annotations

import glob
import os
import struct
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Dict, List, Tuple

GGUF_MAGIC = b"GGUF"

# Default alignment tensor data is padded to (before the first tensor's
# bytes), unless overridden by the file's own "general.alignment" KV entry.
_DEFAULT_ALIGNMENT = 32


class GGUFValueType:
    """GGUF's typed KV-metadata value-type enum (as stored on disk)."""

    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


# struct format char for every scalar (non-STRING, non-ARRAY) value type.
_SCALAR_FMT = {
    GGUFValueType.UINT8: "B",
    GGUFValueType.INT8: "b",
    GGUFValueType.UINT16: "H",
    GGUFValueType.INT16: "h",
    GGUFValueType.UINT32: "I",
    GGUFValueType.INT32: "i",
    GGUFValueType.FLOAT32: "f",
    GGUFValueType.BOOL: "B",
    GGUFValueType.UINT64: "Q",
    GGUFValueType.INT64: "q",
    GGUFValueType.FLOAT64: "d",
}

# GGML tensor quant-type enum (as stored in the tensor-info section), for
# human-readable diagnostics only -- `models-gguf-dequant` (#271) owns
# actually interpreting these block layouts. Names match ggml's own
# ``ggml_type`` enum (llama.cpp/ggml/include/ggml.h).
GGML_TYPE_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    31: "Q4_0_4_4",
    32: "Q4_0_4_8",
    33: "Q4_0_8_8",
    34: "TQ1_0",
    35: "TQ2_0",
}


class GGUFFormatError(ValueError):
    """Raised when a file's bytes don't match GGUF's expected layout."""


@dataclass(frozen=True)
class GGUFTensorInfo:
    """One tensor's descriptor, as read from the tensor-info section.

    ``offset`` is the *absolute* byte offset into the file where this
    tensor's raw data begins (the on-disk offset is relative to the start
    of the tensor-data section; this class resolves it to an absolute file
    offset at parse time so callers -- #271/#273 -- never need to know
    about alignment padding or re-derive the tensor-data section start).
    """

    name: str
    shape: Tuple[int, ...]
    ggml_type: int
    offset: int

    @property
    def ggml_type_name(self) -> str:
        return GGML_TYPE_NAMES.get(self.ggml_type, f"UNKNOWN_{self.ggml_type}")

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


@dataclass(frozen=True)
class GGUFFile:
    """A fully-parsed GGUF header: everything except the tensor-data bytes."""

    path: str
    version: int
    metadata: Dict[str, Any]
    tensors: Dict[str, GGUFTensorInfo] = field(default_factory=dict)
    tensor_data_start: int = 0

    @property
    def tensor_names(self) -> List[str]:
        return list(self.tensors)


def _read_exact(f: BinaryIO, n: int) -> bytes:
    buf = f.read(n)
    if len(buf) != n:
        raise GGUFFormatError(f"unexpected EOF: wanted {n} bytes, got {len(buf)}")
    return buf


def _read_struct(f: BinaryIO, fmt: str):
    size = struct.calcsize(fmt)
    (value,) = struct.unpack(fmt, _read_exact(f, size))
    return value


def _read_gguf_string(f: BinaryIO) -> str:
    """A GGUF ``string``: uint64 length prefix + raw UTF-8 bytes (no NUL)."""
    n = _read_struct(f, "<Q")
    return _read_exact(f, n).decode("utf-8")


def _read_gguf_value(f: BinaryIO, value_type: int) -> Any:
    if value_type == GGUFValueType.STRING:
        return _read_gguf_string(f)
    if value_type == GGUFValueType.ARRAY:
        element_type = _read_struct(f, "<I")
        length = _read_struct(f, "<Q")
        return [_read_gguf_value(f, element_type) for _ in range(length)]
    fmt = _SCALAR_FMT.get(value_type)
    if fmt is None:
        raise GGUFFormatError(f"unknown GGUF value type {value_type}")
    value = _read_struct(f, "<" + fmt)
    if value_type == GGUFValueType.BOOL:
        value = bool(value)
    return value


def _resolve_gguf_file(path: str) -> str:
    """Resolve a checkpoint path (a ``.gguf`` file, or a directory holding
    exactly one) to the concrete file to parse."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        candidates = sorted(glob.glob(os.path.join(path, "*.gguf")))
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise GGUFFormatError(f"no *.gguf file found in directory: {path}")
        raise GGUFFormatError(
            f"expected exactly one *.gguf file in {path}, found {len(candidates)}: "
            f"{candidates} (multi-shard/split GGUF checkpoints are not supported)"
        )
    raise FileNotFoundError(path)


def is_gguf_path(path: str) -> bool:
    """True if ``path`` (a file or a directory) is a GGUF checkpoint.

    Detected by the file's first 4 magic bytes (``b"GGUF"``), not by
    extension guessing -- matches this port's "verify, don't assume"
    convention (e.g. ``is_ftw_dir``). Never raises: any I/O or format
    error (missing path, empty file, not a directory holding a single
    ``.gguf`` file, ...) is treated as "not a GGUF checkpoint" so callers
    can use this as a cheap format-detection probe, same as ``is_ftw_dir``.
    """
    try:
        gguf_path = _resolve_gguf_file(path)
    except (FileNotFoundError, GGUFFormatError, OSError):
        return False
    try:
        with open(gguf_path, "rb") as f:
            magic = f.read(4)
    except OSError:
        return False
    return magic == GGUF_MAGIC


def load_gguf(path: str) -> GGUFFile:
    """Fully parse a GGUF file's header, KV-metadata, and tensor-info
    sections (everything up to, but not including, the tensor-data bytes).

    This is the one real parse; :func:`load_gguf_metadata`,
    :func:`gguf_tensor_names`, and :func:`gguf_tensor_info` are thin
    wrappers around it for callers that only want one piece.
    """
    gguf_path = _resolve_gguf_file(path)
    with open(gguf_path, "rb") as f:
        magic = _read_exact(f, 4)
        if magic != GGUF_MAGIC:
            raise GGUFFormatError(f"not a GGUF file (bad magic {magic!r}): {gguf_path}")
        version = _read_struct(f, "<I")
        # GGUF v1 stored the two header counts as uint32; v2+ widened them to
        # uint64 (every real-world fixture in the wild is v2 or v3 -- llama.cpp
        # dropped v1 support years ago -- but the field width is cheap to get
        # right either way).
        count_fmt = "<I" if version == 1 else "<Q"
        tensor_count = _read_struct(f, count_fmt)
        kv_count = _read_struct(f, count_fmt)

        metadata: Dict[str, Any] = {}
        for _ in range(kv_count):
            key = _read_gguf_string(f)
            value_type = _read_struct(f, "<I")
            metadata[key] = _read_gguf_value(f, value_type)

        raw_tensors: List[Tuple[str, Tuple[int, ...], int, int]] = []
        for _ in range(tensor_count):
            name = _read_gguf_string(f)
            n_dims = _read_struct(f, "<I")
            shape = tuple(_read_struct(f, "<Q") for _ in range(n_dims))
            ggml_type = _read_struct(f, "<I")
            rel_offset = _read_struct(f, "<Q")
            raw_tensors.append((name, shape, ggml_type, rel_offset))

        header_end = f.tell()

    alignment = metadata.get("general.alignment", _DEFAULT_ALIGNMENT)
    tensor_data_start = ((header_end + alignment - 1) // alignment) * alignment

    tensors = {
        name: GGUFTensorInfo(
            name=name,
            shape=shape,
            ggml_type=ggml_type,
            offset=tensor_data_start + rel_offset,
        )
        for name, shape, ggml_type, rel_offset in raw_tensors
    }

    return GGUFFile(
        path=gguf_path,
        version=version,
        metadata=metadata,
        tensors=tensors,
        tensor_data_start=tensor_data_start,
    )


def load_gguf_metadata(path: str) -> Dict[str, Any]:
    """The parsed KV-metadata store, keyed exactly as the file stores it
    (e.g. ``general.architecture``, ``<arch>.block_count``, ...) -- raw and
    unmapped. `models-gguf-config-tokenizer` (#272) owns translating these
    into this port's own ``ModelConfig``.
    """
    return load_gguf(path).metadata


def gguf_tensor_names(path: str) -> List[str]:
    """Every tensor's name, in on-disk (tensor-info section) order."""
    return load_gguf(path).tensor_names


def gguf_tensor_info(path: str) -> Dict[str, GGUFTensorInfo]:
    """Every tensor's :class:`GGUFTensorInfo` (shape, GGML quant type, and
    absolute file byte offset), keyed by tensor name -- enough for
    `models-gguf-dequant` (#271) / `models-gguf-iter-and-loader-wiring`
    (#273) to read and dequantize tensor data without re-parsing the file.
    """
    return load_gguf(path).tensors
