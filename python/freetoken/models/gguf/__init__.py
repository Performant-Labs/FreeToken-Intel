"""Model package stub: gguf.

Upstream NVIDIA path: python/freetoken/models/gguf/
Parent epic: `models-gguf` (#199, see docs/architecture.md).

``reader.py`` (issue `models-gguf-reader`, #270) is implemented: GGUF binary
format parsing (header, typed KV-metadata, tensor-info) is re-exported below.
``dequant.py`` (issue `models-gguf-dequant`, #271) is also implemented: GGML
quant-type dequantization (``Q4_0``/``Q8_0``/``Q4_K``/``Q6_K``). Unlike
``reader.py``, ``dequant.py`` requires torch, so it is *not* imported here at
package-import time (that would break this package's own torch-free
CPU-CLI-smoke import path -- see ``reader.py``'s docstring); instead
``dequantize``/``DequantError``/``GGML_NAME_TO_TYPE`` are re-exported lazily
below via module ``__getattr__`` (PEP 562), so ``import freetoken.models.gguf``
stays torch-free but ``freetoken.models.gguf.dequantize(...)`` still works,
importing torch only at that first access.
``parse_config`` / ``iter_weights`` / ``GgufModel`` remain stubs -- they are
`models-gguf-config-tokenizer` (#272) and `models-gguf-iter-and-loader-wiring`
(#273)'s jobs respectively, both of which build directly on this module's
reader.
"""
from __future__ import annotations

from freetoken._stub import unimplemented

from .reader import (
    GGUF_MAGIC,
    GGML_TYPE_NAMES,
    GGUFFile,
    GGUFFormatError,
    GGUFTensorInfo,
    GGUFValueType,
    gguf_tensor_info,
    gguf_tensor_names,
    is_gguf_path,
    load_gguf,
    load_gguf_metadata,
)

__all__ = [
    "GGUF_MAGIC",
    "GGML_TYPE_NAMES",
    "GGML_NAME_TO_TYPE",
    "GGUFFile",
    "GGUFFormatError",
    "GGUFTensorInfo",
    "GGUFValueType",
    "DequantError",
    "dequantize",
    "gguf_tensor_info",
    "gguf_tensor_names",
    "is_gguf_path",
    "load_gguf",
    "load_gguf_metadata",
    "parse_config",
    "iter_weights",
    "GgufModel",
]


_DEQUANT_LAZY_NAMES = {"dequantize", "DequantError", "GGML_NAME_TO_TYPE"}


def __getattr__(name: str):
    # PEP 562 lazy attribute access: defers importing `dequant.py` (and
    # thus torch) until a caller actually touches one of its names, so
    # `import freetoken.models.gguf` alone stays torch-free.
    if name in _DEQUANT_LAZY_NAMES:
        from . import dequant

        return getattr(dequant, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def parse_config(*args, **kwargs):
    unimplemented("parse_config", "models-gguf-config-tokenizer")


def iter_weights(*args, **kwargs):
    unimplemented("iter_weights", "models-gguf-iter-and-loader-wiring")


class GgufModel:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(self, *args, **kwargs):
        unimplemented("GgufModel.forward", "models-gguf-iter-and-loader-wiring")
