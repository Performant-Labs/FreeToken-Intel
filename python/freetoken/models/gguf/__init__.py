"""Model package stub: gguf.

Upstream NVIDIA path: python/freetoken/models/gguf/
Parent epic: `models-gguf` (#199, see docs/architecture.md).

``reader.py`` (issue `models-gguf-reader`, #270) is implemented: GGUF binary
format parsing (header, typed KV-metadata, tensor-info) is re-exported below.
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
    "GGUFFile",
    "GGUFFormatError",
    "GGUFTensorInfo",
    "GGUFValueType",
    "gguf_tensor_info",
    "gguf_tensor_names",
    "is_gguf_path",
    "load_gguf",
    "load_gguf_metadata",
    "parse_config",
    "iter_weights",
    "GgufModel",
]


def parse_config(*args, **kwargs):
    unimplemented("parse_config", "models-gguf-config-tokenizer")


def iter_weights(*args, **kwargs):
    unimplemented("iter_weights", "models-gguf-iter-and-loader-wiring")


class GgufModel:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(self, *args, **kwargs):
        unimplemented("GgufModel.forward", "models-gguf-iter-and-loader-wiring")
