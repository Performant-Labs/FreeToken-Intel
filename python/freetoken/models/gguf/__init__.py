"""Model package stub: gguf.

Upstream NVIDIA path: python/freetoken/models/gguf/
Fill in: GitHub issue `models-loader` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


def parse_config(*args, **kwargs):
    unimplemented("parse_config", "models-loader")
def iter_weights(*args, **kwargs):
    unimplemented("iter_weights", "models-loader")
class GgufModel:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(self, *args, **kwargs):
        unimplemented("GgufModel.forward", "models-loader")

