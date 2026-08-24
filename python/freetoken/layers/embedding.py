"""Layer stub: embedding.

Upstream NVIDIA path: python/freetoken/layers/embedding.py
Fill in: GitHub issue `layers` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


class Embedding:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(self, *args, **kwargs):
        unimplemented("Embedding.forward", "layers")

