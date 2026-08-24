"""Layer stub: linear.

Upstream NVIDIA path: python/freetoken/layers/linear.py
Fill in: GitHub issue `quant-xpu` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


class Linear:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(self, *args, **kwargs):
        unimplemented("Linear.forward", "quant-xpu")

