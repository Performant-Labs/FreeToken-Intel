"""Layer stub: moe.

Upstream NVIDIA path: python/freetoken/layers/moe.py
Fill in: GitHub issue `moe-fused` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


class Moe:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(self, *args, **kwargs):
        unimplemented("Moe.forward", "moe-fused")

