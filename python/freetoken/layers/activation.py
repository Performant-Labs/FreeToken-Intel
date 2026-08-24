"""Layer stub: activation.

Upstream NVIDIA path: python/freetoken/layers/activation.py
Fill in: GitHub issue `layers` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


class Activation:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(self, *args, **kwargs):
        unimplemented("Activation.forward", "layers")

