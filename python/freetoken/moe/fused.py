"""Fused MoE on XPU (experts resident in VRAM). Upstream: CUDA fused MoE.

Upstream NVIDIA path: python/freetoken/moe/fused.py
Fill in: GitHub issue `moe-fused` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented
from freetoken.moe.base import BaseMoeBackend


class FusedMoe(BaseMoeBackend):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(self, *args, **kwargs):
        unimplemented("FusedMoe.forward", "moe-fused")

