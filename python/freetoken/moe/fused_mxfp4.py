"""MXFP4 fused MoE on Xe2 XMX. Replaces NVFP4 CUDA paths.

Upstream NVIDIA path: python/freetoken/moe/fused_nvfp4.py
Fill in: GitHub issue `quant-xpu` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


class FusedMxfp4Moe:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(self, *args, **kwargs):
        unimplemented("FusedMxfp4Moe.forward", "quant-xpu")

