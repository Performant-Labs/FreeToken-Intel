"""CPU and hybrid CPU+XPU MoE backends (q-star overlap policy).

Upstream NVIDIA path: python/freetoken/moe/cpu_offload.py
Fill in: GitHub issue `moe-hybrid` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented
from freetoken.moe.base import BaseMoeBackend


class CpuOffloadMoeBackend(BaseMoeBackend):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(self, *args, **kwargs):
        unimplemented("CpuOffloadMoeBackend.forward", "moe-cpu")

class HybridMoeBackend(BaseMoeBackend):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(self, *args, **kwargs):
        unimplemented("HybridMoeBackend.forward", "moe-hybrid")

