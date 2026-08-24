"""PCIe offload MoE: experts in host RAM, LRU slots on XPU.

Upstream NVIDIA path: python/freetoken/moe/offload.py
Fill in: GitHub issue `moe-offload` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented
from freetoken.moe.base import BaseMoeBackend


class OffloadMoeBackend(BaseMoeBackend):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(self, *args, **kwargs):
        unimplemented("OffloadMoeBackend.forward", "moe-offload")

