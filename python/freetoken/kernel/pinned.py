"""USM-host / pinned tensors for PCIe expert streaming.

Upstream NVIDIA path: python/freetoken/kernel/pinned.py
Fill in: GitHub issue `moe-offload` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


def alloc_pinned(*args, **kwargs):
    unimplemented("alloc_pinned", "moe-offload")
def driver_version(*args, **kwargs):
    unimplemented("driver_version", "device-layer")

