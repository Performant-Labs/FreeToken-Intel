"""USM-host / pinned tensors for PCIe expert streaming.

Upstream NVIDIA path: python/freetoken/kernel/pinned.py
Fill in: GitHub issue `moe-offload` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


def alloc_pinned(*args, **kwargs):
    unimplemented("alloc_pinned", "moe-offload")


def driver_version() -> str | None:
    """Version string for the Intel Level Zero GPU driver.

    Upstream's ``pinned.driver_version`` read the CUDA UMD version; the Intel
    equivalent is the Level Zero GPU driver version, which ``torch.xpu``
    exposes as the device capability tuple. Read through ``freetoken.utils.arch``
    (no hard torch import here -- this module sits on the MoE offload path and
    must stay importable on a CPU-only box). ``None`` means "not exposed", not
    "driver missing".
    """
    from freetoken.utils.arch import level_zero_driver_version

    return level_zero_driver_version()

