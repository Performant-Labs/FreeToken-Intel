"""Intel Arc Pro B70 / Xe2 (Battlemage) detection.

Replaces the upstream CUDA device probes (``get_device_capability`` / SM90 /
SM100) with a Level Zero device probe. The upstream ``pinned.driver_version``
read the CUDA UMD version; the Intel equivalent reads the Level Zero GPU driver
version. ``torch.xpu`` exposes that as ``torch.xpu.get_device_capability`` (the
oneAPI runtime version, e.g. ``(1, 14)``).
"""
from __future__ import annotations

from freetoken.utils.logger import init_logger

logger = init_logger(__name__)

# Arc Pro B70: Battlemage G31, Xe2-HPG, 32 Xe-cores, 32 GB GDDR6, 608 GB/s.
B70_VRAM_BYTES = 32 * 1024**3
B70_MEMORY_BANDWIDTH_GBS = 608
B70_XE_CORES = 32
B70_TBP_WATTS = 230


def is_xpu_available() -> bool:
    try:
        import torch

        return bool(getattr(torch, "xpu", None) and torch.xpu.is_available())
    except Exception:
        return False


def level_zero_driver_version() -> str | None:
    """Version string for the Intel Level Zero GPU driver.

    The Intel equivalent of upstream's CUDA UMD version. Modern PyTorch XPU
    builds expose it directly as ``driver_version`` on the device properties
    (e.g. ``"1.14.37020"``); older builds only exposed a capability tuple, so
    fall back to formatting that. ``None`` when no XPU is present or neither
    form is exposed -- callers must treat that as "unknown", never "the driver
    is missing" (the loader may still be fine).
    """
    if not is_xpu_available():
        return None
    try:
        import torch

        props = torch.xpu.get_device_properties(0)
        version = getattr(props, "driver_version", None)
        if version:
            return str(version)
        cap = torch.xpu.get_device_capability(0)
        return f"{cap[0]}.{cap[1]}"
    except Exception:
        return None


def xpu_total_memory() -> int | None:
    """Total VRAM (bytes) on the first XPU. ``None`` when no XPU is present."""
    if not is_xpu_available():
        return None
    try:
        import torch

        return int(torch.xpu.get_device_properties(0).total_memory)
    except Exception:
        return None


def is_xe2_family() -> bool:
    """True when the first XPU looks like Xe2 / Battlemage."""
    name = xpu_device_name()
    if name is None:
        return False
    lowered = name.lower()
    return any(tok in lowered for tok in ("b70", "battlemage", "xe2", "bmg", "arc pro"))


def is_battlemage() -> bool:
    return is_xe2_family()


def xpu_device_name() -> str | None:
    if not is_xpu_available():
        return None
    try:
        import torch

        return torch.xpu.get_device_name(0)
    except Exception:
        return None


def xpu_device_count() -> int:
    if not is_xpu_available():
        return 0
    try:
        import torch

        return int(torch.xpu.device_count())
    except Exception:
        return 0


def device_report_line() -> str:
    """One-line device summary shared by ``ft device`` and the ``ft serve`` spine.

    Single source of truth for the summary string so the two callers can never
    drift apart (see issue ``serve-spine``).
    """
    if not is_xpu_available():
        return "none (torch.xpu unavailable)"
    name = xpu_device_name() or "(unknown)"
    return f"{name}, {xpu_device_count()} device(s)"


def print_device_report(argv: list[str] | None = None) -> int:
    del argv
    print("FreeToken-Intel device report")
    print(f"  torch.xpu available: {is_xpu_available()}")
    print(f"  device count:        {xpu_device_count()}")
    print(f"  device 0 name:       {xpu_device_name() or '(none)'}")
    print(f"  xe2/battlemage:      {is_xe2_family()}")
    print(f"  device line:         {device_report_line()}")
    print(f"  Level Zero driver:   {level_zero_driver_version() or '(not exposed)'}")
    vram = xpu_total_memory()
    print(
        f"  VRAM:                "
        + (f"{vram / 1024**3:.0f} GB" if vram else "(not exposed)")
    )
    print(
        f"  B70 reference spec:  {B70_XE_CORES} Xe-cores, "
        f"{B70_MEMORY_BANDWIDTH_GBS} GB/s, 32 GB VRAM, {B70_TBP_WATTS} W"
    )
    if not is_xpu_available():
        print("  hint: install PyTorch with the XPU index and a oneAPI / Level Zero runtime.")
        return 1
    return 0
