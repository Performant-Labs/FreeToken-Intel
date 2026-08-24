"""Intel Arc Pro B70 / Xe2 (Battlemage) detection.

Replaces NVIDIA SM90/SM100 probes in upstream FreeToken.
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


def print_device_report(argv: list[str] | None = None) -> int:
    del argv
    print("FreeToken-Intel device report")
    print(f"  torch.xpu available: {is_xpu_available()}")
    print(f"  device count:        {xpu_device_count()}")
    print(f"  device 0 name:       {xpu_device_name() or '(none)'}")
    print(f"  xe2/battlemage:      {is_xe2_family()}")
    print(
        f"  B70 reference spec:  {B70_XE_CORES} Xe-cores, "
        f"{B70_MEMORY_BANDWIDTH_GBS} GB/s, 32 GB VRAM, {B70_TBP_WATTS} W"
    )
    if not is_xpu_available():
        print("  hint: install PyTorch with the XPU index and a oneAPI / Level Zero runtime.")
        return 1
    return 0
