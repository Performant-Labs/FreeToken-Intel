"""Availability probes for optional Intel kernel packages.

Upstream checks flashinfer / sgl_kernel. Here we probe Triton-Intel, IPEX,
and the in-tree SYCL extension.
"""
from __future__ import annotations

import functools
import importlib.util


def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


@functools.cache
def is_triton_intel_installed() -> bool:
    return _importable("triton")


@functools.cache
def is_ipex_installed() -> bool:
    return _importable("intel_extension_for_pytorch")


@functools.cache
def is_sycl_ext_installed() -> bool:
    return _importable("freetoken.kernel._sycl_ext")


@functools.cache
def level_zero_driver_version() -> str | None:
    try:
        from freetoken.kernel.pinned import driver_version

        return driver_version()
    except Exception:
        return None
