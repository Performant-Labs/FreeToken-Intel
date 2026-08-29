"""Availability probes for optional Intel kernel packages.

Upstream checks flashinfer / sgl_kernel. Here we probe Triton-Intel, IPEX, and
the Intel oneAPI DPC++ toolkit (the ``icpx`` SYCL compiler) that the kernel
toolchain (``freetoken.kernel._toolchain``) locates and runs.
"""
from __future__ import annotations

import functools
import importlib.util
import os
import shutil


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
def is_oneapi_dpcpp_installed() -> bool:
    """True when the Intel oneAPI DPC++ toolkit (``icpx``) is on the box.

    ``icpx`` is the C++/SYCL compiler the kernel toolchain shells out to. It is
    found on ``PATH`` (after ``setvars.sh``) or under the ``ONEAPI_ROOT`` /
    ``CMPLR_ROOT`` roots those scripts export. Importing this never imports
    torch, so it is safe to call from a CPU-only process.
    """
    if shutil.which("icpx") is not None:
        return True
    for root in filter(None, (os.environ.get("ONEAPI_ROOT"), os.environ.get("CMPLR_ROOT"))):
        for candidate in (
            os.path.join(root, "bin", "icpx"),
            os.path.join(root, "compiler", "bin", "icpx"),
        ):
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return True
    return False


@functools.cache
def is_sycl_ext_installed() -> bool:
    """Deprecated alias for :func:`is_oneapi_dpcpp_installed`.

    The old name implied a Python package called ``sycl_ext`` (there is none --
    the oneAPI XPU extensions are a *C++* namespace, ``sycl::ext::oneapi``, not
    an importable module). It kept probing a non-existent module, so it was
    always False. It now reports whether the real toolchain is installed.
    """
    return is_oneapi_dpcpp_installed()


@functools.cache
def level_zero_driver_version() -> str | None:
    try:
        from freetoken.kernel.pinned import driver_version

        return driver_version()
    except Exception:
        return None
