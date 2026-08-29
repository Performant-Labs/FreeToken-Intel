"""Locate the Intel oneAPI DPC++ toolchain (icpx) and its SYCL flags.

Upstream NVIDIA path: the same filename in the reference repo locates the
NVIDIA GPU compiler and checks it matches torch's GPU build. The Intel equivalent is the
oneAPI DPC++ compiler ``icpx`` and the ``-fsycl`` flags that build a Level Zero
program. Filled in for GitHub issue ``kernel-sycl`` (see docs/architecture.md).

Everything here is import-safe on a CPU-only box: locating ``icpx`` shells out
to nothing and reads only the environment (the variables ``setvars.sh`` exports).
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess


class ToolchainError(RuntimeError):
    """Raised when the oneAPI toolchain needed to build a SYCL kernel is absent.

    Distinguished from a generic RuntimeError so a caller can tell "you must
    install oneAPI" apart from "the compile itself failed".
    """


def _candidate_roots() -> list[str]:
    """Roots under which oneAPI installs its component ``bin`` directories.

    ``setvars.sh`` exports ``CMPLR_ROOT`` (e.g. ``/opt/intel/oneapi/compiler/
    2026.1``) and ``ONEAPI_ROOT`` (e.g. ``/opt/intel/oneapi``); both are searched.
    """
    roots: list[str] = []
    for env in ("CMPLR_ROOT", "ONEAPI_ROOT"):
        value = os.environ.get(env)
        if value:
            roots.append(value)
    return roots


def find_icpx() -> pathlib.Path:
    """Return the path to the ``icpx`` DPC++ compiler, or raise ToolchainError.

    Resolution order: an explicit ``FREETOKEN_ICPX`` override, then ``icpx`` on
    ``PATH`` (after ``source setvars.sh``), then the ``bin`` directories under
    ``CMPLR_ROOT`` / ``ONEAPI_ROOT``. The search never hard-codes an install
    prefix, so it works whether oneAPI lives in ``/opt/intel`` or a relocatable
    path.
    """
    override = os.environ.get("FREETOKEN_ICPX")
    if override and os.path.isfile(override) and os.access(override, os.X_OK):
        return pathlib.Path(override)

    on_path = shutil.which("icpx")
    if on_path:
        return pathlib.Path(on_path)

    searched: list[str] = []
    for root in _candidate_roots():
        for bin_dir in (
            pathlib.Path(root) / "bin",
            pathlib.Path(root) / "compiler" / "bin",
        ):
            searched.append(str(bin_dir))
            icpx = bin_dir / "icpx"
            if icpx.is_file() and icpx.access(os.X_OK):
                return icpx
    raise ToolchainError(
        "icpx (Intel oneAPI DPC++ compiler) not found. Install the oneAPI Base "
        "Toolkit and run 'source /opt/intel/oneapi/setvars.sh' before invoking "
        f"the kernel toolchain, or set FREETOKEN_ICPX to the compiler. Searched "
        f"PATH and: {searched}"
    )


def oneapi_include_dir() -> pathlib.Path:
    """Directory holding ``sycl/sycl.hpp`` (passed to icpx as ``-I``).

    ``icpx -fsycl` already puts the SYCL headers on the include path, so this is
    belt-and-braces: it lets a build that passes explicit ``-I`` (the AOT
    builder does) resolve ``#include <sycl/sycl.hpp>`` unambiguously.
    """
    icpx = find_icpx()
    compiler_root = icpx.parent.parent  # .../compiler/<ver>/bin -> .../compiler/<ver>
    include = compiler_root / "include"
    if (include / "sycl" / "sycl.hpp").is_file():
        return include
    # Fall back to a search under the compiler root (layout varies by release).
    for hpp in compiler_root.glob("**/sycl/sycl.hpp"):
        return hpp.parent.parent
    raise ToolchainError(
        f"could not locate the SYCL headers under {compiler_root} "
        "(expected include/sycl/sycl.hpp)"
    )


def _level_zero_lib_dir() -> pathlib.Path | None:
    """Directory holding ``libze_loader.so`` (the Level Zero loader)."""
    for d in (
        pathlib.Path("/usr/lib/x86_64-linux-gnu"),
        pathlib.Path("/usr/lib64"),
        pathlib.Path("/usr/lib"),
        pathlib.Path("/usr/local/lib"),
    ):
        if (d / "libze_loader.so").is_file() or any(d.glob("libze_loader.so*")):
            return d
    return None


def sycl_flags() -> list[str]:
    """The ``icpx`` flags that compile+link a Level Zero SYCL program.

    ``-fsycl`` both compiles the SYCL frontend and links the SYCL runtime, which
    in turn links the Level Zero loader -- so a program built with these flags
    binds the XPU backend rather than compiling against a host/fake SYCL. The
    Level Zero API headers (``ze_api.h``) live in a system include dir, so they
    are added explicitly for builds that pass an explicit include set.
    """
    find_icpx()  # fail fast (with the helpful message) if there is no toolchain
    flags = ["-fsycl", "-std=c++20"]
    flags += [f"-I{oneapi_include_dir()}"]
    lib_dir = _level_zero_lib_dir()
    if lib_dir is not None:
        flags += [f"-L{lib_dir}", "-lze_loader"]
    # The system include dir holding the Level Zero API headers, if present.
    for inc in ("/usr/include/level_zero",):
        if pathlib.Path(inc, "ze_api.h").is_file():
            flags += [f"-I{inc}"]
    return flags


def icpx_version() -> str | None:
    """The oneAPI DPC++ compiler version string (for the AOT cache key).

    ``None`` when the toolchain is unavailable -- the cache-key builder treats
    that as "cannot build" and the caller reports a clean toolchain error.
    """
    try:
        icpx = find_icpx()
    except ToolchainError:
        return None
    try:
        out = subprocess.run(
            [str(icpx), "--version"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.stdout.splitlines():
        # The first line is the banner, e.g.
        # "Intel(R) OneAPI DPC++ and C++ Compiler 2026.1.1 (2026.1.1)".
        if "DPC++" in line or "DPC" in line:
            return line.strip()
    return out.stdout.splitlines()[0].strip() if out.stdout.strip() else None
