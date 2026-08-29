"""Ahead-of-time SYCL kernel cache (replaces the upstream CUDA cubin cache).

Upstream NVIDIA path: ``python/freetoken/kernel/aot.py``. Filled in for GitHub
issue ``kernel-sycl`` (see docs/architecture.md).

The AOT cache is the compiled counterpart of the JIT path in
``freetoken.kernel.utils``. A kernel's precompiled module is stored as a shared
object under a directory keyed by the *build inputs* (the oneAPI compiler
version, the Level Zero driver version, and the Xe ISA the module was compiled
for). Because a precompiled SYCL module is only loadable by the same toolchain
that produced it (unlike PTX, a precompiled Xe module has no forward-JIT form),
the key must be exact: a mismatched compiler or driver means "rebuild", never
"try the old one and hope".

This module owns the cache *directory layout and hit/miss logic*; the actual
compile is performed by :func:`freetoken.kernel.utils.sycl_jit_compile` (which
shares the same key and dir, so a JIT build populates the cache and a later
process's AOT load is a hit).
"""
from __future__ import annotations

import hashlib
import importlib
import os
import pathlib

from freetoken.kernel._toolchain import ToolchainError, icpx_version

KERNEL_CACHE_PACKAGE = "freetoken_kernel_cache"
KERNEL_CACHE_DIR_ENV = "FREETOKEN_KERNEL_CACHE_DIR"
DISABLE_KERNEL_CACHE_ENV = "FREETOKEN_DISABLE_KERNEL_CACHE"
_DISABLE_TRUE = {"1", "true", "yes", "on"}


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _DISABLE_TRUE


def _oneapi_version() -> str | None:
    """The oneAPI DPC++ compiler version, or None when the toolchain is absent.

    Sourced from the toolchain (``icpx --version``), not a hard-coded string, so
    the cache key tracks whatever toolchain actually built the module.
    """
    return icpx_version()


def _driver_version() -> str | None:
    """The Level Zero driver version, or None when no XPU is present.

    Read through ``freetoken.kernel.backend`` (which imports torch only inside
    the call), so importing this module stays safe on a CPU-only box.
    """
    try:
        from freetoken.kernel.backend import level_zero_driver_version

        return level_zero_driver_version()
    except Exception:
        return None


def _isa_name() -> str | None:
    """The Xe ISA the local XPU reports (e.g. ``bmg``), or None when no XPU.

    The ISA is the third leg of the AOT cache key: a module compiled for one Xe
    generation cannot run on another. ``torch.xpu`` exposes it as the device
    capability; reading it here is best-effort and never raises.
    """
    try:
        import torch

        if not torch.xpu.is_available():
            return None
        caps = torch.xpu.get_device_properties(0).capability
        # torch reports the arch string on some builds, a tuple on others.
        if isinstance(caps, str):
            return caps
        if caps:
            return str(caps[0])
        return None
    except Exception:
        return None


def cache_key(name: str) -> str:
    """A short, stable key identifying (kernel name, toolchain, driver, ISA).

    The full input string is hashed to keep directory names short and free of
    path/whitespace. Any change to a leg (recompile with a new oneAPI, a driver
    update, a different Xe) changes the key, which is exactly what forces a
    rebuild instead of a silent stale, unloadable module.
    """
    oneapi = _oneapi_version() or "icpx-missing"
    driver = _driver_version() or "driver-unknown"
    isa = _isa_name() or "isa-unknown"
    material = f"freetoken__{name}|oneapi={oneapi}|ze={driver}|isa={isa}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    return f"{name}-{digest}"


def aot_cache_dir() -> pathlib.Path | None:
    """Directory that holds the precompiled kernel modules (or None to disable).

    Resolution mirrors the upstream: an explicit ``FREETOKEN_KERNEL_CACHE_DIR``
    override wins; otherwise the ``freetoken-kernel-cache`` package's
    ``get_jit_cache_dir()`` (its installed data dir) is used. When neither is
    available -- or caching is disabled -- the AOT path is a no-op and the JIT
    path in ``utils`` is the only way to build a kernel.
    """
    if _env_enabled(DISABLE_KERNEL_CACHE_ENV):
        return None
    override = os.getenv(KERNEL_CACHE_DIR_ENV)
    if override:
        return pathlib.Path(override).expanduser()
    try:
        package = importlib.import_module(KERNEL_CACHE_PACKAGE)
    except ModuleNotFoundError:
        return None
    getter = getattr(package, "get_jit_cache_dir", None)
    if getter is None:
        return None
    return pathlib.Path(getter()).expanduser()


def build_aot_cache(name: str, source_file: str, output_dir: str | None = None) -> pathlib.Path:
    """Compile ``source_file`` with icpx into the AOT cache and return the .so path.

    The module is written under ``<cache_dir>/<cache_key(name)>/<name>.so``. The
    directory is created on demand; the cache key already encodes the build
    inputs, so a directory existing means "a module for this exact toolchain /
    driver / ISA is present". The build shells out to ``icpx`` (via
    ``freetoken.kernel.utils.sycl_compile_to``), so this requires the toolchain
    and raises :class:`ToolchainError` if it is missing.
    """
    from freetoken.kernel.utils import sycl_compile_to

    target_dir = pathlib.Path(output_dir) if output_dir else (aot_cache_dir() or pathlib.Path.cwd())
    target_dir.mkdir(parents=True, exist_ok=True)
    so_path = target_dir / f"{name}.so"
    sycl_compile_to(source_file, str(so_path))
    return so_path


def load_aot_cache(name: str) -> pathlib.Path | None:
    """Return the cached module for ``name`` if present and current, else None.

    A hit means the cache holds a module built for *this* oneAPI / driver / ISA
    (the key encodes them). If the directory or ``<name>.so`` is absent the
    caller falls through to JIT. This function only inspects the filesystem --
    it never compiles -- so it is cheap and safe to call on every process start.
    """
    cache_dir = aot_cache_dir()
    if cache_dir is None:
        return None
    so_path = cache_dir / cache_key(name) / f"{name}.so"
    return so_path if so_path.is_file() else None
