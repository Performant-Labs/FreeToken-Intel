"""Kernel build/load helpers for the Intel SYCL toolchain.

Upstream NVIDIA path: the same filename in the reference repo compiles with
the NVIDIA GPU compiler and loads with tvm-ffi. The Intel equivalent: compile
with ``icpx -fsycl`` (which links the SYCL runtime + Level Zero loader), and
load the resulting shared object with :mod:`ctypes` -- ``tvm_ffi`` is not a
dependency of the XPU venv, and a raw ``extern "C"`` entry point is all the
toolchain smoke test needs.

Filled in for GitHub issue ``kernel-sycl`` (see docs/architecture.md).
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import pathlib
import subprocess
from typing import NamedTuple

from freetoken.kernel._toolchain import ToolchainError, find_icpx, sycl_flags

KERNEL_PATH = pathlib.Path(__file__).parent / "csrc"
HELLO_COPY_SRC = KERNEL_PATH / "sycl" / "hello_copy.cpp"

# The kernel-cache package (built artifact, separate distributable) provides the
# shared cache directory; an explicit env override beats it.
KERNEL_CACHE_DIR_ENV = "FREETOKEN_KERNEL_CACHE_DIR"
JIT_CACHE_DIR_ENV = "FREETOKEN_JIT_CACHE_DIR"


class KernelModule(NamedTuple):
    """A compiled-and-loaded SYCL kernel module.

    ``path`` is the shared object on disk; ``loaded`` is the live ctypes handle.
    ``from_cache`` is True when the module was loaded from the AOT cache (a hit)
    rather than freshly compiled.
    """

    path: pathlib.Path
    loaded: ctypes.CDLL
    from_cache: bool


def _jit_cache_dir() -> pathlib.Path:
    """Directory for JIT-compiled modules (the warm cache the AOT path reads).

    Explicit ``FREETOKEN_JIT_CACHE_DIR`` wins, then the kernel-cache package's
    dir, then a per-user fallback under the system cache dir. Never raises.
    """
    override = os.getenv(JIT_CACHE_DIR_ENV)
    if override:
        return pathlib.Path(override).expanduser()
    override = os.getenv(KERNEL_CACHE_DIR_ENV)
    if override:
        return pathlib.Path(override).expanduser()
    try:
        import freetoken_kernel_cache as cache_pkg

        getter = getattr(cache_pkg, "get_jit_cache_dir", None)
        if getter is not None:
            return pathlib.Path(getter()).expanduser()
    except ModuleNotFoundError:
        pass
    base = pathlib.Path(os.getenv("XDG_CACHE_HOME", "~/.cache")).expanduser()
    return base / "freetoken" / "kernel-jit"


def _source_fingerprint(source_file: str) -> str:
    """A content hash of a kernel source file (so an edit invalidates the cache).

    A missing file hashes as empty -- the caller validates existence first
    (``hello_copy`` raises before the key is built), so this never crashes on a
    not-yet-present source.
    """
    path = pathlib.Path(source_file)
    data = path.read_bytes() if path.is_file() else b""
    return hashlib.sha256(data).hexdigest()[:16]


def _build_key(name: str, source_file: str) -> str:
    """Key a compiled module by (name, source bytes, toolchain, driver, ISA).

    Mirrors ``aot.cache_key`` (adds the source bytes, since a JIT build keys on
    the exact source being compiled). Any change to a leg produces a new
    directory, so a stale module is never loaded.
    """
    from freetoken.kernel.aot import _driver_version, _isa_name

    # The oneapi leg is the toolchain version that actually compiles the module,
    # read here (not via aot) so it is self-sufficient and never the empty
    # string when a compiler is present.
    oneapi = _oneapi_version_for_key()
    driver = _driver_version() or "driver-unknown"
    isa = _isa_name() or "isa-unknown"
    material = (
        f"freetoken__{name}|oneapi={oneapi}|ze={driver}|isa={isa}|"
        f"src={_source_fingerprint(source_file)}"
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    return f"{name}-{digest}"


def _oneapi_version_for_key() -> str:
    """The oneAPI compiler version for the cache key (never the empty string)."""
    from freetoken.kernel._toolchain import icpx_version

    return icpx_version() or "icpx-missing"


def _compile(source_file: str, output_so: str) -> None:
    """Compile a SYCL source to a shared object with icpx (raises ToolchainError on failure)."""
    icpx = find_icpx()
    out = pathlib.Path(output_so)
    # The linker cannot create the output's parent directory; ensure it exists.
    out.parent.mkdir(parents=True, exist_ok=True)
    flags = sycl_flags() + ["-O2", "-shared", "-fPIC"]
    cmd = [str(icpx), *flags, str(source_file), "-o", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ToolchainError(
            f"icpx failed to compile {source_file} (exit {proc.returncode}):\n"
            f"{proc.stderr}"
        )


def sycl_compile_to(source_file: str, output_so: str) -> pathlib.Path:
    """Compile a SYCL source file to a shared object. Used by the AOT builder."""
    _compile(source_file, output_so)
    return pathlib.Path(output_so)


def _load(so_path: pathlib.Path) -> ctypes.CDLL:
    """dlopen a compiled kernel module.

    The SYCL runtime + Level Zero loader must be resolvable at load time. When
    ``LD_LIBRARY_PATH`` does not already cover them (e.g. setvars.sh was not
    sourced in this process), we extend the search with the oneAPI lib dir before
    dlopen. We never modify the *process* environment globally -- ctypes'
    dlopen honors LD_LIBRARY_PATH at the moment of the call.
    """
    _ensure_sycl_lib_path()
    return ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)


def _ensure_sycl_lib_path() -> None:
    """Make the SYCL runtime + Level Zero loader discoverable by dlopen."""
    candidates: list[str] = []
    for env in ("LD_LIBRARY_PATH",):
        value = os.environ.get(env, "")
        candidates.extend(p for p in value.split(os.pathsep) if p)
    # oneAPI runtime (libsycl, libsycl-jit, ...) lives under the compiler lib dir.
    try:
        compiler_root = find_icpx().parent.parent
        candidates.append(str(compiler_root / "lib"))
        candidates.append(str(compiler_root / "opt" / "compiler" / "lib"))
    except ToolchainError:
        pass
    # The Level Zero loader is a system lib.
    for d in ("/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib", "/usr/local/lib"):
        if os.path.isdir(d):
            candidates.append(d)
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    wanted = [c for c in candidates if c and c not in existing.split(os.pathsep)]
    if wanted:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(wanted + ([existing] if existing else []))


def hello_copy(n: int = 1024) -> KernelModule:
    """Compile (or load from the cache) the hello-copy smoke-test kernel.

    Returns the loaded module. A cache hit (a module built for this exact
    toolchain / driver / ISA / source is present) skips the compile entirely,
    which is the issue's "cache hits skip recompile across processes" acceptance
    criterion. The first call in a process compiles; a second process (or a
    second call with the same key) loads the cached .so.
    """
    if not HELLO_COPY_SRC.is_file():
        raise ToolchainError(f"hello_copy source not found at {HELLO_COPY_SRC}")
    name = "hello_copy"
    key = _build_key(name, str(HELLO_COPY_SRC))
    cache_dir = _jit_cache_dir()
    so_path = cache_dir / key / f"{name}.so"

    if so_path.is_file():
        return KernelModule(path=so_path, loaded=_load(so_path), from_cache=True)

    cache_dir.mkdir(parents=True, exist_ok=True)
    _compile(str(HELLO_COPY_SRC), str(so_path))
    return KernelModule(path=so_path, loaded=_load(so_path), from_cache=False)


def run_hello_copy(module: KernelModule, count: int = 16) -> int:
    """Run the hello-copy kernel and return the number of elements it copied.

    Binds the ``extern "C" run_hello_copy(float*, size_t) -> int`` entry point.
    Returns ``count`` on success; raises on a device/runtime error (the C++ side
    throws, which surfaces here as the process aborting -- a CPU-only box has no
    XPU, so call this only after confirming ``torch.xpu.is_available()``).
    """
    fn = module.loaded.run_hello_copy
    fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
    fn.restype = ctypes.c_int
    buf = (ctypes.c_float * count)()
    return int(fn(buf, ctypes.c_size_t(count)))


def get_xpu_stream():
    """Return the default torch.xpu current stream (for enqueuing kernel work).

    Imported lazily so this module stays importable on a CPU-only box. Returns
    None when no XPU is available (the caller decides whether that is an error).
    """
    try:
        import torch

        if not torch.xpu.is_available():
            return None
        return torch.xpu.current_stream()
    except Exception:
        return None


__all__ = [
    "KernelModule",
    "get_xpu_stream",
    "hello_copy",
    "run_hello_copy",
    "sycl_compile_to",
]
