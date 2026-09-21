"""Compile/load wrapper for the native CPU MoE expert-GEMM kernel (issue #250).

Parent epic #249 / #248 (GIL-free hybrid dispatch): the current pure-Python
hybrid MoE split (`_Qwen3MoE._forward_hybrid` /
`_Qwen3MoE._cpu_subset_math`, and the plain `cpu` backend's
`CpuMoeExecutor.forward`) pays a per-layer `ThreadPoolExecutor` submit/result
round trip that costs orders of magnitude more than the actual expert GEMM
(#247). Fixing that (#248) needs a *native* function to submit/wait against
instead of re-entering the Python interpreter every layer. This module is
step one: a real, loadable, single-threaded, correctness-first C++ port of
the expert GEMM math (`python/freetoken/kernel/csrc/cpu_moe/cpu_moe.cpp`),
with no threading/async of its own -- #248 builds the GIL-free dispatch on
top of the synchronous entry point this module exposes.

Compiler choice -- plain system C++, not icpx
-----------------------------------------------
``kernel/utils.py``'s existing JIT+AOT pattern (issue #3) always compiles
with ``icpx -fsycl`` because every kernel it has built so far links the SYCL
runtime + Level Zero loader to run *on the XPU*. This kernel is different: it
is pure host C++ with zero SYCL/XPU surface (no ``sycl.hpp``, no Level Zero,
no ``-fsycl``) -- it must build and run correctly on a CPU-only box with no
oneAPI install at all, which is exactly the box the hybrid split's CPU half
(and this module's own unit test) needs to work on.

Reusing ``icpx`` *without* ``-fsycl`` was considered (one compiler for the
whole project) but rejected: ``icpx`` is still an oneAPI Base Toolkit
component, so a box with only a plain system toolchain (``g++``/``clang++``,
which is the common case on a CI runner or a dev box that never installed
oneAPI -- this very sandbox is one) would report "no toolchain" for a kernel
that has no actual XPU/SYCL dependency and could otherwise build and pass its
correctness test right now. That would make issue #250's own acceptance bar
("compiles and loads on a box with the toolchain present", "CPU-only unit
test") strictly harder to satisfy than necessary, and would block the
hybrid-split call site's CPU half from working in environments where the XPU
side of the stack is not installed at all. So this module introduces a small,
separate toolchain lookup (:func:`find_cxx_compiler`) for a plain C++
compiler: an explicit ``FREETOKEN_CXX`` override, else the first of
``c++``/``g++``/``clang++`` found on ``PATH``. It deliberately does *not*
fall back to :func:`freetoken.kernel._toolchain.find_icpx` -- if ``icpx`` is
the only compiler present it will also be found here (it accepts plain C++
flags fine), but the lookup itself stays independent of oneAPI so this module
never requires it.

Everything else mirrors ``kernel/utils.py``'s JIT pattern deliberately: a
``KernelModule``-shaped return (reused directly -- same fields, same
meaning), a cache key over (name, source bytes, compiler identity, host
architecture) under the same ``freetoken-kernel-cache`` / ``FREETOKEN_*_DIR``
directory convention (:func:`freetoken.kernel.utils._jit_cache_dir`), and a
clean :class:`~freetoken.kernel._toolchain.ToolchainError` when no compiler is
found -- so a caller (or a test) can catch exactly the same exception type
the SYCL kernels raise and skip gracefully.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import pathlib
import platform
import shutil
import subprocess
import tempfile
from typing import TYPE_CHECKING, Iterable, Optional

from freetoken.kernel._toolchain import ToolchainError
from freetoken.kernel.utils import KernelModule, _jit_cache_dir, _source_fingerprint

if TYPE_CHECKING:
    # torch is an optional extra (the "xpu" install target) -- this module
    # (and the CPU-only toolchain/build-key helpers in it) must stay
    # importable without it, so torch is imported lazily inside the
    # functions that actually need tensors (mirrors kernel/utils.py's
    # get_xpu_stream / kernel/aot.py's _isa_name, which do the same).
    import torch

KERNEL_PATH = pathlib.Path(__file__).parent / "csrc"
CPU_MOE_SRC = KERNEL_PATH / "cpu_moe" / "cpu_moe.cpp"

CXX_ENV_OVERRIDE = "FREETOKEN_CXX"
_CXX_CANDIDATES: tuple[str, ...] = ("c++", "g++", "clang++")


# --- Toolchain (plain system C++, no oneAPI dependency) ------------------


def find_cxx_compiler() -> pathlib.Path:
    """Return a usable C++ compiler, or raise ToolchainError.

    Resolution order: an explicit ``FREETOKEN_CXX`` override, then the first
    of ``c++`` / ``g++`` / ``clang++`` found on ``PATH``. Independent of the
    oneAPI ``icpx`` lookup in ``_toolchain.py`` -- see this module's
    docstring for why.
    """
    override = os.environ.get(CXX_ENV_OVERRIDE)
    if override and os.path.isfile(override) and os.access(override, os.X_OK):
        return pathlib.Path(override)

    for candidate in _CXX_CANDIDATES:
        found = shutil.which(candidate)
        if found:
            return pathlib.Path(found)

    raise ToolchainError(
        "no C++ compiler found for the native CPU MoE kernel. Install a "
        "system toolchain (g++ or clang++) or set FREETOKEN_CXX to a "
        f"compiler path. Searched PATH for: {list(_CXX_CANDIDATES)}"
    )


def cxx_version() -> Optional[str]:
    """The compiler's version banner (first line of ``--version``), or None.

    ``None`` when no compiler is found -- the cache-key builder treats that
    as "cannot build" (mirrors ``_toolchain.icpx_version``).
    """
    try:
        cxx = find_cxx_compiler()
    except ToolchainError:
        return None
    try:
        out = subprocess.run(
            [str(cxx), "--version"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = out.stdout.splitlines()[0].strip() if out.stdout.strip() else None
    return line


def cxx_flags() -> list[str]:
    """The compiler flags for a naive, single-threaded, optimized shared object.

    No SIMD flags (``-mavx512f`` etc.) here on purpose -- vectorization is
    issue #252's job on top of this naive port, not this issue's.
    """
    find_cxx_compiler()  # fail fast with the helpful message
    return ["-O2", "-shared", "-fPIC", "-std=c++17"]


# --- Build key / cache (mirrors kernel/utils.py's _build_key) -------------


def _build_key(name: str, source_file: str) -> str:
    """Key a compiled module by (name, source bytes, compiler identity, host arch).

    No Level Zero driver / Xe ISA legs here (unlike the SYCL kernels' key) --
    this kernel has no XPU dependency, so the only build inputs that can
    invalidate a cached module are the compiler itself, the host CPU
    architecture (a module built for one arch is not portable to another),
    and the source bytes.
    """
    cxx = cxx_version() or "cxx-missing"
    arch = platform.machine() or "arch-unknown"
    material = (
        f"freetoken__{name}|cxx={cxx}|arch={arch}|src={_source_fingerprint(source_file)}"
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    return f"{name}-{digest}"


def _compile(source_file: str, output_so: str) -> None:
    """Compile a plain C++ source to a shared object (raises ToolchainError on failure)."""
    cxx = find_cxx_compiler()
    out = pathlib.Path(output_so)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(cxx), *cxx_flags(), str(source_file), "-o", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ToolchainError(
            f"{cxx.name} failed to compile {source_file} (exit {proc.returncode}):\n"
            f"{proc.stderr}"
        )


def _load(so_path: pathlib.Path) -> ctypes.CDLL:
    """dlopen a compiled cpu_moe module. No SYCL/Level Zero lib path needed."""
    return ctypes.CDLL(str(so_path))


def cpu_moe(name: str = "cpu_moe") -> KernelModule:
    """Compile (or load from cache) the native CPU MoE kernel.

    Returns the loaded module. A cache hit (a module already built for this
    exact compiler + host architecture + source) skips the compile entirely;
    a cold call compiles once and populates the cache for later calls/processes
    (same "cache hit skips recompile" shape as ``kernel.utils.hello_copy``).
    Raises :class:`~freetoken.kernel._toolchain.ToolchainError` when no C++
    compiler is available -- callers (and tests) should catch this and skip.
    """
    if not CPU_MOE_SRC.is_file():
        raise ToolchainError(f"cpu_moe source not found at {CPU_MOE_SRC}")
    key = _build_key(name, str(CPU_MOE_SRC))
    cache_dir = _jit_cache_dir()
    so_path = cache_dir / key / f"{name}.so"

    if so_path.is_file():
        return KernelModule(path=so_path, loaded=_load(so_path), from_cache=True)

    cache_dir.mkdir(parents=True, exist_ok=True)
    key_dir = so_path.parent
    key_dir.mkdir(parents=True, exist_ok=True)
    # Compile to a per-call temp file in the same directory, then atomically
    # rename into place. so_path is content-addressed by _build_key (compiler +
    # host arch + source hash), so any caller racing us to the same key is
    # compiling byte-identical output -- os.replace is atomic on the same
    # filesystem, so a concurrent reader either sees the old (absent) path or
    # the fully-written file, never a partially-written one.
    #
    # tempfile.mkstemp (not "pid-named") because os.getpid() alone collides
    # across *threads* in the same process: _compile shells out via
    # subprocess.run, which releases the GIL, so two threads racing a cold
    # cache could both pick the same pid-named tmp path and stomp each
    # other's compile output mid-write. mkstemp atomically creates a
    # guaranteed-unique file (O_EXCL under the hood), so this holds under
    # both multi-process and multi-thread races.
    fd, tmp_name = tempfile.mkstemp(dir=key_dir, prefix=f".{name}.", suffix=".tmp.so")
    os.close(fd)
    tmp_so = pathlib.Path(tmp_name)
    try:
        _compile(str(CPU_MOE_SRC), str(tmp_so))
        # mkstemp creates the placeholder 0600 (owner-only); the compiler
        # overwrites the *content* but not that mode, so restore the normal
        # 0644 a directly-compiled .so would have gotten (readable by anyone
        # who can already read the cache dir -- the file has no secret
        # content, it's a compiled kernel).
        tmp_so.chmod(0o644)
        os.replace(tmp_so, so_path)
    finally:
        tmp_so.unlink(missing_ok=True)
    return KernelModule(path=so_path, loaded=_load(so_path), from_cache=False)


# --- ctypes call -----------------------------------------------------------


def _bind_forward(module: KernelModule):
    fn = module.loaded.freetoken_cpu_moe_forward
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # x
        ctypes.c_int,  # num_tokens
        ctypes.c_int,  # hidden
        ctypes.POINTER(ctypes.c_int32),  # expert_ids
        ctypes.POINTER(ctypes.c_float),  # expert_weights
        ctypes.c_int,  # topk
        ctypes.POINTER(ctypes.c_float),  # gate_up
        ctypes.POINTER(ctypes.c_float),  # down
        ctypes.c_int,  # num_experts
        ctypes.c_int,  # intermediate
        ctypes.POINTER(ctypes.c_uint8),  # expert_mask (nullable)
        ctypes.POINTER(ctypes.c_float),  # out
    ]
    fn.restype = ctypes.c_int
    return fn


def _as_c_float_ptr(t: torch.Tensor):
    return t.contiguous().numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def cpu_moe_forward(
    module: KernelModule,
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_weights: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    num_experts: int,
    intermediate: int,
    expert_mask: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """Run the native CPU MoE forward and return the routed contribution.

    Mirrors ``CpuMoeExecutor.forward`` (``expert_mask=None`` -- every expert
    is a candidate) and ``_Qwen3MoE._cpu_subset_math`` (``expert_mask`` names
    the hybrid split's CPU-computed expert ids) with the same expert-major
    then top-k-column accumulation order (see the kernel's docstring for why
    that order is load-bearing).

    Args:
        module: a loaded module from :func:`cpu_moe`.
        x: ``[T, H]`` float32 CPU tensor, token-major input activations.
        expert_ids: ``[T, k]`` int tensor of routed expert ids per token.
        expert_weights: ``[T, k]`` float tensor of router weights per token.
        gate_up: ``[E, 2*I, H]`` float32 CPU tensor (gate rows then up rows).
        down: ``[E, H, I]`` float32 CPU tensor.
        num_experts: E.
        intermediate: I.
        expert_mask: optional iterable of expert ids that are candidates
            (the hybrid split's CPU-computed set); ``None`` means every
            expert is a candidate (the plain ``cpu`` backend's use case).

    Returns:
        ``[T, H]`` float32 CPU tensor, the routed contribution -- the caller
        casts to the model dtype/device (matching both Python reference call
        sites, which do the same after this compute).
    """
    import torch  # lazy: torch is an optional extra, see the top-of-file note

    fn = _bind_forward(module)

    x_c = x.to("cpu", dtype=torch.float32).contiguous()
    ids_c = expert_ids.to("cpu", dtype=torch.int32).contiguous()
    w_c = expert_weights.to("cpu", dtype=torch.float32).contiguous()
    gu_c = gate_up.to("cpu", dtype=torch.float32).contiguous()
    dn_c = down.to("cpu", dtype=torch.float32).contiguous()

    num_tokens, hidden = int(x_c.shape[0]), int(x_c.shape[1])
    topk = int(ids_c.shape[1]) if ids_c.dim() == 2 else int(expert_weights.shape[-1])

    out = torch.zeros((num_tokens, hidden), dtype=torch.float32)

    mask_ptr = None
    if expert_mask is not None:
        mask = bytearray(num_experts)
        for e in expert_mask:
            mask[int(e)] = 1
        mask_arr = (ctypes.c_uint8 * num_experts).from_buffer(mask)
        mask_ptr = ctypes.cast(mask_arr, ctypes.POINTER(ctypes.c_uint8))

    rc = fn(
        _as_c_float_ptr(x_c),
        num_tokens,
        hidden,
        ids_c.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        _as_c_float_ptr(w_c),
        topk,
        _as_c_float_ptr(gu_c),
        _as_c_float_ptr(dn_c),
        int(num_experts),
        int(intermediate),
        mask_ptr,
        out.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if rc != 0:
        raise RuntimeError(f"freetoken_cpu_moe_forward failed (rc={rc})")
    return out


__all__ = [
    "CPU_MOE_SRC",
    "cpu_moe",
    "cpu_moe_forward",
    "cxx_flags",
    "cxx_version",
    "find_cxx_compiler",
]
