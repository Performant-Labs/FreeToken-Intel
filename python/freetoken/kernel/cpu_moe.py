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
    """The compiler flags for a naive, optimized shared object.

    No SIMD flags (``-mavx512f`` etc.) here on purpose -- vectorization is
    issue #252's job on top of this naive port, not this issue's.

    ``-pthread`` is included unconditionally, needed for two independent
    reasons that both landed in ``cpu_moe.cpp``: issue #252's thread-pooled
    fast path (``freetoken_cpu_moe_forward_fast``, partitioning experts
    across ``std::thread`` workers) and issue #248's persistent *dispatch*
    worker thread (``std::thread`` + ``<mutex>``/``<condition_variable>``,
    the async submit/wait entry points). Either alone would require it;
    every libstdc++/libc++ on Linux needs it linked in to actually spawn and
    join a thread (a plain ``-lpthread`` would work for glibc but
    ``-pthread`` is the portable spelling recommended for both GCC and
    Clang, and also flips on the right preprocessor defines). This is a
    link-time requirement, not an ISA/vectorization flag, so it does not
    touch the "no SIMD flags" guarantee above (see
    ``test_cxx_flags_excludes_simd_flags``, which only checks for "avx"/"amx"
    substrings and is unaffected by this).
    """
    find_cxx_compiler()  # fail fast with the helpful message
    return ["-O2", "-shared", "-fPIC", "-std=c++17", "-pthread"]


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


def _bind_forward_fast(module: KernelModule):
    fn = module.loaded.freetoken_cpu_moe_forward_fast
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
        ctypes.c_int,  # num_threads (<=0 == auto)
        ctypes.c_int,  # force_scalar (nonzero forces the scalar row-compute)
        ctypes.POINTER(ctypes.c_float),  # out
    ]
    fn.restype = ctypes.c_int
    return fn


def _bind_avx512_available(module: KernelModule):
    fn = module.loaded.freetoken_cpu_moe_avx512_available
    fn.argtypes = []
    fn.restype = ctypes.c_int
    return fn


def cpu_moe_avx512_available(module: KernelModule) -> bool:
    """Whether this process's CPU supports AVX-512F (the fast path's gate).

    A pure runtime probe -- independent of how the .so was compiled, since
    the AVX-512 row-compute is emitted via a per-function
    ``__attribute__((target(...)))`` rather than a global ``-mavx512f`` flag
    (see ``cxx_flags``'s docstring / the kernel source for why).
    """
    return bool(_bind_avx512_available(module)())


def _as_c_float_ptr(t: torch.Tensor):
    return t.contiguous().numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _as_c_bf16_bits_ptr(t: torch.Tensor):
    """A ``ctypes.POINTER(c_uint16)`` over a bfloat16 tensor's raw storage.

    ctypes (and numpy) have no native bfloat16 concept, so this reinterprets
    the tensor's bit pattern as ``uint16`` (same size, same bytes, zero
    copy) rather than converting the *value* -- ``.view(torch.uint16)`` is
    exactly this bit-reinterpretation, not a numeric cast (unlike
    ``.to(torch.uint16)``, which would truncate/round the value). Issue
    #257: this is how the raw bf16 bytes cross the ctypes boundary with no
    float32 materialization anywhere on the way.
    """
    import torch  # lazy: torch is an optional extra, see the top-of-file note

    t = t.contiguous()
    if t.dtype != torch.bfloat16:
        raise TypeError(f"expected a bfloat16 tensor, got {t.dtype}")
    return t.view(torch.uint16).numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))


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


def cpu_moe_forward_fast(
    module: KernelModule,
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_weights: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    num_experts: int,
    intermediate: int,
    expert_mask: Optional[Iterable[int]] = None,
    *,
    threads: int = 0,
    force_scalar: bool = False,
) -> torch.Tensor:
    """Issue #252's fast path: runtime AVX-512 dispatch + thread-pool parallelism.

    Same math, arguments, and expert-major/top-k-column accumulation contract
    as :func:`cpu_moe_forward`; this entry point additionally vectorizes the
    per-row GEMM with AVX-512 when the host CPU supports it (falling back to
    the identical scalar math otherwise) and, when ``threads`` requests more
    than one worker, parallelizes across the expert range. Multi-threaded
    runs are only guaranteed to match :func:`cpu_moe_forward` within float32
    tolerance (summation order differs across threads/lanes), not bit-exact
    -- see the kernel source's design note.

    Args:
        threads: worker thread count. ``0`` (the default) means "auto"
            (``std::thread::hardware_concurrency()``, clamped to
            ``num_experts``); ``1`` runs single-threaded with the exact same
            iteration order as :func:`cpu_moe_forward`.
        force_scalar: forces the scalar row-compute even when AVX-512 is
            available -- a deterministic test hook for exercising the
            fallback branch on a host that does have AVX-512 (mirrors this
            project's "force what the host can't otherwise exercise"
            testing pattern). Also settable via the
            ``FREETOKEN_CPU_MOE_FORCE_SCALAR`` environment variable (checked
            when this argument is left at its default ``False``), so a test
            run or deployment can force the fallback without touching call
            sites.
    """
    import torch  # lazy: torch is an optional extra, see the top-of-file note

    fn = _bind_forward_fast(module)

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

    force_scalar_effective = force_scalar or os.environ.get("FREETOKEN_CPU_MOE_FORCE_SCALAR", "") not in (
        "",
        "0",
    )

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
        int(threads),
        1 if force_scalar_effective else 0,
        out.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if rc != 0:
        raise RuntimeError(f"freetoken_cpu_moe_forward_fast failed (rc={rc})")
    return out


# --- Async ctypes call (issue #248) ----------------------------------------


def _bind_submit(module: KernelModule):
    fn = module.loaded.freetoken_cpu_moe_submit
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # x
        ctypes.c_int,  # num_tokens
        ctypes.c_int,  # hidden
        ctypes.POINTER(ctypes.c_int32),  # expert_ids
        ctypes.POINTER(ctypes.c_float),  # expert_weights
        ctypes.c_int,  # topk
        ctypes.POINTER(ctypes.c_uint16),  # gate_up -- raw bf16 bit patterns (issue #257)
        ctypes.POINTER(ctypes.c_uint16),  # down -- raw bf16 bit patterns (issue #257)
        ctypes.c_int,  # num_experts
        ctypes.c_int,  # intermediate
        ctypes.POINTER(ctypes.c_uint8),  # expert_mask (nullable)
        ctypes.c_int,  # force_scalar (nonzero forces the scalar bf16 row-compute)
        ctypes.POINTER(ctypes.c_float),  # out
    ]
    fn.restype = ctypes.c_int64
    return fn


def _bind_wait(module: KernelModule):
    fn = module.loaded.freetoken_cpu_moe_wait
    fn.argtypes = [ctypes.c_int64]
    fn.restype = ctypes.c_int
    return fn


class CpuMoeJob:
    """A job submitted to the native CPU MoE worker thread (issue #248).

    Returned by :func:`cpu_moe_submit`. Call :meth:`result` (mirrors
    ``concurrent.futures.Future.result()`` -- the shape the old
    ``ThreadPoolExecutor`` handoff had, so call sites keep the same
    submit-then-later-call-result() structure) once to block until the
    native worker thread signals completion and get the output tensor.
    ``wait()`` is an alias, matching the ``cpu_moe_wait(job_handle)``
    language issue #248 itself uses.

    Holds a reference to every buffer the native call was hand the raw
    address of (``ctypes...ctypes.data_as`` pointers are only valid for as
    long as the backing numpy/torch storage is alive) -- none of them may be
    garbage collected before :meth:`result` returns, since the worker thread
    may still be reading or writing them at any point before then.
    """

    __slots__ = ("_module", "_handle", "_out", "_keep_alive", "_done")

    def __init__(self, module: KernelModule, handle: int, out: "torch.Tensor", keep_alive: tuple) -> None:
        self._module = module
        self._handle = handle
        self._out = out
        self._keep_alive = keep_alive
        self._done = False

    def result(self) -> "torch.Tensor":
        if self._done:
            # Already collected: cheap to allow a second call to return the
            # same tensor (unlike a real Future this cannot re-block on the
            # native side -- the slot was already freed for reuse), but a
            # *third* party calling result() twice more likely indicates a
            # logic bug (double-consuming a job) than a legitimate re-read,
            # so this stays intentionally strict rather than silently
            # re-returning a value that might no longer be this job's.
            raise RuntimeError("CpuMoeJob.result() already called for this job")
        self._done = True
        fn = _bind_wait(self._module)
        rc = fn(self._handle)
        if rc != 0:
            raise RuntimeError(f"freetoken_cpu_moe_wait failed (rc={rc})")
        return self._out

    def wait(self) -> "torch.Tensor":
        """Alias for :meth:`result` (the ``cpu_moe_wait`` naming issue #248 uses)."""
        return self.result()


def cpu_moe_submit(
    module: KernelModule,
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_weights: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    num_experts: int,
    intermediate: int,
    expert_mask: Optional[Iterable[int]] = None,
    *,
    force_scalar: bool = False,
) -> CpuMoeJob:
    """Hand a CPU MoE forward job to the persistent native worker thread; return immediately.

    Same argument contract as :func:`cpu_moe_forward` -- identical layout and
    expert-major-then-top-k-column accumulation order (see that function's
    docstring) -- except ``gate_up``/``down``: issue #257 keeps those in
    their native ``bfloat16`` storage the whole way to the native call, never
    materializing a float32 copy of a full expert weight bank (the O(bank
    size) conversion pass #251's live B70 testing found costs *more* than
    the compute it feeds). Pass ``gate_up``/``down`` as ``bfloat16`` tensors
    (any other dtype is cast to ``bfloat16`` here, which is itself the only
    conversion this function performs on them -- no float32 stop along the
    way). ``x``/``expert_ids``/``expert_weights`` are unaffected -- the
    activation conversion was never the bottleneck (#257's own scope note).

    The only other difference from :func:`cpu_moe_forward` is *where* the
    compute runs: asynchronously, on the native module's persistent
    ``std::thread`` worker (issue #248), rather than synchronously on the
    calling thread. Because ``ctypes`` releases the GIL for the duration of
    this call, the (cheap -- just claiming a job slot and handing off a
    struct) act of submitting does not itself contend with whatever the
    caller does next; call :meth:`CpuMoeJob.result` on the returned job once
    that other work is done to block for this job's result.

    Args:
        force_scalar: forces the scalar bf16 row-compute even when AVX-512
            is available -- a deterministic test hook for the fallback
            branch (mirrors :func:`cpu_moe_forward_fast`'s own
            ``force_scalar``), also settable via the
            ``FREETOKEN_CPU_MOE_FORCE_SCALAR`` environment variable when
            left at its default ``False``.
    """
    import torch  # lazy: torch is an optional extra, see the top-of-file note

    fn = _bind_submit(module)

    x_c = x.to("cpu", dtype=torch.float32).contiguous()
    ids_c = expert_ids.to("cpu", dtype=torch.int32).contiguous()
    w_c = expert_weights.to("cpu", dtype=torch.float32).contiguous()
    # bf16-preserving -- NOT `.to("cpu", dtype=torch.float32)`. This is the
    # crux of issue #257: gate_up/down cross into the native call still in
    # their native bf16 storage, dequantized only inside the kernel's FMA.
    gu_c = gate_up.to(device="cpu", dtype=torch.bfloat16).contiguous()
    dn_c = down.to(device="cpu", dtype=torch.bfloat16).contiguous()

    num_tokens, hidden = int(x_c.shape[0]), int(x_c.shape[1])
    topk = int(ids_c.shape[1]) if ids_c.dim() == 2 else int(expert_weights.shape[-1])

    out = torch.zeros((num_tokens, hidden), dtype=torch.float32)

    mask_buf = None
    mask_ptr = None
    if expert_mask is not None:
        mask_buf = bytearray(num_experts)
        for e in expert_mask:
            mask_buf[int(e)] = 1
        mask_arr = (ctypes.c_uint8 * num_experts).from_buffer(mask_buf)
        mask_ptr = ctypes.cast(mask_arr, ctypes.POINTER(ctypes.c_uint8))

    force_scalar_effective = force_scalar or os.environ.get("FREETOKEN_CPU_MOE_FORCE_SCALAR", "") not in (
        "",
        "0",
    )

    handle = fn(
        _as_c_float_ptr(x_c),
        num_tokens,
        hidden,
        ids_c.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        _as_c_float_ptr(w_c),
        topk,
        _as_c_bf16_bits_ptr(gu_c),
        _as_c_bf16_bits_ptr(dn_c),
        int(num_experts),
        int(intermediate),
        mask_ptr,
        1 if force_scalar_effective else 0,
        out.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if handle < 0:
        raise RuntimeError(f"freetoken_cpu_moe_submit failed (rc={handle})")
    # Every buffer the native side now holds a raw pointer into must outlive
    # the worker thread's read/write window, which ends only when
    # CpuMoeJob.result()'s freetoken_cpu_moe_wait() call returns -- so the
    # job keeps a reference to all of them (mask_buf may be None; that's a
    # harmless no-op entry in the tuple).
    return CpuMoeJob(module, handle, out, (x_c, ids_c, w_c, gu_c, dn_c, mask_buf))


def cpu_moe_wait(job: CpuMoeJob) -> "torch.Tensor":
    """Block until ``job`` (from :func:`cpu_moe_submit`) has finished; return its result."""
    return job.result()


def hybrid_subset_submit(
    module: KernelModule,
    x: torch.Tensor,
    top_idx: torch.Tensor,
    top_w: Optional[torch.Tensor],
    gate_up: torch.Tensor,
    down: torch.Tensor,
    candidate_experts: Iterable[int],
    intermediate: int,
) -> CpuMoeJob:
    """Submit the hybrid MoE split's CPU-half job (issue #248), touching only
    ``candidate_experts``' bank bytes -- not the whole per-layer bank.

    A plain ``cpu_moe_submit(..., expert_mask=candidate_experts)`` call would
    still have to copy *every* expert's ``gate_up``/``down`` rows to a
    contiguous buffer first (the native call's pointer arithmetic needs the
    full ``[num_experts, ...]`` stride, even for the masked-out rows it never
    reads). On a many-expert model -- the real target here is
    Qwen3.6-35B-A3B, 256 experts/layer -- that is tens of megabytes of wasted
    copying per layer per decode step even when the hybrid split names only
    half of them.

    Since issue #257, ``gate_up``/``down`` stay in their native ``bfloat16``
    storage the whole way through this gather (``index_select`` on bf16 --
    half the bytes moved compared to a float32 copy) and into the native
    call (:func:`cpu_moe_submit` casts to ``bfloat16``, not ``float32``, if
    they arrive in some other dtype) -- no float32-materialized copy of a
    full expert weight bank is ever created anywhere on this path, matching
    upstream FreeToken's zero-materialization design (see #257).

    Instead: gather just the ``candidate_experts`` bank rows into a compact
    ``[len(candidate_experts), ...]`` buffer and remap ``top_idx`` to that
    buffer's local indices (0..``len(candidate_experts)``-1); a routed slot
    whose *global* expert id is not a candidate remaps to -1, which the
    native loop's ``expert_ids[...] == e`` check (``e`` always in
    ``[0, len(candidate_experts))``) never matches, so it contributes
    nothing -- exactly mirroring the old per-expert Python loop
    (``_cpu_subset_math``), which only ever touched the candidate experts'
    tensor slices. ``candidate_experts`` is sorted before assigning local
    indices, so accumulation stays in ascending *global* expert-id order
    (then top-k-column), matching every other backend's accumulation order.

    ``top_w=None`` means every routed slot contributes with an implicit
    weight of 1.0 -- the ``qwen3_5_moe`` hybrid split's CPU half never
    applied the real per-row router weight (its own code documents this as
    "always identically 1.0 here", i.e. a no-op multiply); passing a real
    tensor (the ``qwen3_moe`` hybrid split's contract) applies it.
    """
    import torch  # lazy: torch is an optional extra, see the top-of-file note

    top_idx_cpu = top_idx.to("cpu")
    ids_sorted = sorted(int(e) for e in candidate_experts)
    n_local = len(ids_sorted)

    id_tensor = torch.tensor(ids_sorted, dtype=torch.long)
    gu_compact = gate_up.index_select(0, id_tensor.to(gate_up.device))
    dn_compact = down.index_select(0, id_tensor.to(down.device))

    # expert id -> local index (0..n_local-1); every id not named by
    # candidate_experts maps to -1 (never matched by the native loop, whose
    # `e` always stays in [0, n_local)). Sized to the largest id that can
    # legitimately appear in top_idx_cpu; candidate_experts is always a
    # subset of top_idx_cpu's own values in every real call site, so this is
    # always big enough to hold every id in ids_sorted too.
    table_size = int(top_idx_cpu.max().item()) + 1 if top_idx_cpu.numel() else 1
    remap = torch.full((max(table_size, 1),), -1, dtype=torch.int64)
    for local_i, e in enumerate(ids_sorted):
        remap[e] = local_i
    local_top_idx = remap[top_idx_cpu]

    if top_w is not None:
        weight = top_w.to("cpu")
    else:
        weight = torch.ones(top_idx_cpu.shape, dtype=torch.float32)

    return cpu_moe_submit(
        module,
        x,
        local_top_idx,
        weight,
        gu_compact,
        dn_compact,
        n_local,
        intermediate,
        expert_mask=None,
    )


class HybridCpuPool:
    """Persistent native-worker dispatch pool for the hybrid MoE split's CPU
    half (issue #248), replacing the previous per-model ``ThreadPoolExecutor``
    (see ``_Qwen3MoE._hybrid_cpu_pool`` / ``_Qwen35MoE._hybrid_cpu_pool``,
    which cache one instance of this class on the model exactly as they
    cached the old executor -- compiling/loading the native module once and
    reusing its persistent worker thread across every layer / every decode
    step, not per call).
    """

    def __init__(self, name: str = "cpu_moe") -> None:
        self._module = cpu_moe(name)

    def submit(
        self,
        x: torch.Tensor,
        top_idx: torch.Tensor,
        top_w: Optional[torch.Tensor],
        gate_up: torch.Tensor,
        down: torch.Tensor,
        candidate_experts: Iterable[int],
        intermediate: int,
    ) -> CpuMoeJob:
        return hybrid_subset_submit(
            self._module, x, top_idx, top_w, gate_up, down, candidate_experts, intermediate
        )


__all__ = [
    "CPU_MOE_SRC",
    "CpuMoeJob",
    "HybridCpuPool",
    "cpu_moe",
    "cpu_moe_avx512_available",
    "cpu_moe_forward",
    "cpu_moe_forward_fast",
    "cpu_moe_submit",
    "cpu_moe_wait",
    "cxx_flags",
    "cxx_version",
    "find_cxx_compiler",
    "hybrid_subset_submit",
]
