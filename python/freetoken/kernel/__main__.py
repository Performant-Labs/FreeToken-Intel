"""``python -m freetoken.kernel`` -- print toolchain / XPU / AOT cache status.

A single, CPU-safe (no torch import) entry point that answers "is this box ready
to build and run SYCL kernels, and what's cached?". Filled in for GitHub issue
``kernel-sycl``.
"""
from __future__ import annotations

import sys


def _line(label: str, value: str) -> None:
    print(f"{label:<22} {value}")


def _run_smoke_test() -> int:
    """Compile (or load from cache) the hello-copy kernel and run it on the XPU.

    Returns 0 on success, non-zero on failure. The XPU is required: on a
    CPU-only box the SYCL queue would bind a CPU device, which would pass the
    smoke test while falsely claiming XPU readiness -- so we refuse to run
    without torch.xpu and say so.
    """
    try:
        import torch

        if not torch.xpu.is_available():
            print("No XPU available -- the hello-copy smoke test requires an Intel XPU.")
            print("On a CPU-only box, exit 2 is expected (see `ft device` / preflight).")
            return 2
    except Exception:
        print("torch is not importable -- cannot run the XPU smoke test.")
        return 2

    from freetoken.kernel.utils import hello_copy, run_hello_copy

    module = hello_copy()
    action = "loaded from cache" if module.from_cache else "compiled (cold)"
    copied = run_hello_copy(module, count=1024)
    print(f"hello_copy: {action}; copied {copied} floats (host -> XPU -> host).")
    return 0 if copied == 1024 else 1


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv and argv[0] == "run":
        return _run_smoke_test()

    from freetoken.kernel import backend
    from freetoken.kernel._toolchain import ToolchainError, find_icpx, icpx_version
    from freetoken.kernel import aot

    print("FreeToken-Intel SYCL kernel toolchain status")
    print("-" * 52)

    # --- Toolchain ---------------------------------------------------------
    if backend.is_oneapi_dpcpp_installed():
        try:
            _line("icpx", str(find_icpx()))
            _line("DPC++ version", icpx_version() or "unknown")
        except ToolchainError as exc:
            _line("icpx", f"MISSING ({exc})")
    else:
        _line("oneAPI DPC++", "NOT FOUND (run: source /opt/intel/oneapi/setvars.sh)")

    _line("Triton-Intel", "installed" if backend.is_triton_intel_installed() else "absent")
    _line("IPEX", "installed" if backend.is_ipex_installed() else "absent")

    # --- XPU (torch imported lazily; never fatal on a CPU-only box) --------
    try:
        import torch

        xpu_ok = torch.xpu.is_available()
        _line("torch.xpu", f"{'available' if xpu_ok else 'unavailable'} (torch {torch.__version__})")
        if xpu_ok:
            try:
                _line("Level Zero driver", backend.level_zero_driver_version() or "not exposed")
            except Exception:
                pass
    except Exception:
        _line("torch", "not importable")

    # --- AOT / JIT cache ---------------------------------------------------
    cache_dir = aot.aot_cache_dir()
    _line("AOT cache dir", str(cache_dir) if cache_dir else "disabled / not configured")
    if cache_dir is not None and cache_dir.is_dir():
        entries = sorted(p.name for p in cache_dir.iterdir() if p.is_dir())
        _line("cached modules", ", ".join(entries) if entries else "(empty -- nothing built yet)")
    else:
        _line("cached modules", "n/a")

    print("-" * 52)
    print("hello_copy smoke test: run with `python -m freetoken.kernel run`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
