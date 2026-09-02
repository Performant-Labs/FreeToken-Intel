"""``ft bench bw`` -- calibrate CPU RAM vs PCIe vs XPU HBM and pick the MoE backend.

Upstream NVIDIA path: python/freetoken/moe/benchbw.py (issue ``moe-hybrid``).

The hybrid MoE backend (issue #9, ADR 0002) splits each decode step's routed-
expert misses between two transports: a fraction ``f`` PCIe-fetched into the XPU
LRU slot pool and computed there (what ``offload`` does), the rest ``(1 - f)``
computed on the host CPU from the same pinned host banks (what ``cpu`` does).
This command measures the *real* bandwidths those two halves ride on, on *this*
machine, and writes the per-XPU profile JSON that ``freetoken.moe.bench_profile``
reads at engine startup to (1) recommend ``hybrid`` over ``offload`` when the CPU
MoE GEMV beats the PCIe gather by the threshold, and (2) yield the bandwidth-
matched fetch fraction ``f`` (``q*``: pcie / (pcie + cpu) -- the reader's formula).

Three measurements, per expert format:

* **CPU MoE GEMV** -- the host-side expert GEMM (``CpuMoeExecutor.forward``) for
  one decode token routed to ``E`` experts, reading the ``[E, 2I, H]`` gate_up +
  ``[E, H, I]`` down banks. Bytes = E * (2*I*H + H*I) * dt. This is the CPU half.
* **PCIe gather** -- streaming the same E expert rows from pinned host RAM into
  an XPU buffer (the offload half; ``OffloadMoeCache.copy_missing``'s per-row
  ``.to(device)`` in the pure-torch port, oneAPI ``queue.memcpy`` on the B70).
  Bytes = the same E * (2*I*H + H*I) * dt.
* **Overlapped** -- both at once on separate streams/threads; the contention
  regime the hybrid backend actually runs in (both halves are concurrent during a
  decode step). Each half's effective bandwidth is its bytes / the *wall* time
  of the concurrent run (both are slower than standalone; the ratio is what the
  ``q*`` split keys on).

Profile JSON shape (per-XPU file; see ``bench_profile`` for the reader):

.. code-block:: json

    {
      "xpu": {"name": "<device name>", "uuid": "<device uuid>"},
      "dtypes": {"<fmt>": "hybrid" | "offload"},
      "dtype_kernels": {
        "<fmt>": {
          "cpu_moe_gbs":  ..., "pcie_gather_gbs":  ...,
          "cpu_moe_overlap_gbs": ..., "pcie_gather_overlap_gbs": ...
        }
      }
    }

This command imports torch (it runs in the XPU venv, not the torch-free CPU
venv): it exercises real XPU + CPU tensors to measure real hardware bandwidths.
A missing / absent XPU is a clean error (the profile is hardware-specific).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from freetoken.utils import init_logger, is_xpu_available

logger = init_logger(__name__)

# The threshold the q* recommendation keys on: hybrid is recommended when the CPU
# MoE bandwidth exceeds this multiple of the PCIe gather bandwidth (under the
# overlap regime -- both are contended). A value near 2.0 matches upstream's
# "CPU beats PCIe by ~2x" heuristic; the reader (bench_profile) compares against
# this, so a profile's verdict and the engine's startup pick stay in lock-step.
HYBRID_CPU_OVER_PCIE_THRESHOLD = 2.0

# Default per-expert shape (Qwen3 MoE / Qwen3.5 MoE, bf16): hidden H = 2048,
# moe intermediate I = 768, E = 64 (Qwen3-30B-A3B), 4 MoE layers. These are the
# shape defaults for a bare `ft bench bw`; --model overrides them from the
# checkpoint (the real workload). bf16 is the only format the #17 loader wires
# today, so the default bench is bf16.
_DEFAULT_NUM_EXPERTS = 64
_DEFAULT_MOE_INTERMEDIATE = 768
_DEFAULT_HIDDEN = 2048
_DEFAULT_NUM_LAYERS = 4
_DEFAULT_DTYPE = "bf16"

# Warmup + measure iterations for each bandwidth sub-measurement. Small numbers
# are fine: these are large contiguous copies / GEMMs, not tiny kernels, so a
# handful of iterations is enough to average out jitter.
_WARMUP_ITERS = 3
_MEASURE_ITERS = 10


def _dtype_bytes(fmt: str) -> int:
    """Bytes per element for an expert-format key (the profile's format)."""
    if fmt in ("nvfp4", "ds_fp4"):
        return 1  # 4-bit packed (nvfp4 / deepseek fp4: 1 byte per 2 elements,
        # but the banks store one byte per 2-weight group; for a byte-count the
        # effective per-element is 0.5 -- use 1 to bound the gather size).
    if fmt == "mxfp4_triton":
        return 1
    if fmt in ("bf16", "fp16"):
        return 2
    if fmt in ("fp32", "fp8_block"):
        return 4 if fmt == "fp32" else 1
    return 2  # default to a 16-bit weight


def _xpu_identity() -> tuple[str, str]:
    """(name, uuid) of the first XPU (the profile is keyed per-XPU).

    ``freetoken.utils.arch`` is imported here (not at module top) so this module
    stays importable in the torch-free CPU venv; ``main`` only calls it after the
    ``is_xpu_available()`` guard has confirmed an XPU (and imported torch).
    """
    from freetoken.utils.arch import xpu_device_name

    name = xpu_device_name() or "(unknown)"
    uuid = ""
    try:
        import torch

        props = torch.xpu.get_device_properties(0)
        uuid = str(getattr(props, "uuid", None) or "")
    except Exception:
        pass
    return name, uuid


def _experts_bytes(num_experts: int, intermediate: int, hidden: int, fmt: str) -> int:
    """Bytes for one layer's E expert banks: E * (gate_up[2I,H] + down[H,I]).

    gate_up is [E, 2I, H] (gate [I,H] + up [I,H]); down is [E, H, I]. Per expert
    that is 2*I*H + H*I = 3*I*H elements. (The reader uses this same byte count to
    sanity-check the profile's *_gbs entries.)
    """
    bytes_per_elem = _dtype_bytes(fmt)
    per_expert = 3 * intermediate * hidden
    return num_experts * per_expert * bytes_per_elem


def _bench_pcie_gather(
    device, num_experts: int, intermediate: int, hidden: int, fmt: str
) -> float | None:
    """PCIe host->XPU copy bandwidth (GB/s) for E expert rows.

    Mirrors ``OffloadMoeCache.copy_missing``'s per-row ``host_row -> slot`` copy
    (the pure-torch port of the B70's oneAPI ``queue.memcpy``): a pinned host
    buffer of E rows, each copied into a preallocated XPU buffer. Timing is the
    XPU event around the copies (the copy engine), and the bytes are the E rows'
    total size.
    """
    if not is_xpu_available():
        logger.error("no XPU present: `ft bench bw` needs an Intel XPU")
        raise SystemExit(1)
    import torch

    bytes_per_elem = _dtype_bytes(fmt)
    dt = (
        torch.bfloat16
        if fmt in ("bf16", "fp16")
        else torch.float8_e8m0fnu if fmt in ("nvfp4", "ds_fp4", "mxfp4_triton", "fp8_block")
        else torch.float32
    )
    # gate_up shape per expert [2I, H]; down shape per expert [H, I].
    gu_shape = (2 * intermediate, hidden)
    dn_shape = (hidden, intermediate)
    gu_bytes = num_experts * gu_shape[0] * gu_shape[1] * bytes_per_elem
    dn_bytes = num_experts * dn_shape[0] * dn_shape[1] * bytes_per_elem
    total_bytes = gu_bytes + dn_bytes

    # Pinned host sources (page-locked for the DMA path on the B70).
    host_gu = torch.empty(gu_shape, dtype=dt, pin_memory=True).expand(num_experts, -1, -1)
    host_dn = torch.empty(dn_shape, dtype=dt, pin_memory=True).expand(num_experts, -1, -1)
    # XPU destination slot buffers (the offload slot pool's row shape).
    dev_gu = torch.empty((num_experts, *gu_shape), dtype=dt, device=device)
    dev_dn = torch.empty((num_experts, *dn_shape), dtype=dt, device=device)

    def _copy_once():
        dev_gu.copy_(host_gu, non_blocking=True)
        dev_dn.copy_(host_dn, non_blocking=True)

    start = torch.xpu.Event(enable_timing=True)
    stop = torch.xpu.Event(enable_timing=True)
    # Warmup.
    for _ in range(_WARMUP_ITERS):
        _copy_once()
    torch.xpu.synchronize()
    start.record()
    for _ in range(_MEASURE_ITERS):
        _copy_once()
    stop.record()
    torch.xpu.synchronize()
    ms = start.elapsed_time(stop)  # wall time for _MEASURE_ITERS copies
    secs = ms / 1000.0
    if secs <= 0:
        return None
    total_moved = total_bytes * _MEASURE_ITERS
    return (total_moved / 1e9) / secs  # GB/s


def _bench_cpu_moe(
    device, num_experts: int, intermediate: int, hidden: int, fmt: str
) -> float | None:
    """CPU MoE GEMV bandwidth (GB/s): one decode token routed to E experts.

    Reads the E expert banks (``[E, 2I, H]`` gate_up + ``[E, H, I]`` down) off the
    pinned host banks and runs ``CpuMoeExecutor.forward`` for a single token
    (``T=1``) with top_k=E (every expert is routed -- the cold-miss decode step
    the q* split is calibrated for). Bytes = the E banks' total size; time = the
    GEMV wall time. This is the CPU half the hybrid backend computes on the host.
    """
    import torch

    from freetoken.moe.cpu_executor import CpuMoeExecutor

    bytes_per_elem = _dtype_bytes(fmt)
    dt = torch.bfloat16 if fmt in ("bf16", "fp16") else (torch.float32 if fmt == "fp32" else torch.float32)
    # Host expert banks (the pinned sources the executor reads).
    gate_up = torch.randn((num_experts, 2 * intermediate, hidden), dtype=torch.float32) * 0.01
    down = torch.randn((num_experts, hidden, intermediate), dtype=torch.float32) * 0.01
    if dt == torch.bfloat16:
        gate_up = gate_up.bfloat16()
        down = down.bfloat16()
    # A single decode token routed to all E experts (top_k = E), weights uniform.
    flat = torch.randn((1, hidden), dtype=dt, device=device)
    top_idx = torch.zeros((1, num_experts), dtype=torch.int64, device=device)
    for e in range(num_experts):
        top_idx[0, e] = e
    top_w = torch.full((1, num_experts), 1.0 / num_experts, dtype=dt, device=device)

    executor = CpuMoeExecutor(num_experts=num_experts, intermediate=intermediate, threads=0)
    total_bytes = _experts_bytes(num_experts, intermediate, hidden, fmt)

    # Warmup (populates any lazy allocations, settles BLAS threading).
    for _ in range(_WARMUP_ITERS):
        executor.forward(flat, top_idx, top_w, gate_up, down)
    torch.xpu.synchronize()
    start = time.perf_counter()
    for _ in range(_MEASURE_ITERS):
        executor.forward(flat, top_idx, top_w, gate_up, down)
    torch.xpu.synchronize()
    secs = time.perf_counter() - start
    if secs <= 0:
        return None
    # The executor reads the E banks from host RAM; the bytes it touches are the
    # full E banks (cold miss) * iters.
    return (total_bytes * _MEASURE_ITERS / 1e9) / secs


def _bench_overlap(
    device, num_experts: int, intermediate: int, hidden: int, fmt: str
) -> tuple[float | None, float | None]:
    """Overlapped (contended) bandwidths: run the PCIe gather and the CPU MoE GEMV
    concurrently on separate streams/threads, and return each half's effective
    GB/s under the contention (bytes / shared wall time). This is the regime the
    hybrid backend actually runs in -- both halves are concurrent during a decode
    step -- so the q* fetch fraction keys on these, not the standalone numbers.
    """
    import threading

    if not is_xpu_available():
        return None, None
    import torch

    bytes_per_elem = _dtype_bytes(fmt)
    dt = (
        torch.bfloat16
        if fmt in ("bf16", "fp16")
        else torch.float8_e8m0fnu if fmt in ("nvfp4", "ds_fp4", "mxfp4_triton", "fp8_block")
        else torch.float32
    )
    gu_shape = (2 * intermediate, hidden)
    dn_shape = (hidden, intermediate)
    gu_bytes = num_experts * gu_shape[0] * gu_shape[1] * bytes_per_elem
    dn_bytes = num_experts * dn_shape[0] * dn_shape[1] * bytes_per_elem
    pcie_bytes = gu_bytes + dn_bytes

    host_gu = torch.empty(gu_shape, dtype=dt, pin_memory=True).expand(num_experts, -1, -1)
    host_dn = torch.empty(dn_shape, dtype=dt, pin_memory=True).expand(num_experts, -1, -1)
    dev_gu = torch.empty((num_experts, *gu_shape), dtype=dt, device=device)
    dev_dn = torch.empty((num_experts, *dn_shape), dtype=dt, device=device)

    gate_up = torch.randn((num_experts, 2 * intermediate, hidden), dtype=torch.float32) * 0.01
    down = torch.randn((num_experts, hidden, intermediate), dtype=torch.float32) * 0.01
    if dt == torch.bfloat16:
        gate_up = gate_up.bfloat16()
        down = down.bfloat16()
    flat = torch.randn((1, hidden), dtype=dt, device=device)
    top_idx = torch.zeros((1, num_experts), dtype=torch.int64, device=device)
    for e in range(num_experts):
        top_idx[0, e] = e
    top_w = torch.full((1, num_experts), 1.0 / num_experts, dtype=dt, device=device)
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    executor = CpuMoeExecutor(num_experts=num_experts, intermediate=intermediate, threads=0)
    # cpu_bytes = the E banks the GEMV reads (same byte count the CPU half moves).
    cpu_bytes = _experts_bytes(num_experts, intermediate, hidden, fmt)

    # Warmup both halves.
    for _ in range(_WARMUP_ITERS):
        dev_gu.copy_(host_gu, non_blocking=True)
        dev_dn.copy_(host_dn, non_blocking=True)
        executor.forward(flat, top_idx, top_w, gate_up, down)
    torch.xpu.synchronize()

    cpu_done = threading.Event()

    def _cpu_work():
        for _ in range(_MEASURE_ITERS):
            executor.forward(flat, top_idx, top_w, gate_up, down)
        cpu_done.set()

    cpu_thread = threading.Thread(target=_cpu_work, daemon=True)
    start = torch.xpu.Event(enable_timing=True)
    stop = torch.xpu.Event(enable_timing=True)
    t0 = time.perf_counter()
    cpu_thread.start()
    start.record()
    for _ in range(_MEASURE_ITERS):
        dev_gu.copy_(host_gu, non_blocking=True)
        dev_dn.copy_(host_dn, non_blocking=True)
    # Wait for the CPU half to finish so the wall time covers the slower half
    # (the real decode step is gated on the slower of the two).
    while not cpu_done.is_set():
        time.sleep(0.001)
    stop.record()
    torch.xpu.synchronize()
    wall = time.perf_counter() - t0
    if wall <= 0:
        return None, None
    pcie_gbs = (pcie_bytes * _MEASURE_ITERS / 1e9) / wall
    cpu_gbs = (cpu_bytes * _MEASURE_ITERS / 1e9) / wall
    return pcie_gbs, cpu_gbs


def _recommend(pcie_gbs: float, cpu_gbs: float) -> str:
    """hybrid iff the CPU MoE BW beats the PCIe gather BW by the threshold."""
    if pcie_gbs is None or cpu_gbs is None or pcie_gbs <= 0:
        return "offload"
    return "hybrid" if cpu_gbs > HYBRID_CPU_OVER_PCIE_THRESHOLD * pcie_gbs else "offload"


def _fetch_fraction(pcie_ov, cpu_ov, pcie, cpu) -> float:
    """The profile's fetch fraction, matching the reader's formula.

    ``bench_profile.load_hybrid_fetch_fraction`` reads ``pcie / (pcie + cpu)``
    (prefer the overlapped pair, else the standalone pair), so the writer stores
    the *same* value the reader will derive -- keeping the profile's own
    ``fetch_fraction`` field consistent with what the engine actually computes.
    """
    if pcie_ov and cpu_ov:
        return min(1.0, pcie_ov / (pcie_ov + cpu_ov))
    if pcie and cpu:
        return min(1.0, pcie / (pcie + cpu))
    return 0.0


def _bench_format(device, fmt: str, num_experts: int, intermediate: int, hidden: int) -> "dict | None":
    """Bench one expert format: standalone + overlapped BWs, recommendation, fraction."""
    pcie = _bench_pcie_gather(device, num_experts, intermediate, hidden, fmt)
    cpu = _bench_cpu_moe(device, num_experts, intermediate, hidden, fmt)
    pcie_ov, cpu_ov = _bench_overlap(device, num_experts, intermediate, hidden, fmt)
    if pcie is None and cpu is None:
        return None
    recommended = _recommend(pcie, cpu)
    frac = _fetch_fraction(pcie_ov, cpu_ov, pcie, cpu)
    return {
        "cpu_moe_gbs": round(cpu, 3) if cpu else None,
        "pcie_gather_gbs": round(pcie, 3) if pcie else None,
        "cpu_moe_overlap_gbs": round(cpu_ov, 3) if cpu_ov else None,
        "pcie_gather_overlap_gbs": round(pcie_ov, 3) if pcie_ov else None,
        "recommended": recommended,
        "fetch_fraction": round(frac, 4) if frac else None,
    }


def _model_shapes(argv: list[str]):
    """Parse --model and resolve per-format (E, I, H) from its config (or defaults)."""
    # Defaults when no --model: shape from the _DEFAULT_* constants.
    E, I, H = _DEFAULT_NUM_EXPERTS, _DEFAULT_MOE_INTERMEDIATE, _DEFAULT_HIDDEN
    dtypes = [_DEFAULT_DTYPE]
    if argv and argv[0] == "--model" and len(argv) >= 2:
        model_path = argv[1]
        try:
            from freetoken.models.loader import parse_hf_config  # noqa
        except Exception:
            parse_hf_config = None
        # Resolve the checkpoint's config for the real E/I/H + format.
        try:
            import json as _json
            from freetoken.utils import init_logger as _il

            _log = _il(__name__)
            cfg_path = os.path.join(model_path, "config.json")
            with open(cfg_path) as f:
                cfg = _json.load(f)
            tc = cfg.get("text_config", cfg)  # Qwen3.5 nests under text_config
            E = int(tc.get("num_experts") or tc.get("num_local_experts") or E)
            I = int(tc.get("moe_intermediate_size") or tc.get("intermediate_size") or I)
            H = int(tc.get("hidden_size") or H)
            _log.info(f"--model {model_path}: E={E} I={I} H={H}")
            # The model's quant format (if it declares one).
            qf = cfg.get("quant_format") or tc.get("quant_format")
            if qf:
                from freetoken.moe.bench_profile import _QUANT_TO_BENCH_FORMAT

                dtypes = [_QUANT_TO_BENCH_FORMAT.get(qf, qf)]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            from freetoken.utils import init_logger as _il2

            _il2(__name__).warning(f"--model {model_path}: could not read config ({exc}); using defaults")
    return E, I, H, dtypes


def parse_argv(argv: list[str] | None = None, prog: str = "ft bench bw") -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Calibrate CPU RAM vs PCIe vs XPU HBM bandwidths and pick the MoE "
            "backend (hybrid/offload). Writes a per-XPU profile JSON that the "
            "engine reads at startup to (1) recommend hybrid over offload and "
            "(2) set the hybrid fetch fraction (q* split)."
        ),
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="Expert format to bench (bf16 / fp8_block / nvfp4 / ds_fp4 / mxfp4). "
        "Default: bf16 (the format the #17 loader wires today).",
    )
    parser.add_argument(
        "--model",
        nargs="?",
        const="",
        default=None,
        help="Checkpoint path: bench with that model's real E/I/H + format "
        "instead of the defaults (64/768/2048, bf16).",
    )
    parser.add_argument(
        "--experts",
        type=int,
        default=None,
        help="Override the expert count E (default: the model's, else 64).",
    )
    parser.add_argument(
        "--intermediate",
        type=int,
        default=None,
        help="Override the MoE intermediate size I (default: the model's, else 768).",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=None,
        help="Override the hidden size H (default: the model's, else 2048).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Profile output path (default: $XDG_CACHE_HOME/freetoken/benchbw/<xpu-uuid>.json).",
    )
    parser.add_argument("--repeats", type=int, default=None, help="Measure iterations (default 10).")
    parser.add_argument("--quiet", action="store_true", help="Only print the profile JSON, not the report.")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    # Thread the override knobs into the module-level defaults the bench fns read.
    global _MEASURE_ITERS
    if args.repeats and args.repeats > 0:
        _MEASURE_ITERS = args.repeats
    return args


def main(argv: list[str] | None = None, prog: str = "ft bench bw") -> int:
    # Lazy XPU/torch import: this command runs in the XPU venv; a CPU-only box
    # (torch without xpu) gets a clean error rather than an ImportError.
    args = parse_argv(argv, prog=prog)
    if not is_xpu_available():
        logger.error("`ft bench bw` requires an Intel XPU (torch.xpu). None detected.")
        return 1
    import torch

    device = torch.device("xpu", 0)
    E, I, H, dtypes = _model_shapes([])  # shapes resolved in parse_argv below
    # Re-resolve shapes from the parsed args (the _model_shapes call above used
    # the raw argv; parse_argv consumed it). Re-derive from args.
    if args.experts:
        E = args.experts
    if args.intermediate:
        I = args.intermediate
    if args.hidden:
        H = args.hidden
    if args.dtype:
        dtypes = [args.dtype]

    name, uuid = _xpu_identity()
    out_path = args.out
    if not out_path:
        from freetoken.moe.bench_profile import default_profile_path

        out_path = default_profile_path(uuid) if uuid else default_profile_path()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    profile = {
        "xpu": {"name": name, "uuid": uuid},
        "dtypes": {},
        "dtype_kernels": {},
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "threshold_cpu_over_pcie": HYBRID_CPU_OVER_PCIE_THRESHOLD,
    }
    if not args.quiet:
        print(f"ft bench bw: {name} (uuid {uuid or 'unknown'})")
        print(f"  shapes: E={E} I={I} H={H}  formats={', '.join(dtypes)}")
        print(f"  output: {out_path}\n")
    for fmt in dtypes:
        if not args.quiet:
            print(f"  benching {fmt} ...")
        entry = _bench_format(device, fmt, E, I, H)
        if entry is None:
            if not args.quiet:
                print(f"  {fmt}: no measurement (XPU unavailable?) -- skipped")
            continue
        profile["dtypes"][fmt] = entry["recommended"]
        profile["dtype_kernels"][fmt] = entry
        if not args.quiet:
            print(
                f"    cpu_moe={entry['cpu_moe_gbs']} GB/s  pcie_gather={entry['pcie_gather_gbs']} GB/s"
            )
            if entry["cpu_moe_overlap_gbs"]:
                print(
                    f"    overlap: cpu={entry['cpu_moe_overlap_gbs']}  pcie={entry['pcie_gather_overlap_gbs']} GB/s"
                )
            print(f"    -> recommended: {entry['recommended']}   fetch_fraction={entry['fetch_fraction']}")
    with open(out_path, "w") as f:
        json.dump(profile, f, indent=2, sort_keys=False)
    if not args.quiet:
        print(f"\nprofile written to {out_path}")
        print("  `ft serve --moe-backend auto` will now consult this profile.")
    else:
        print(json.dumps(profile, indent=2))
    return 0


__all__ = ["main", "parse_argv", "HYBRID_CPU_OVER_PCIE_THRESHOLD"]
