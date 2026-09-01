"""Read the ``ft bench bw`` profile and pick the offload-family MoE backend.

Upstream NVIDIA path: python/freetoken/moe/bench_profile.py.

``ft bench bw`` (see ``benchbw.py``) measures, per expert format, the *real*
bandwidths the offload-family backends actually ride on: the CPU MoE GEMV (what
``hybrid`` computes on the host) and the PCIe gather (what ``offload`` streams
over PCIe on a decode miss). This module is the *reader* the engine consults at
startup (torch-free, so the CPU venv loads it without a device): given the
profile JSON, it (1) recommends ``hybrid`` when the CPU MoE bandwidth exceeds
``threshold``x the PCIe gather bandwidth (else the always-working ``offload``),
and (2) yields the hybrid backend's per-step **fetch fraction** -- of a decode
step's expert misses, the share to PCIe-fetch (vs. the rest computed on the
CPU) -- so the two finish together (the ``q*`` split).

Torch-free and device-agnostic: profile paths are keyed by *XPU* UUID (``<name>
<uuid>``; this is the Intel port of upstream's per-GPU keying), and the only
device read is ``freetoken.utils.arch`` (no torch import). A missing / mismatched
profile resolves to ``None`` and the caller keeps its own default (offload), so
a box that never ran ``ft bench bw`` is unaffected.
"""
from __future__ import annotations

import json
import os
from typing import Any

from freetoken.utils import init_logger

logger = init_logger(__name__)

# Engine quant_format (models/config.py) -> benchbw format key. Only the
# offload-family formats with a CPU MoE weight path can ever resolve to hybrid;
# anything not listed falls through unmapped and finds no profile entry
# (-> None -> offload), the safe default.
_QUANT_TO_BENCH_FORMAT = {
    "nvfp4": "nvfp4",
    "ds_fp4": "ds_fp4",
    "mxfp4": "mxfp4_triton",
    "bf16": "bf16",
    "fp8_block": "fp8_block",
}

# Friendly CLI aliases for the internal quant_format keys.
_FORMAT_ALIASES = {"fp8": "fp8_block", "mxfp4": "mxfp4_triton"}

# Long torch dtype names -> the short bench format keys. ``str(torch.bfloat16)`` is
# "torch.bfloat16" (-> "bfloat16" once the prefix is stripped), but the profile is
# keyed on the short "bf16" the ``ft bench bw --dtype`` flag takes. A model parsed
# with no quant_format (a plain bf16 checkpoint) must still resolve to the
# "bf16" profile entry, so map the long dtype spellings to the short keys.
_DTYPES_TO_FORMAT = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}


def quant_format_for_dtype(dtype: Any) -> str | None:
    """The bench profile key for a torch dtype, or ``None`` when un-mappable.

    ``str(torch.bfloat16)`` -> "torch.bfloat16" -> "bf16"; quant-format spellings
    (``fp8``/``mxfp4``) pass through the ``_FORMAT_ALIASES`` map; an empty or
    un-recognized dtype yields ``None`` so the caller keeps its safe default.
    """
    if dtype is None:
        return None
    dt = str(dtype).lower()
    if dt.startswith("torch."):
        dt = dt.removeprefix("torch.")
    if not dt:
        return None
    return _FORMAT_ALIASES.get(dt, _DTYPES_TO_FORMAT.get(dt, dt))


def _cache_dir() -> str:
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(cache, "freetoken")


def default_profile_path(xpu_uuid: str | None = None) -> str:
    """``$XDG_CACHE_HOME/freetoken/benchbw/<xpu-uuid>.json``, else ``benchbw.json``.

    One file per XPU: bandwidth differs between slots. (Upstream keys by GPU UUID;
    this is the Intel port of the same scheme.)
    """
    if xpu_uuid:
        return os.path.join(_cache_dir(), "benchbw", f"{xpu_uuid}.json")
    return os.path.join(_cache_dir(), "benchbw.json")


def latest_profile_path() -> str | None:
    """Newest ``benchbw/*.json``, else the legacy ``benchbw.json``, else ``None``."""
    per_xpu = os.path.join(_cache_dir(), "benchbw")
    newest: tuple[float, str] | None = None
    try:
        for name in os.listdir(per_xpu):
            if not name.endswith(".json"):
                continue
            path = os.path.join(per_xpu, name)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if newest is None or mtime > newest[0]:
                newest = (mtime, path)
    except OSError:
        pass
    if newest is not None:
        return newest[1]
    legacy = default_profile_path()
    return legacy if os.path.isfile(legacy) else None


def _load(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _xpu_identity() -> tuple[str | None, str | None]:
    """``(name, uuid)`` of the first XPU, each ``None`` when absent/unknown.

    Read through ``freetoken.utils.arch`` (no torch import). ``uuid`` keys the
    per-device profile file; ``name`` is the mismatch-check key.
    """
    from freetoken.utils.arch import xpu_device_name, is_xpu_available

    if not is_xpu_available():
        return None, None
    try:
        import torch

        props = torch.xpu.get_device_properties(0)
        return str(getattr(props, "name", None) or ""), str(getattr(props, "uuid", None) or "")
    except Exception:
        return xpu_device_name(), None


def _usable_profile(
    xpu_name: str | None, path: str | None, xpu_uuid: str | None = None
) -> tuple[dict | None, str | None]:
    """The cached profile to trust, or ``None``.

    Lookup: explicit ``path`` (else ``FREETOKEN_BENCHBW_PATH``) ->
    ``benchbw/<xpu_uuid>.json`` -> legacy ``benchbw.json``. A profile measured on
    a different XPU (name mismatch) is hardware-specific and is *ignored* rather
    than trusted -- returning ``(None, path)`` so the caller logs the reason and
    keeps its default (offload).
    """
    name, uuid = xpu_name, xpu_uuid
    if path is None and name is None and uuid is None:
        # Auto-lookup: resolve this box's identity to key the per-uuid path.
        # ``path`` itself is kept None (not clobbered with the device *name* --
        # the first ``_xpu_identity()`` return -- which would make
        # ``path or <env>`` below truthy on the name string and silently ignore an
        # explicit FREETOKEN_BENCHBW_PATH in favor of a non-path). When the caller
        # supplies an explicit path, the caller-provided name/uuid (if any) are
        # trusted as-is and _xpu_identity() is not consulted, so an explicit path
        # + mismatched live identity cannot shadow the caller's verdict.
        name, uuid = _xpu_identity()
    explicit = path or os.environ.get("FREETOKEN_BENCHBW_PATH")
    if explicit:
        candidates = [explicit]
    else:
        candidates = [default_profile_path(uuid)] if uuid else []
        candidates.append(default_profile_path())
    prof, src = None, None
    for cand in candidates:
        prof = _load(cand)
        if isinstance(prof, dict):
            src = cand
            break
        if os.path.exists(cand):
            return None, cand  # unreadable profile for this card: stay on the default
    if not isinstance(prof, dict):
        return None, None
    prof_xpu = (prof.get("xpu") or prof.get("gpu") or {}).get("name")
    if name and prof_xpu and prof_xpu != name:
        logger.warning(
            f"benchbw profile {src} was measured on {prof_xpu!r}, not this XPU "
            f"({name!r}); ignoring it"
        )
        return None, src
    return prof, src


def load_backend_recommendation(
    quant_format: str,
    xpu_name: str | None = None,
    path: str | None = None,
    xpu_uuid: str | None = None,
) -> str | None:
    """Bench-recommended offload-family backend for ``quant_format``, or ``None``.

    Returns ``"hybrid"`` only when *every* benched workload sharing this format
    recommended hybrid (CPU MoE BW > threshold x PCIe gather BW); a mixed verdict
    (a near-threshold format) resolves conservatively to ``"offload"``. ``None``
    means "no usable profile" (see ``_usable_profile``) or no entry for this
    format. The caller keeps its own default (offload) on ``None``.
    """
    fmt = _QUANT_TO_BENCH_FORMAT.get(quant_format, quant_format)
    prof, _src = _usable_profile(xpu_name, path, xpu_uuid)
    if prof is None:
        return None

    # Preferred: the per-dtype tuning verdicts (`ft bench bw --dtype`), the direct
    # format->backend map the backend pick is meant to key on.
    dtypes = prof.get("dtypes")
    if isinstance(dtypes, dict) and dtypes.get(fmt) in ("hybrid", "offload"):
        return dtypes[fmt]

    # Fallback: a per-model profile (`ft bench bw --model`). Aggregate the workloads
    # sharing this format -- unanimous hybrid -> hybrid; any offload (a near-
    # threshold split) -> offload.
    workloads = prof.get("workloads")
    if not isinstance(workloads, dict):
        return None
    picks = [
        entry["recommended"]
        for wl in workloads.values()
        if isinstance(wl, dict)
        for entry in [(wl.get("kernels") or {}).get(fmt)]
        if isinstance(entry, dict) and entry.get("recommended")
    ]
    if not picks:
        return None
    return "hybrid" if all(p == "hybrid" for p in picks) else "offload"


def load_hybrid_fetch_fraction(
    quant_format: str,
    xpu_name: str | None = None,
    path: str | None = None,
    xpu_uuid: str | None = None,
) -> float | None:
    """Benched hybrid fetch fraction for ``quant_format``, or ``None``.

    The hybrid backend's bandwidth-matched fetch split: of a decode step's expert
    misses, fetch this fraction over PCIe and compute the rest on the CPU, so both
    finish together. The fraction is ``pcie / (pcie + cpu)`` -- of the two halves'
    combined bandwidth, the share carried by PCIe. Preferred source is the
    *overlapped* pair (CPU MoE and PCIe gather measured while running concurrently
    -- the real contention regime both halves actually run in):
    ``pcie_ov / (pcie_ov + cpu_ov)``. Older profiles without it fall back to the
    standalone bandwidths (``pcie / (pcie + cpu)``). Per-dtype entry first, then any
    per-model entry with this format. ``None`` = no usable profile; clamped to [0, 1].
    """
    fmt = _QUANT_TO_BENCH_FORMAT.get(quant_format, quant_format)
    prof, _src = _usable_profile(xpu_name, path, xpu_uuid)
    if prof is None:
        return None
    entries = [(prof.get("dtype_kernels") or {}).get(fmt)] + [
        (wl.get("kernels") or {}).get(fmt)
        for wl in (prof.get("workloads") or {}).values()
        if isinstance(wl, dict)
    ]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        cpu_ov, pcie_ov = entry.get("cpu_moe_overlap_gbs"), entry.get("pcie_gather_overlap_gbs")
        if cpu_ov and pcie_ov:
            return min(1.0, pcie_ov / (pcie_ov + cpu_ov))
        cpu, pcie = entry.get("cpu_moe_gbs"), entry.get("pcie_gather_gbs")
        if cpu and pcie:
            return min(1.0, pcie / cpu)
    return None


def resolve_backend(backend, *, is_moe: bool, quant_format: str | None = None) -> str | None:
    """Resolve the ``"auto"`` MoE backend, consulting the bench profile.

    Mirrors ``resolve_moe_backend`` but adds the ``ft bench bw`` recommendation:
    for a MoE on an XPU the default is still ``offload`` (the ADR 0002 LRU slot
    pool), but when the profile -- measured on *this* XPU -- recommends ``hybrid``
    for this expert format, ``auto`` upgrades to ``hybrid`` (CPU MoE BW beats
    PCIe by the threshold). A missing / mismatched profile leaves ``offload``
    untouched, so the upgrade only ever fires on a box that benched and said so.
    Non-MoE and non-XPU boxes are unchanged (``fused``).
    """
    from freetoken.utils.arch import is_xpu_available

    if backend != "auto":
        return backend
    if not is_moe:
        return "fused"
    if not is_xpu_available():
        return "fused"
    if quant_format is not None:
        rec = load_backend_recommendation(quant_format)
        if rec is not None:
            return rec
    return "offload"


__all__ = [
    "default_profile_path",
    "latest_profile_path",
    "load_backend_recommendation",
    "load_hybrid_fetch_fraction",
    "resolve_backend",
    "_FORMAT_ALIASES",
]
