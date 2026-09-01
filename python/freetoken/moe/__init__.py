from __future__ import annotations

from typing import Protocol

from freetoken.utils import Registry, init_logger

from .base import BaseMoeBackend

logger = init_logger(__name__)


class MoeBackendCreator(Protocol):
    def __call__(self) -> BaseMoeBackend: ...


SUPPORTED_MOE_BACKENDS = Registry[MoeBackendCreator]("MoE Backend")
OFFLOAD_MOE_BACKENDS = frozenset({"offload", "cpu", "hybrid"})


def is_offload_moe_backend(backend: str) -> bool:
    return backend in OFFLOAD_MOE_BACKENDS


def resolve_moe_backend(backend, *, is_moe: bool) -> str | None:
    """Resolve the ``"auto"`` MoE backend to a concrete one.

    Only ``"auto"`` is resolved: it picks the host-offload backend when the model
    is a MoE and an XPU is available (the B70 cannot hold a 35B-class expert set
    in 32 GB VRAM alongside the KV pool), otherwise the in-VRAM fused backend.
    An explicit backend name -- or ``None`` (the legacy default, which the loader
    treats as "in-VRAM / whatever the model built") -- is returned unchanged, so
    callers that already pass ``"fused"`` / ``"offload"`` (or ``None``) are
    unaffected by the resolution.
    """
    if backend != "auto":
        return backend
    if is_moe:
        from freetoken.utils.arch import is_xpu_available

        return "offload" if is_xpu_available() else "fused"
    return "fused"


def parse_moe_cpu_layers(spec: str | None, num_moe_layers: int) -> list[int] | None:
    """Expand a ``--moe-cpu-layers`` spec into the MoE-layer indices on the CPU.

    Issue #8 (moe-cpu) / ADR 0002: with the offload or hybrid backend, only some
    MoE layers compute on the CPU expert executor (host RAM) while the rest stay
    on the XPU offload slot pool. The spec (a ``ft serve --moe-cpu-layers`` string)
    is:

    * ``None`` / ``""`` -- **no CPU override**: no MoE layer is steered to the
      CPU, so the resolved MoE backend runs entirely as chosen (``auto`` on an XPU
      -> ``offload``, the ADR 0002 LRU slot pool, the 32 GB B70 default). This is
      the serve default: ``--moe-backend auto`` without ``--moe-cpu-layers`` must
      keep the offload invariant (all MoE layers on the XPU slot pool), so an
      *unspecified* spec must NOT pull layers onto the CPU.
    * ``"auto"`` -- explicitly opt in to the CPU path: **all** MoE layers on the
      CPU (the ``--moe-backend cpu`` default). The #9 bandwidth profile that would
      select a head+tail subset instead is not online yet; issue #9 refines this
      once it exists.
    * ``"0"`` -- no MoE layers on the CPU (all on the XPU offload slot pool); the
      explicit form of the ``None``/``""`` default.
    * an integer ``N`` -- the first ``N`` MoE layers (0-based) on the CPU; ``N >=
      num_moe_layers`` means all of them.
    * a fraction ``F`` in ``(0, 1]`` (e.g. ``0.5``) -- the first ``ceil(F *
      num_moe_layers)`` MoE layers on the CPU.
    * an explicit id list ``"3,7,11"`` -- those MoE layers on the CPU (validated
      in-bounds, de-duplicated, ascending).

    Returns ``None`` for the "all MoE layers on the CPU" cases (the model then
    runs every MoE layer on the CPU), ``[]`` for the "no CPU override" cases, or
    the concrete ascending list otherwise. Pure Python (torch-free): the CPU venv
    parses the spec without importing torch, so it is unit-testable off-GPU.
    """
    if num_moe_layers <= 0:
        return []
    if spec is None:
        # No CPU override: the resolved backend runs as chosen (offload for
        # --moe-backend auto on an XPU). Distinct from "all on CPU" (None, below).
        return []
    s = str(spec).strip()
    if s == "":
        return []
    if s.lower() == "auto":
        # Explicit opt-in to the CPU path: the #9 bandwidth profile that would
        # select a head+tail subset is not online yet, so "auto" = all MoE layers
        # on the CPU (the --moe-backend cpu default).
        return list(range(num_moe_layers))
    if s == "0":
        return []
    try:
        fval = float(s)
    except ValueError:
        ids = [int(tok) for tok in s.split(",") if tok.strip() != ""]
        if not ids:
            return []
        for i in ids:
            if not (0 <= i < num_moe_layers):
                raise ValueError(
                    f"--moe-cpu-layers id {i} is out of range [0, {num_moe_layers - 1})"
                )
        return sorted(set(ids))
    if fval >= 0.0 and fval == int(fval):
        # Integer-valued spec. fval == 1.0 is treated as the FRACTION "1" (== all
        # MoE layers on the CPU), NOT as "the first 1 layer": a count of 1 is the
        # ambiguous case where "1" most naturally reads as the fraction 1.0. Any
        # other whole number (>= 0) is a count of leading MoE layers (N >= total
        # -> all).
        if fval == 1.0:
            return None
        n = int(fval)
        if n >= num_moe_layers:
            return None
        return list(range(n))
    if 0.0 < fval <= 1.0:
        # Fraction spec: the first ceil(F * total) MoE layers on the CPU.
        import math

        n = math.ceil(fval * num_moe_layers)
        if n >= num_moe_layers:
            return None
        return list(range(n))
    # A non-integer float that is negative or > 1.0 (e.g. "1.5") is neither a
    # valid count nor a valid fraction -- reject loudly. A silent None here would
    # mean "all MoE layers on the CPU", which a typo'd spec never intended.
    raise ValueError(
        f"--moe-cpu-layers: {spec!r} is not a valid spec "
        "(int, fraction in (0, 1], 'auto', or a comma-separated id list)"
    )


@SUPPORTED_MOE_BACKENDS.register("fused")
def create_fused_moe_backend():
    from .fused import FusedMoe

    return FusedMoe()


@SUPPORTED_MOE_BACKENDS.register("offload")
def create_offload_moe_backend():
    from .offload import OffloadMoeBackend

    return OffloadMoeBackend()


@SUPPORTED_MOE_BACKENDS.register("cpu")
def create_cpu_moe_backend():
    from .cpu_offload import CpuOffloadMoeBackend

    return CpuOffloadMoeBackend()


@SUPPORTED_MOE_BACKENDS.register("hybrid")
def create_hybrid_moe_backend():
    from .cpu_offload import HybridMoeBackend

    return HybridMoeBackend()


def create_moe_backend(backend: str) -> BaseMoeBackend:
    return SUPPORTED_MOE_BACKENDS[backend]()


__all__ = [
    "BaseMoeBackend",
    "create_moe_backend",
    "SUPPORTED_MOE_BACKENDS",
    "OFFLOAD_MOE_BACKENDS",
    "is_offload_moe_backend",
    "resolve_moe_backend",
    "parse_moe_cpu_layers",
]
