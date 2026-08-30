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
]
