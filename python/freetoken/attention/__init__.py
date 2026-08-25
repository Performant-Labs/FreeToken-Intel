from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from freetoken.utils import Registry, init_logger

from .base import AttentionSpec, AttnType, BaseAttnBackend, BaseAttnMetadata, HybridBackend

logger = init_logger(__name__)


class BackendCreator(Protocol):
    def __call__(self, config) -> BaseAttnBackend: ...


@dataclass(frozen=True)
class BackendInfo:
    supported_types: frozenset[AttnType]
    requires_sycl: bool = False
    requires_triton_intel: bool = False
    page_sizes: tuple[int, ...] | None = None
    consumes_attn_spec: bool = False
    hybrid_linear_ok: bool = True


SUPPORTED_ATTENTION_BACKENDS = Registry[BackendCreator]("Attention Backend")


# Default, dependency-free backend: pure-torch GQA attention that runs on the
# XPU (and CPU). This is what "auto" resolves to on Intel, so the engine loop
# works without triton-intel or a SYCL fork.
@SUPPORTED_ATTENTION_BACKENDS.register(
    "torch",
    BackendInfo(
        supported_types=frozenset({AttnType.FULL, AttnType.SWA}),
    ),
)
def create_torch_backend(config):
    from .triton import TritonAttentionBackend

    return TritonAttentionBackend(config)


@SUPPORTED_ATTENTION_BACKENDS.register(
    "triton",
    BackendInfo(
        supported_types=frozenset({AttnType.FULL, AttnType.SWA}),
        requires_triton_intel=True,
        consumes_attn_spec=True,
    ),
)
def create_triton_backend(config):
    from .triton import TritonAttentionBackend

    return TritonAttentionBackend(config)


@SUPPORTED_ATTENTION_BACKENDS.register(
    "sycl",
    BackendInfo(
        supported_types=frozenset({AttnType.FULL, AttnType.SWA}),
        requires_sycl=True,
    ),
)
def create_sycl_backend(config):
    from .sycl import SyclAttentionBackend

    return SyclAttentionBackend(config)


def attention_backend_info(name: str) -> BackendInfo:
    return SUPPORTED_ATTENTION_BACKENDS.info(name)


def validate_attn_backend(backend: str, allow_auto: bool = True):
    if backend != "auto":
        parts = backend.split(",")
        if len(parts) > 2:
            from argparse import ArgumentTypeError

            raise ArgumentTypeError(
                f"At most two comma-separated attention backends are allowed "
                f"(prefill,decode), got {backend!r}"
            )
        SUPPORTED_ATTENTION_BACKENDS.assert_supported(parts)
    else:
        assert allow_auto, "auto is not allowed here"
    return backend


def create_attention_backend(backend: str, config) -> BaseAttnBackend:
    if backend == "auto":
        backend = "torch"
    validate_attn_backend(backend, allow_auto=False)
    if "," in backend:
        p_backend, d_backend = backend.split(",", 1)
        if p_backend != d_backend:
            logger.info(f"Using hybrid attention backend: prefill={p_backend}, decode={d_backend}")
            return HybridBackend(
                create_attention_backend(p_backend, config),
                create_attention_backend(d_backend, config),
            )
        backend = p_backend
    return SUPPORTED_ATTENTION_BACKENDS[backend](config)


__all__ = [
    "AttnType",
    "BackendInfo",
    "BaseAttnMetadata",
    "BaseAttnBackend",
    "AttentionSpec",
    "attention_backend_info",
    "create_attention_backend",
    "SUPPORTED_ATTENTION_BACKENDS",
    "validate_attn_backend",
]
