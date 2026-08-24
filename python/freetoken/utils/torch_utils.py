"""Torch helpers. NVIDIA NVTX becomes ITT / oneAPI tracing on Intel."""
from __future__ import annotations

from contextlib import contextmanager

from freetoken._stub import unimplemented


def torch_dtype(name: str):
    unimplemented("torch_dtype", "device-layer")


@contextmanager
def itt_annotate(name: str):
    """Placeholder for Intel ITT / unitrace spans (upstream: nvtx_annotate)."""
    del name
    yield
