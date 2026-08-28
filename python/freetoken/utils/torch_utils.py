"""Torch helpers. NVIDIA NVTX becomes ITT / oneAPI tracing on Intel."""
from __future__ import annotations

from contextlib import contextmanager


def torch_dtype(name: str):
    """Resolve a dtype by name (``"bfloat16"``, ``"float32"``, ``"int8"`` ...).

    Upstream resolves dtypes inline; this is the single named entry point the
    loader and model configs use. The name is looked up on ``torch`` (e.g.
    ``torch.bfloat16``), so the XPU build and the CPU build resolve to the same
    dtype object -- no CUDA-only assumption here.

    Raises ``ValueError`` for a name ``torch`` does not know, so a typo in a
    config surfaces at load time instead of silently defaulting.
    """
    import torch

    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(
            f"unknown dtype {name!r} (expected a torch dtype name such as 'bfloat16')"
        )
    return dtype


@contextmanager
def itt_annotate(name: str):
    """Placeholder for Intel ITT / unitrace spans (upstream: nvtx_annotate)."""
    del name
    yield
