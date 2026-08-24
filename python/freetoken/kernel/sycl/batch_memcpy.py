"""Batched XPU memcpy (Level Zero). Upstream: cudaMemcpyBatchAsync.

Upstream NVIDIA path: python/freetoken/kernel/batch_memcpy.py
Fill in: GitHub issue `moe-offload` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


def batch_memcpy(*args, **kwargs):
    unimplemented("batch_memcpy", "moe-offload")

