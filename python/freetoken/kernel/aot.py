"""Ahead-of-time SYCL kernel cache (replaces CUDA cubin cache).

Upstream NVIDIA path: python/freetoken/kernel/aot.py
Fill in: GitHub issue `kernel-sycl` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


def build_aot_cache(*args, **kwargs):
    unimplemented("build_aot_cache", "kernel-sycl")
def load_aot_cache(*args, **kwargs):
    unimplemented("load_aot_cache", "kernel-sycl")

