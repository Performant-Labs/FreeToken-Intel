"""Locate icpx / oneAPI / Level Zero for JIT SYCL.

Upstream NVIDIA path: python/freetoken/kernel/_toolchain.py
Fill in: GitHub issue `kernel-sycl` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


def find_icpx(*args, **kwargs):
    unimplemented("find_icpx", "kernel-sycl")
def sycl_flags(*args, **kwargs):
    unimplemented("sycl_flags", "kernel-sycl")

