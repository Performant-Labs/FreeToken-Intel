"""Tensor-parallel collectives. Upstream uses NCCL; this port uses oneCCL.

Upstream NVIDIA path: python/freetoken/distributed/impl.py
Fill in: GitHub issue `oneccl-tp` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


def init_distributed(*args, **kwargs):
    unimplemented("init_distributed", "oneccl-tp")
def all_reduce(*args, **kwargs):
    unimplemented("all_reduce", "oneccl-tp")
def all_gather(*args, **kwargs):
    unimplemented("all_gather", "oneccl-tp")

