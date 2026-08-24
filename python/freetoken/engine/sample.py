"""Sampling (greedy / top-k / top-p) on XPU.

Upstream NVIDIA path: python/freetoken/engine/sample.py
Fill in: GitHub issue `engine-loop` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


class BatchSamplingArgs:
    def __init__(self, *args, **kwargs) -> None:
        pass
def sample(*args, **kwargs):
    unimplemented("sample", "engine-loop")

