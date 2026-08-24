"""SYCL / Level Zero graph capture for decode (upstream: CUDA graphs).

Upstream NVIDIA path: python/freetoken/engine/graph.py
Fill in: GitHub issue `engine-graph` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


class XpuGraphRunner:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def capture(self, *args, **kwargs):
        unimplemented("XpuGraphRunner.capture", "engine-graph")

    def replay(self, *args, **kwargs):
        unimplemented("XpuGraphRunner.replay", "engine-graph")

