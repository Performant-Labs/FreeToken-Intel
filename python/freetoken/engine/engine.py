"""Main engine loop: prefill streaming, decode, cache rebuild.

Upstream NVIDIA path: python/freetoken/engine/engine.py
Fill in: GitHub issue `engine-loop` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


class Engine:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def generate(self, *args, **kwargs):
        unimplemented("Engine.generate", "engine-loop")

    def add_request(self, *args, **kwargs):
        unimplemented("Engine.add_request", "engine-loop")

    def step(self, *args, **kwargs):
        unimplemented("Engine.step", "engine-loop")

    def rebuild_cache(self, *args, **kwargs):
        unimplemented("Engine.rebuild_cache", "engine-loop")

class ForwardOutput:
    def __init__(self, *args, **kwargs) -> None:
        pass

