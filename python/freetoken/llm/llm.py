"""High-level in-process LLM helper.

Upstream NVIDIA path: python/freetoken/llm/llm.py
Fill in: GitHub issue `engine-loop` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


class LLM:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def generate(self, *args, **kwargs):
        unimplemented("LLM.generate", "engine-loop")

