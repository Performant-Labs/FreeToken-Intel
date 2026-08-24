"""Base LLM module.

Upstream NVIDIA path: python/freetoken/models/blocks.py
Fill in: GitHub issue `models-qwen35` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


class BaseLLMModel:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(self, *args, **kwargs):
        unimplemented("BaseLLMModel.forward", "models-qwen35")

