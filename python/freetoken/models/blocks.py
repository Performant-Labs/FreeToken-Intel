"""Base LLM module.

Upstream NVIDIA path: python/freetoken/models/blocks.py
Fill in: GitHub issue `models-qwen35` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


class BaseLLMModel:
    """The real model owns a parameter set; the stub exposes an empty one so
    the loader's ``named_parameters`` contract works before the forward pass
    (issue ``models-qwen3-5`` / ``engine-loop``) is implemented."""

    def __init__(self, *args, **kwargs) -> None:
        self._parameters: dict = {}
        self._buffers: dict = {}

    def named_parameters(self, prefix: str = ""):
        yield from ((f"{prefix}.{name}" if prefix else name, value) for name, value in self._parameters.items())

    def named_buffers(self, prefix: str = ""):
        yield from ((f"{prefix}.{name}" if prefix else name, value) for name, value in self._buffers.items())

    def forward(self, *args, **kwargs):
        unimplemented("BaseLLMModel.forward", "models-qwen35")

