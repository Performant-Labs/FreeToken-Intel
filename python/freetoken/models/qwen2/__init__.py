"""Model package stub: qwen2.

Upstream NVIDIA path: python/freetoken/models/qwen2/
Fill in: GitHub issue `models-dense` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


def parse_config(*args, **kwargs):
    unimplemented("parse_config", "models-dense")
def iter_weights(*args, **kwargs):
    unimplemented("iter_weights", "models-dense")
class Qwen2ForCausalLM:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(self, *args, **kwargs):
        unimplemented("Qwen2ForCausalLM.forward", "models-dense")

