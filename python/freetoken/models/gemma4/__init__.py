"""Model package stub: gemma4.

Upstream NVIDIA path: python/freetoken/models/gemma4/
Fill in: GitHub issue `models-dense` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


def parse_config(*args, **kwargs):
    unimplemented("parse_config", "models-dense")
def iter_weights(*args, **kwargs):
    unimplemented("iter_weights", "models-dense")
class Gemma4ForCausalLM:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(self, *args, **kwargs):
        unimplemented("Gemma4ForCausalLM.forward", "models-dense")

