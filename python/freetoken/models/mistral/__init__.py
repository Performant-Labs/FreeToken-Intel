"""Model package stub: mistral.

Upstream NVIDIA path: python/freetoken/models/mistral/
Fill in: GitHub issue `models-dense` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


def parse_config(*args, **kwargs):
    unimplemented("parse_config", "models-dense")
def iter_weights(*args, **kwargs):
    unimplemented("iter_weights", "models-dense")
class MistralForCausalLM:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(self, *args, **kwargs):
        unimplemented("MistralForCausalLM.forward", "models-dense")

