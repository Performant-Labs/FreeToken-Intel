"""Model package stub: glm4_moe.

Upstream NVIDIA path: python/freetoken/models/glm4_moe/
Fill in: GitHub issue `models-glm` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


def parse_config(*args, **kwargs):
    unimplemented("parse_config", "models-glm")
def iter_weights(*args, **kwargs):
    unimplemented("iter_weights", "models-glm")
class Glm4MoeForCausalLM:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(self, *args, **kwargs):
        unimplemented("Glm4MoeForCausalLM.forward", "models-glm")

