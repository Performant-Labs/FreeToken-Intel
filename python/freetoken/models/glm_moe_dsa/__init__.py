"""Model package stub: glm_moe_dsa.

Upstream NVIDIA path: python/freetoken/models/glm_moe_dsa/
Fill in: GitHub issue `models-glm` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


def parse_config(*args, **kwargs):
    unimplemented("parse_config", "models-glm")
def iter_weights(*args, **kwargs):
    unimplemented("iter_weights", "models-glm")
class GlmMoeDsaForCausalLM:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(self, *args, **kwargs):
        unimplemented("GlmMoeDsaForCausalLM.forward", "models-glm")

