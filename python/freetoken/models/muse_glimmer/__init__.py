"""Model package stub: muse_glimmer.

Upstream NVIDIA path: python/freetoken/models/muse_glimmer/
Fill in: GitHub issue `models-dense` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


def parse_config(*args, **kwargs):
    unimplemented("parse_config", "models-dense")
def iter_weights(*args, **kwargs):
    unimplemented("iter_weights", "models-dense")
class MuseGlimmerForCausalLM:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(self, *args, **kwargs):
        unimplemented("MuseGlimmerForCausalLM.forward", "models-dense")

