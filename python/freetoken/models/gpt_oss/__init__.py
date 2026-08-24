"""Model package stub: gpt_oss.

Upstream NVIDIA path: python/freetoken/models/gpt_oss/
Fill in: GitHub issue `models-gpt-oss` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


def parse_config(*args, **kwargs):
    unimplemented("parse_config", "models-gpt-oss")
def iter_weights(*args, **kwargs):
    unimplemented("iter_weights", "models-gpt-oss")
class GptOssForCausalLM:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(self, *args, **kwargs):
        unimplemented("GptOssForCausalLM.forward", "models-gpt-oss")

