"""Triton-Intel attention (XPU). Upstream CUDA Triton attention.

Upstream NVIDIA path: python/freetoken/attention/triton.py
Fill in: GitHub issue `attn-triton` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


class TritonAttentionBackend:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(self, *args, **kwargs):
        unimplemented("TritonAttentionBackend.forward", "attn-triton")

    def prepare_metadata(self, *args, **kwargs):
        unimplemented("TritonAttentionBackend.prepare_metadata", "attn-triton")

    def init_capture_graph(self, *args, **kwargs):
        unimplemented("TritonAttentionBackend.init_capture_graph", "attn-triton")

    def prepare_for_capture(self, *args, **kwargs):
        unimplemented("TritonAttentionBackend.prepare_for_capture", "attn-triton")

    def prepare_for_replay(self, *args, **kwargs):
        unimplemented("TritonAttentionBackend.prepare_for_replay", "attn-triton")

