"""Native SYCL attention on Xe2. Replaces FlashInfer / sgl-kernel CUDA.

Upstream NVIDIA path: python/freetoken/attention/fi.py + fa.py
Fill in: GitHub issue `attn-sycl` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


class SyclAttentionBackend:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(self, *args, **kwargs):
        unimplemented("SyclAttentionBackend.forward", "attn-sycl")

    def prepare_metadata(self, *args, **kwargs):
        unimplemented("SyclAttentionBackend.prepare_metadata", "attn-sycl")

    def init_capture_graph(self, *args, **kwargs):
        unimplemented("SyclAttentionBackend.init_capture_graph", "attn-sycl")

    def prepare_for_capture(self, *args, **kwargs):
        unimplemented("SyclAttentionBackend.prepare_for_capture", "attn-sycl")

    def prepare_for_replay(self, *args, **kwargs):
        unimplemented("SyclAttentionBackend.prepare_for_replay", "attn-sycl")

