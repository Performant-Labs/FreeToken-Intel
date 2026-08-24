"""Hybrid sliding-window + full KV pool.

Upstream NVIDIA path: python/freetoken/kvcache/hybrid_swa_pool.py
Fill in: GitHub issue `kvcache` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


class HybridSWAKVCache:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def allocate(self, *args, **kwargs):
        unimplemented("HybridSWAKVCache.allocate", "kvcache")

