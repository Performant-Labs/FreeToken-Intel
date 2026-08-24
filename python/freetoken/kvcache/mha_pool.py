"""Paged MHA/GQA KV pool on XPU.

Upstream NVIDIA path: python/freetoken/kvcache/mha_pool.py
Fill in: GitHub issue `kvcache` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


class MHAKVCache:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def allocate(self, *args, **kwargs):
        unimplemented("MHAKVCache.allocate", "kvcache")

