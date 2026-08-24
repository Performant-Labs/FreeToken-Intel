"""Prefix-sharing radix KV cache.

Upstream NVIDIA path: python/freetoken/kvcache/radix_cache.py
Fill in: GitHub issue `kvcache` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


class RadixCache:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def match_prefix(self, *args, **kwargs):
        unimplemented("RadixCache.match_prefix", "kvcache")

    def insert(self, *args, **kwargs):
        unimplemented("RadixCache.insert", "kvcache")

