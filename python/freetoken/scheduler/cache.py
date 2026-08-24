"""Scheduler cache manager (radix / SWA / rebuild).

Upstream NVIDIA path: python/freetoken/scheduler/cache.py
Fill in: GitHub issue `kvcache` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


class CacheManager:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def match(self, *args, **kwargs):
        unimplemented("CacheManager.match", "kvcache")

    def commit(self, *args, **kwargs):
        unimplemented("CacheManager.commit", "kvcache")

