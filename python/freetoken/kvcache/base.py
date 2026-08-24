"""KV pool / handle interfaces.

Upstream NVIDIA path: python/freetoken/kvcache/base.py
Fill in: GitHub issue `kvcache` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


class BaseKVCachePool:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def allocate(self, *args, **kwargs):
        unimplemented("BaseKVCachePool.allocate", "kvcache")

    def free(self, *args, **kwargs):
        unimplemented("BaseKVCachePool.free", "kvcache")
class BaseCacheHandle:
    def __init__(self, *args, **kwargs) -> None:
        pass

