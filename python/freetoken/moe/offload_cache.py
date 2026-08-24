"""Global LRU expert cache (XPU slots + host banks).

Upstream NVIDIA path: python/freetoken/moe/offload_cache.py
Fill in: GitHub issue `moe-offload` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


class OffloadMoeCache:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def ensure_experts(self, *args, **kwargs):
        unimplemented("OffloadMoeCache.ensure_experts", "moe-offload")

    def copy_missing(self, *args, **kwargs):
        unimplemented("OffloadMoeCache.copy_missing", "moe-offload")

