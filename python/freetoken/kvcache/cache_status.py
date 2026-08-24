"""Runtime cache occupancy / rebuild status.

Upstream NVIDIA path: python/freetoken/kvcache/cache_status.py
Fill in: GitHub issue `elastic-memory` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


class CacheStatus:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def report(self, *args, **kwargs):
        unimplemented("CacheStatus.report", "elastic-memory")

