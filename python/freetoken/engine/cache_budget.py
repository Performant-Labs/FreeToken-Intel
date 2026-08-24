"""Elastic VRAM split between expert cache and KV (B70 has 32 GB).

Upstream NVIDIA path: python/freetoken/engine/cache_budget.py
Fill in: GitHub issue `elastic-memory` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


def plan_cache_budget(*args, **kwargs):
    unimplemented("plan_cache_budget", "elastic-memory")

