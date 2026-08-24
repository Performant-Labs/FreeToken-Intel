"""Prefill/decode scheduler with chunked prefill.

Upstream NVIDIA path: python/freetoken/scheduler/scheduler.py
Fill in: GitHub issue `scheduler` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


class Scheduler:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def schedule(self, *args, **kwargs):
        unimplemented("Scheduler.schedule", "scheduler")

    def abort(self, *args, **kwargs):
        unimplemented("Scheduler.abort", "scheduler")

