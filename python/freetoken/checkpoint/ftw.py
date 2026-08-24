"""FTW reader/writer. Pack experts for Xe2-friendly copies.

Upstream NVIDIA path: python/freetoken/checkpoint/ftw.py
Fill in: GitHub issue `ftw-checkpoint` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


class FtwArchive:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def read(self, *args, **kwargs):
        unimplemented("FtwArchive.read", "ftw-checkpoint")

    def write(self, *args, **kwargs):
        unimplemented("FtwArchive.write", "ftw-checkpoint")

