"""Pinned host expert banks (USM host / Level Zero host ptrs).

Upstream NVIDIA path: python/freetoken/moe/host_banks.py
Fill in: GitHub issue `moe-offload` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


class HostBanks:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def view(self, *args, **kwargs):
        unimplemented("HostBanks.view", "moe-offload")

