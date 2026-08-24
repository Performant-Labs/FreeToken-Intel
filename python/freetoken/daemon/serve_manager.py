"""Spawn / restart `ft serve` children.

Upstream NVIDIA path: python/freetoken/daemon/serve_manager.py
Fill in: GitHub issue `shell-daemon` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


class ServeManager:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def start(self, *args, **kwargs):
        unimplemented("ServeManager.start", "shell-daemon")

    def stop(self, *args, **kwargs):
        unimplemented("ServeManager.stop", "shell-daemon")

