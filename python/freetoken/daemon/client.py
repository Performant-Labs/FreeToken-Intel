"""Daemon client used by `ft ctl`.

Upstream NVIDIA path: python/freetoken/daemon/client.py
Fill in: GitHub issue `shell-daemon` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


class DaemonClient:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def status(self, *args, **kwargs):
        unimplemented("DaemonClient.status", "shell-daemon")

