"""Spawn / restart `ft serve` children.

Upstream NVIDIA path: python/freetoken/daemon/serve_manager.py
Issue: `shell-daemon` (#27, see docs/architecture.md).

``ServeManager`` supervises one child process via plain ``subprocess``
(torch-free, works whether or not the daemon process itself ever imports
torch). It is generic over the child's argv rather than hardcoded to
build ``ft serve``'s specific command line -- :mod:`freetoken.daemon.app`
is what constructs the real ``[sys.executable, "-m", "freetoken.cli",
"serve", model, ...]`` argv, which keeps this class unit-testable with a
trivial subprocess instead of a real ``ft serve`` (which needs an XPU and
a real checkpoint).
"""
from __future__ import annotations

import subprocess
import time
from typing import Optional


class ServeManager:
    """Starts, stops, and reports the status of one supervised child process."""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._argv: list[str] = []
        self._started_at: Optional[float] = None
        self._meta: dict = {}

    def start(self, argv: list[str], *, meta: Optional[dict] = None) -> None:
        """Spawn ``argv`` as the supervised child. Raises ``RuntimeError`` if
        a child is already running (``stop()`` it first)."""
        if self.is_running():
            raise RuntimeError("a child process is already running; stop it first")
        self._proc = subprocess.Popen(argv)
        self._argv = list(argv)
        self._started_at = time.monotonic()
        self._meta = dict(meta) if meta else {}

    def stop(self, timeout: float = 10.0) -> None:
        """Terminate the supervised child (SIGTERM, then SIGKILL after
        ``timeout`` seconds). A no-op if nothing is running."""
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=timeout)
        self._proc = None
        self._argv = []
        self._started_at = None
        self._meta = {}

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def status(self) -> dict:
        """JSON-able status: what ``ft ctl`` / the daemon's ``/status`` route
        report. ``uptime_s`` and the rest are ``None``/empty when nothing is
        running (a stopped/never-started child is not an error)."""
        running = self.is_running()
        return {
            "running": running,
            "pid": self._proc.pid if running else None,
            "argv": self._argv if running else [],
            "meta": self._meta if running else {},
            "uptime_s": (time.monotonic() - self._started_at) if running else None,
        }
