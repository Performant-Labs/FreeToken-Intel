"""ServeManager: spawn/stop/status of one supervised child process
(issue `shell-daemon`, #27).

Uses trivial `python -c ...` children instead of a real `ft serve` (which
needs an XPU + a real checkpoint) -- ServeManager is deliberately generic
over the child argv for exactly this reason.
"""
from __future__ import annotations

import sys
import time

import pytest

from freetoken.daemon.serve_manager import ServeManager


def _sleep_argv(seconds: float) -> list[str]:
    return [sys.executable, "-c", f"import time; time.sleep({seconds})"]


def test_status_when_nothing_started():
    mgr = ServeManager()
    status = mgr.status()
    assert status == {"running": False, "pid": None, "argv": [], "meta": {}, "uptime_s": None}


def test_start_reports_running_with_pid_and_meta():
    mgr = ServeManager()
    mgr.start(_sleep_argv(5), meta={"model": "tiny-test"})
    try:
        status = mgr.status()
        assert status["running"] is True
        assert isinstance(status["pid"], int)
        assert status["meta"] == {"model": "tiny-test"}
        assert status["uptime_s"] >= 0
    finally:
        mgr.stop()


def test_stop_terminates_the_child_and_status_reports_not_running():
    mgr = ServeManager()
    mgr.start(_sleep_argv(30))
    mgr.stop(timeout=5)
    assert mgr.status()["running"] is False


def test_stop_is_a_noop_when_nothing_running():
    mgr = ServeManager()
    mgr.stop()  # must not raise
    assert mgr.status()["running"] is False


def test_start_raises_if_already_running():
    mgr = ServeManager()
    mgr.start(_sleep_argv(5))
    try:
        with pytest.raises(RuntimeError, match="already running"):
            mgr.start(_sleep_argv(5))
    finally:
        mgr.stop()


def test_status_reflects_a_child_that_exited_on_its_own():
    mgr = ServeManager()
    mgr.start([sys.executable, "-c", "pass"])  # exits almost immediately
    deadline = time.monotonic() + 5.0
    while mgr.status()["running"] and time.monotonic() < deadline:
        time.sleep(0.05)
    assert mgr.status()["running"] is False
