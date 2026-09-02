"""DaemonClient + a real end-to-end daemon spawn/stop cycle (issue `shell-daemon`, #27).

Stands up the real daemon control-plane app (the same object `ft daemon`
runs) on a real loopback socket -- same pattern as
tests/test_launch_cli_cpu.py's `live_server` fixture -- so DaemonClient's
urllib HTTP layer is exercised end-to-end, not mocked. `start`/`stop` spawn
a real (trivial, short-lived) child process through the real ServeManager,
proving the whole "daemon starts/stops a child and exposes status" path
without needing a real `ft serve` + XPU + checkpoint.
"""
from __future__ import annotations

import socket
import sys
import threading
import time
from urllib.request import urlopen

import pytest

from freetoken.daemon.app import create_app
from freetoken.daemon.client import DaemonClient, DaemonConnectionError
from freetoken.daemon.serve_manager import ServeManager


class _SleepServeManager(ServeManager):
    """A ServeManager whose start() ignores the real `ft serve` argv the
    daemon app builds and spawns a trivial sleep child instead -- the route
    wiring under test is genuinely exercised (real HTTP request in, real
    subprocess started/stopped), just not a real (XPU-needing) `ft serve`."""

    def start(self, argv, *, meta=None) -> None:
        del argv
        super().start([sys.executable, "-c", "import time; time.sleep(30)"], meta=meta)


@pytest.fixture
def live_daemon():
    import uvicorn

    app = create_app(_SleepServeManager())

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            with urlopen(f"http://127.0.0.1:{port}/status", timeout=0.5) as resp:
                if resp.status == 200:
                    break
        except OSError:
            time.sleep(0.05)
    else:
        raise RuntimeError("daemon did not come up for the client test")

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        app.state.serve_manager.stop()
        server.should_exit = True
        thread.join(timeout=5)


def test_status_before_start(live_daemon):
    client = DaemonClient(live_daemon)
    assert client.status()["running"] is False


def test_start_then_status_then_stop(live_daemon):
    client = DaemonClient(live_daemon)
    started = client.start("tiny-test-model", port=9191)
    assert started["running"] is True
    assert isinstance(started["pid"], int)

    status = client.status()
    assert status["running"] is True
    assert status["meta"]["model"] == "tiny-test-model"

    stopped = client.stop()
    assert stopped["running"] is False


def test_connection_error_when_nothing_listening():
    client = DaemonClient("http://127.0.0.1:1")  # port 1: nothing listens here
    with pytest.raises(DaemonConnectionError):
        client.status()
