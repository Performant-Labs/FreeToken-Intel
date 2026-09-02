"""Daemon control-plane app routing (issue `shell-daemon`, #27).

Route wiring is tested against a fake ServeManager double (fast, no real
subprocess) -- the real ServeManager lifecycle is covered separately in
tests/test_daemon_serve_manager.py, and a real end-to-end spawn/stop through
this app is covered in tests/test_daemon_client.py's live-daemon fixture.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from freetoken.daemon.app import create_app


class _FakeServeManager:
    def __init__(self) -> None:
        self.started_with: list[str] | None = None
        self.started_meta: dict | None = None
        self.stopped = False
        self._running = False
        self._raise_on_start: Exception | None = None

    def start(self, argv, *, meta=None) -> None:
        if self._raise_on_start is not None:
            raise self._raise_on_start
        self.started_with = argv
        self.started_meta = meta
        self._running = True

    def stop(self) -> None:
        self.stopped = True
        self._running = False

    def status(self) -> dict:
        return {"running": self._running, "pid": 1234 if self._running else None, "argv": [], "meta": {}, "uptime_s": 0.0 if self._running else None}


def test_status_route_reflects_not_running():
    fake = _FakeServeManager()
    client = TestClient(create_app(fake))
    resp = client.get("/status")
    assert resp.status_code == 200
    assert resp.json()["running"] is False


def test_start_route_builds_ft_serve_argv_and_reports_running():
    fake = _FakeServeManager()
    client = TestClient(create_app(fake))
    resp = client.post("/start", json={"model": "tiny-model", "host": "127.0.0.1", "port": 9090})
    assert resp.status_code == 200
    assert resp.json()["running"] is True
    assert "freetoken.cli" in fake.started_with
    assert "serve" in fake.started_with
    assert "tiny-model" in fake.started_with
    assert "9090" in fake.started_with
    assert fake.started_meta == {"model": "tiny-model", "host": "127.0.0.1", "port": 9090}


def test_start_route_passes_through_extra_args():
    fake = _FakeServeManager()
    client = TestClient(create_app(fake))
    client.post("/start", json={"model": "m", "extra_args": ["--moe-backend", "cpu"]})
    assert fake.started_with[-2:] == ["--moe-backend", "cpu"]


def test_start_route_returns_409_when_already_running():
    fake = _FakeServeManager()
    fake._raise_on_start = RuntimeError("a child process is already running; stop it first")
    client = TestClient(create_app(fake))
    resp = client.post("/start", json={"model": "m"})
    assert resp.status_code == 409


def test_stop_route_stops_and_reports_not_running():
    fake = _FakeServeManager()
    fake._running = True
    client = TestClient(create_app(fake))
    resp = client.post("/stop")
    assert resp.status_code == 200
    assert fake.stopped is True
    assert resp.json()["running"] is False


def test_create_app_defaults_to_a_real_serve_manager():
    from freetoken.daemon.serve_manager import ServeManager

    app = create_app()
    assert isinstance(app.state.serve_manager, ServeManager)
