"""`ft ctl` CLI (issue `shell-daemon`, #27)."""
from __future__ import annotations

import io
import json

import pytest

from freetoken.control_cli import main
from freetoken.daemon.client import DaemonConnectionError


def test_help_exits_zero():
    assert main(["--help"], out=io.StringIO()) == 0


def test_status_prints_daemon_response(monkeypatch):
    from freetoken import control_cli

    class _FakeClient:
        def __init__(self, base_url):
            self.base_url = base_url

        def status(self):
            return {"running": True, "pid": 42}

    monkeypatch.setattr(control_cli, "DaemonClient", _FakeClient)
    out = io.StringIO()
    assert main(["status"], out=out) == 0
    assert json.loads(out.getvalue()) == {"running": True, "pid": 42}


def test_start_forwards_model_host_port(monkeypatch):
    from freetoken import control_cli

    captured = {}

    class _FakeClient:
        def __init__(self, base_url):
            pass

        def start(self, model, *, host, port):
            captured["model"], captured["host"], captured["port"] = model, host, port
            return {"running": True}

    monkeypatch.setattr(control_cli, "DaemonClient", _FakeClient)
    out = io.StringIO()
    assert main(["start", "my-model", "--host", "0.0.0.0", "--port", "9999"], out=out) == 0
    assert captured == {"model": "my-model", "host": "0.0.0.0", "port": 9999}


def test_stop_calls_client_stop(monkeypatch):
    from freetoken import control_cli

    class _FakeClient:
        def __init__(self, base_url):
            pass

        def stop(self):
            return {"running": False}

    monkeypatch.setattr(control_cli, "DaemonClient", _FakeClient)
    out = io.StringIO()
    assert main(["stop"], out=out) == 0
    assert json.loads(out.getvalue()) == {"running": False}


def test_unreachable_daemon_exits_one(monkeypatch, capsys):
    from freetoken import control_cli

    class _FakeClient:
        def __init__(self, base_url):
            pass

        def status(self):
            raise DaemonConnectionError("could not reach daemon at http://x")

    monkeypatch.setattr(control_cli, "DaemonClient", _FakeClient)
    assert main(["status"]) == 1
    assert "could not reach daemon" in capsys.readouterr().err


def test_missing_subcommand_exits_two():
    assert main([]) == 2


def test_daemon_url_accepted_after_the_subcommand(monkeypatch):
    """PR-Agent review, PR #128: argparse only accepts a parent parser's own
    options before the subcommand by default, so `ft ctl status --daemon-url
    <url>` (the natural place to put it) used to fail with "unrecognized
    arguments"."""
    from freetoken import control_cli

    captured = {}

    class _FakeClient:
        def __init__(self, base_url):
            captured["base_url"] = base_url

        def status(self):
            return {"running": False}

    monkeypatch.setattr(control_cli, "DaemonClient", _FakeClient)
    assert main(["status", "--daemon-url", "http://example:9"], out=io.StringIO()) == 0
    assert captured["base_url"] == "http://example:9"


def test_daemon_url_before_subcommand_is_not_clobbered_by_the_subparser_default(monkeypatch):
    """A regression this fix could plausibly introduce: the subparser's own
    --daemon-url copy re-parsing over the top-level value with its own
    default instead of leaving it alone."""
    from freetoken import control_cli

    captured = {}

    class _FakeClient:
        def __init__(self, base_url):
            captured["base_url"] = base_url

        def status(self):
            return {"running": False}

    monkeypatch.setattr(control_cli, "DaemonClient", _FakeClient)
    assert main(["--daemon-url", "http://example:9", "status"], out=io.StringIO()) == 0
    assert captured["base_url"] == "http://example:9"
