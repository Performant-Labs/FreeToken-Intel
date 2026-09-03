"""`ft daemon` entry point: arg parsing, help, and the PYTEST_CURRENT_TEST
guard that keeps tests from binding a real port (issue `shell-daemon`, #27).

PR-Agent review on PR #128 flagged that `ft ctl`/`ft shell` each had a
covering test for their entry point but `ft daemon`'s did not.
"""
from __future__ import annotations

import os

from freetoken.daemon import DEFAULT_HOST, DEFAULT_PORT, _is_loopback_host, main


def test_help_exits_zero():
    assert main(["--help"]) == 0


def test_unknown_flag_exits_two():
    assert main(["--not-a-flag"]) == 2


def test_defaults_and_under_pytest_never_binds_a_port():
    # PYTEST_CURRENT_TEST is already set by the pytest runner itself, so this
    # exercises the exact guard main() uses in real test runs -- if it were
    # ever removed, this call would hang trying to bind DEFAULT_PORT.
    assert os.environ.get("PYTEST_CURRENT_TEST")
    assert main([]) == 0


def test_custom_host_and_port_parsed(monkeypatch):
    captured = {}

    def _fake_create_app():
        captured["called"] = True
        return object()

    import freetoken.daemon as daemon_mod

    monkeypatch.setattr(daemon_mod, "create_app", _fake_create_app)
    assert main(["--host", "0.0.0.0", "--port", "9999"]) == 0
    assert captured["called"] is True


def test_module_level_defaults():
    assert DEFAULT_HOST == "127.0.0.1"
    assert DEFAULT_PORT == 8500


def test_is_loopback_host():
    assert _is_loopback_host("127.0.0.1")
    assert _is_loopback_host("::1")
    assert _is_loopback_host("localhost")
    assert not _is_loopback_host("0.0.0.0")
    assert not _is_loopback_host("192.168.1.5")
    assert not _is_loopback_host("example.com")


def test_non_loopback_host_warns(monkeypatch, capsys):
    import freetoken.daemon as daemon_mod

    monkeypatch.setattr(daemon_mod, "create_app", lambda: object())
    assert main(["--host", "0.0.0.0"]) == 0
    err = capsys.readouterr().err
    assert "not loopback" in err
    assert "not real authentication" in err


def test_loopback_host_does_not_warn(monkeypatch, capsys):
    import freetoken.daemon as daemon_mod

    monkeypatch.setattr(daemon_mod, "create_app", lambda: object())
    assert main(["--host", "127.0.0.1"]) == 0
    assert capsys.readouterr().err == ""
