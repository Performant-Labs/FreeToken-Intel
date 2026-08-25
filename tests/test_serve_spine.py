"""Tests for the ``ft serve`` spine (issue ``serve-spine``).

The spine's contract: it walks the layer list, prints one line per layer
reached, stops at the first stub, and returns a process exit code. Stub layers
never produce tracebacks; only genuine bugs do.

The layer *walk* tests inject fake ``Layer`` objects, so they run on a
CPU-only box (no ``xpu`` marker, no torch). The ``launch_server`` tests run
the real entry point: on a box without torch.xpu the walk stops at the device
layer (exit 1), on an XPU box it stops at the first stub (exit 1) — the exit
code and the no-traceback guarantee are machine-independent, and the
``--help`` / usage paths never touch torch at all.
"""
from __future__ import annotations

import argparse
from io import StringIO

import pytest

from freetoken._stub import NotYetImplemented
from freetoken.server.launch import EXIT_BLOCKED, EXIT_OK, EXIT_USAGE, Layer, _walk
from freetoken.utils.arch import device_report_line


def _ns(**kwargs) -> argparse.Namespace:
    base = {"model": "qwen3-30b-a3b", "host": "127.0.0.1", "port": 8080}
    base.update(kwargs)
    return argparse.Namespace(**base)


def _ok_layer(name: str) -> Layer:
    def check(_args):
        return None

    return Layer(name, check)


def _stub_layer(name: str, issue: str) -> Layer:
    def check(_args):
        raise NotYetImplemented(f"{name} is a stub — implement under `{issue}`")

    return Layer(name, check)


def _error_layer(name: str, msg: str) -> Layer:
    def check(_args):
        raise RuntimeError(msg)

    return Layer(name, check)


def test_walk_stops_at_first_stub():
    layers = (_ok_layer("device"), _stub_layer("args", "server-openai (#25)"), _ok_layer("server"))
    buf = StringIO()
    code = _walk(layers, _ns(), buf)
    out = buf.getvalue()
    assert code == EXIT_BLOCKED
    assert "[ok]" in out
    assert "[stub]  args" in out
    assert "blocked: server-openai (#25)" in out
    # The layer after the stub must never have been printed.
    assert "[ok]    server" not in out


def test_walk_all_live_exits_zero():
    layers = (_ok_layer("device"), _ok_layer("args"), _ok_layer("server"))
    buf = StringIO()
    code = _walk(layers, _ns(), buf)
    out = buf.getvalue()
    assert code == EXIT_OK
    assert out.count("[ok]") == 3
    assert "all layers live" in out
    assert "[stub]" not in out


def test_walk_runtime_error_is_blocked_not_traceback():
    # The no-XPU path surfaces as RuntimeError, which the spine must render as
    # a clean [error] line, not a stack trace.
    layers = (_error_layer("device", "no XPU visible"), _ok_layer("server"))
    buf = StringIO()
    code = _walk(layers, _ns(), buf)
    out = buf.getvalue()
    assert code == EXIT_BLOCKED
    assert "[error] device" in out
    assert "no XPU visible" in out
    assert "Traceback" not in out


def test_launch_server_help_exits_zero_without_torch(capsys):
    from freetoken.server import launch_server

    capsys.readouterr()
    assert launch_server(["--help"]) == EXIT_OK
    assert "usage" in capsys.readouterr().out


def test_launch_server_usage_error_exits_two(capsys):
    from freetoken.server import launch_server

    capsys.readouterr()
    assert launch_server(["--bogus"]) == EXIT_USAGE
    assert launch_server([]) == EXIT_USAGE
    capsys.readouterr()


def test_launch_server_never_tracebacks(capsys):
    # Running ``ft serve <model>`` must never emit a Python traceback, on any
    # machine. With the engine implemented (#14) and the server layer live, the
    # walk reaches the end:
    #   - on an XPU box every layer reports [ok] and the spine starts the
    #     server -> EXIT_OK (under pytest the server is built but not bound, so
    #     this returns cleanly instead of blocking);
    #   - on a box without torch.xpu the device layer (the first one) errors,
    #     so the walk stops there with a clean [error] line -> EXIT_BLOCKED.
    # Both outcomes are "no traceback", which is the guarantee being tested.
    from freetoken.server import launch_server
    from freetoken.utils.arch import is_xpu_available

    capsys.readouterr()
    code = launch_server(["qwen3-30b-a3b"])
    out = capsys.readouterr().out
    assert "ft serve qwen3-30b-a3b" in out
    assert "Traceback" not in out
    if is_xpu_available():
        # Engine + server are live: the spine reports every layer [ok] and
        # starts (under pytest: builds the app, does not bind a port).
        assert code == EXIT_OK
        assert "[ok]    engine" in out
        assert "[ok]    server" in out
        assert "all layers live" in out
        assert "[stub]" not in out
        assert "[error]" not in out
    else:
        # No XPU: the first layer is a misconfig, so the walk is blocked with a
        # clean [error] line (no issue pointer, no traceback).
        assert code == EXIT_BLOCKED
        assert "[error] device" in out
        assert "blocked:" not in out
        assert "ft device" in out


def test_device_report_line_is_shared():
    # The spine and `ft device` must never drift: both read the same helper.
    assert isinstance(device_report_line(), str)
    assert device_report_line()  # non-empty on every machine
