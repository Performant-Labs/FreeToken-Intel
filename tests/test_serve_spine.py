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


def test_launch_server_blocked_without_xpu_never_tracebacks(capsys):
    # On a box without torch.xpu the device layer (the first one) errors, so
    # the walk stops there with exit 1. On an XPU box the same call stops at
    # the first stub — also exit 1. Both are "blocked", so the exit code and
    # the no-traceback guarantee hold on every machine.
    from freetoken.server import launch_server

    capsys.readouterr()
    code = launch_server(["qwen3-30b-a3b"])
    out = capsys.readouterr().out
    assert code == EXIT_BLOCKED
    assert "ft serve qwen3-30b-a3b" in out
    # No XPU -> [error] device line (a misconfig, no issue pointer); an XPU
    # box -> the walk continues and stops at a [stub] layer with a blocked:
    # line. The exit code is EXIT_BLOCKED on both.
    if "[error] device" in out:
        assert "blocked:" not in out
        assert "ft device" in out
    else:
        assert "[stub]" in out
        assert "blocked:" in out
    assert "Traceback" not in out


def test_device_report_line_is_shared():
    # The spine and `ft device` must never drift: both read the same helper.
    assert isinstance(device_report_line(), str)
    assert device_report_line()  # non-empty on every machine
