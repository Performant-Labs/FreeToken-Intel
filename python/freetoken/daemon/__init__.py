"""Persistent supervisor (torch-free). Upstream daemon package.

Upstream NVIDIA path: python/freetoken/daemon/__init__.py
Issue: `shell-daemon` (#27, see docs/architecture.md).

``ft daemon`` runs the control-plane app (:mod:`freetoken.daemon.app`) that
starts/stops/reports on one supervised ``ft serve`` child
(:mod:`freetoken.daemon.serve_manager`). Entirely torch-free by itself --
the daemon process never imports torch; only the ``ft serve`` *child* it
spawns does, in its own process.
"""
from __future__ import annotations

import argparse
import os
import sys

from .app import create_app

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8500


def _parse_args(argv: list[str], prog: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run the FreeToken-Intel daemon: a persistent supervisor for `ft serve`.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"control-plane bind address (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"control-plane bind port (default: {DEFAULT_PORT})")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, prog: str = "ft daemon") -> int:
    """Entry point for ``ft daemon``. Returns a process exit code.

    Serves the control-plane app with uvicorn until the process is stopped.
    Mirrors ``server/api_server.py::run_api_server``'s test-harness guard:
    under pytest (``PYTEST_CURRENT_TEST`` set) the app is built to confirm
    wiring but never bound to a port -- uvicorn would otherwise block the
    test runner. Test code that needs a live daemon drives ``create_app`` +
    ``TestClient`` directly.
    """
    argv = list(argv) if argv is not None else sys.argv[1:]
    try:
        args = _parse_args(argv, prog=prog)
    except SystemExit as exc:
        return int(exc.code if exc.code is not None else 0)

    app = create_app()
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return 0
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0
