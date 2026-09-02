"""``ft ctl`` — inspect a running server.

Upstream NVIDIA path: python/freetoken/control_cli.py
Issue: `shell-daemon` (#27, see docs/architecture.md).

Talks to the daemon's control-plane app (:mod:`freetoken.daemon.app`) via
:class:`freetoken.daemon.client.DaemonClient` -- plain JSON-over-HTTP, no
torch import anywhere on this path, so ``ft ctl`` never needs an XPU (or
even a GPU-capable machine) to query/manage a server running elsewhere.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import TextIO

from freetoken.daemon.client import DaemonClient, DaemonConnectionError, DEFAULT_BASE_URL


def _parse_args(argv: list[str], prog: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=prog, description="Query and manage a running ft daemon.")
    parser.add_argument(
        "--daemon-url",
        default=DEFAULT_BASE_URL,
        help=f"daemon control-plane base URL (default: {DEFAULT_BASE_URL})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show whether ft serve is running, and its metadata")

    start_p = sub.add_parser("start", help="Start ft serve under the daemon's supervision")
    start_p.add_argument("model", help="model reference (HF repo id, FTW path, or registered name)")
    start_p.add_argument("--host", default="127.0.0.1")
    start_p.add_argument("--port", type=int, default=8080)

    sub.add_parser("stop", help="Stop the supervised ft serve process")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None, prog: str = "ft ctl", out: TextIO | None = None) -> int:
    stream = out if out is not None else sys.stdout
    argv = list(argv) if argv is not None else sys.argv[1:]
    try:
        args = _parse_args(argv, prog=prog)
    except SystemExit as exc:
        return int(exc.code if exc.code is not None else 0)

    client = DaemonClient(args.daemon_url)
    try:
        if args.command == "status":
            result = client.status()
        elif args.command == "start":
            result = client.start(args.model, host=args.host, port=args.port)
        elif args.command == "stop":
            result = client.stop()
        else:  # pragma: no cover -- argparse's `required=True` rules this out
            return 2
    except DaemonConnectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    stream.write(json.dumps(result, indent=2) + "\n")
    return 0
