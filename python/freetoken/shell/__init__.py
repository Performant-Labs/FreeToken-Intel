"""ft shell entry.

Upstream NVIDIA path: python/freetoken/shell/__init__.py
Issue: `shell-daemon` (#27, see docs/architecture.md).
"""
from __future__ import annotations

import argparse
import sys


def _parse_args(argv: list[str], prog: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=prog, description="Chat with a running ft-serve endpoint in the terminal.")
    parser.add_argument("--server-url", default="http://127.0.0.1:8080", help="ft serve base URL (default: http://127.0.0.1:8080)")
    parser.add_argument("--model", default="default", help="model name to send with each request")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, prog: str = "ft shell") -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    try:
        args = _parse_args(argv, prog=prog)
    except SystemExit as exc:
        return int(exc.code if exc.code is not None else 0)

    from freetoken.shell.tui import run_tui

    return run_tui(args.server_url, args.model)
