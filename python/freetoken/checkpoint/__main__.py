"""``ft checkpoint`` CLI: convert an HF safetensors checkpoint to FTW.

Upstream NVIDIA path: python/freetoken/checkpoint/__main__.py
Issue: `ftw-checkpoint` (#11, see docs/architecture.md).

Usage: ``python -m freetoken.checkpoint convert <model_path> <output_path>``
"""
from __future__ import annotations

import argparse
import sys

from .convert import convert_checkpoint, convert_summary


def main(argv: list[str] | None = None, prog: str = "ft checkpoint") -> int:
    parser = argparse.ArgumentParser(prog=prog)
    sub = parser.add_subparsers(dest="command", required=True)

    convert_p = sub.add_parser("convert", help="Convert a safetensors checkpoint to FTW")
    convert_p.add_argument("model_path", help="Source checkpoint directory (HF safetensors)")
    convert_p.add_argument("output_path", help="Destination FTW directory")

    argv = list(argv) if argv is not None else sys.argv[1:]
    # argparse prints to stdout/stderr and raises SystemExit on both --help
    # (0) and usage errors (2); honor whatever it chose rather than letting
    # it propagate past this function (matches server/launch.py's pattern).
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code if exc.code is not None else 0)
    if args.command == "convert":
        convert_checkpoint(args.model_path, args.output_path)
        summary = convert_summary(args.output_path)
        print(
            f"Wrote FTW archive: {summary['tensors']} tensors, "
            f"{summary['total_bytes'] / (1024 * 1024):.1f} MiB -> {summary['path']}"
        )
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
