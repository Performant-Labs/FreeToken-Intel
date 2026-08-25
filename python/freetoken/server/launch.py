"""``ft serve`` spine: run the end-to-end path, fail loud at the deepest stub.

The project's end command is ``ft serve <model>`` — an OpenAI-compatible
endpoint served from B70 VRAM. Until the port is complete, every layer below
the device check is a stub. This module walks the layers in dependency order,
prints one ``[ok]`` / ``[stub]`` line per layer reached, and stops at the first
stub, reporting which issue owns the gap. The command's output is the project's
progress map: each product issue that lands removes one stub line.

Layer ownership (see docs/architecture.md):
  device  — real since the XPU software epic (#33)
  args    — real since ``server-openai`` (#25); parse_args returns a ServerArgs
  resolve — real (registry + create_model are wired); the model *stubs* that
            raise live in the engine layer, so this layer reports ok
  loader  — ``models-loader`` (#17) — the first stub today
  engine  — ``engine-loop`` (#14)
  server  — real since ``server-openai`` (#25); create_app builds the FastAPI app

On today's tree the walk is: [ok] device, [ok] args, [ok] resolve, then
[stub] loader -> blocked: models-loader (#17). When #17 lands the next line
reports engine-loop (#14).
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import TextIO

from freetoken._stub import NotYetImplemented
from freetoken.utils.arch import device_report_line, is_xpu_available

EXIT_OK = 0
EXIT_BLOCKED = 1
EXIT_USAGE = 2

# First-class B70 model (epic #18/#19): when the registry layer is still a
# stub, the gap reported is the issue owning this architecture.
FIRST_CLASS_ARCH = "Qwen3MoeForCausalLM"
FIRST_CLASS_ISSUE = "models-qwen3-moe (#19)"

_ISSUE_BY_LAYER = {
    "args": "server-openai (#25)",
    "resolve": FIRST_CLASS_ISSUE,
    "loader": "models-loader (#17)",
    "engine": "engine-loop (#14)",
    "server": "server-openai (#25)",
}


@dataclass(frozen=True)
class Layer:
    name: str
    check: Callable[[argparse.Namespace], None]


def _check_device(_args: argparse.Namespace) -> None:
    # Real layer since the XPU software epic. If torch.xpu is missing this
    # raises (not NotYetImplemented) so the caller reports it as a
    # misconfigured machine, not as unimplemented code.
    if not is_xpu_available():
        raise RuntimeError(
            "no XPU visible — run `ft device` and see docs/dev-setup.md "
            "(PyTorch XPU + oneAPI / Level Zero, then re-login if you were "
            "just added to the render/video groups)"
        )


# The sub-command argv the CLI handed to ``ft serve`` (everything after the
# word ``serve``). The spine's real ``parse_args`` must succeed against this
# exact list; a list (not the model string) is what argparse expects.
_SERVE_ARGV: list[str] = []


def set_serve_argv(argv: list[str]) -> None:
    """Record the serve argv so the args layer can re-parse it faithfully."""
    global _SERVE_ARGV
    _SERVE_ARGV = list(argv)


def _check_args(_args: argparse.Namespace) -> None:
    # Real layer: parse_args now returns a ServerArgs (issue #25). It must
    # succeed against the same argv the CLI parsed, or the layer is broken.
    from freetoken.server.args import parse_args

    parse_args(_SERVE_ARGV, prog="ft serve")


class _StubConfig:
    """Minimal model config carrying just the architecture string.

    ``ModelConfig`` is itself a stub (issue #18) with no attributes, so
    ``create_model`` cannot run on it; the resolve layer only needs the
    architecture to look up the class, which ``get_model_class`` does directly.
    """

    def __init__(self, architecture: str) -> None:
        self.architectures = [architecture]


def _check_resolve(_args: argparse.Namespace) -> None:
    # "Resolve" = map the first-class B70 architecture to a model class via
    # the registry. The registry and the (stub) model class are real enough
    # that this resolves today, so the layer reports ok; the model's actual
    # forward/weight stubs live in the engine layer below. An architecture
    # that is not registered is a genuine resolve failure (ValueError).
    from freetoken.models.register import get_model_class

    model_cls = get_model_class(FIRST_CLASS_ARCH, _StubConfig(FIRST_CLASS_ARCH))
    del model_cls
    return None


def _check_loader(_args: argparse.Namespace) -> None:
    from freetoken.models.loader import load_model

    try:
        load_model(_args.model)
    except NotYetImplemented as exc:
        raise NotYetImplemented(f"loader: {exc}") from exc


def _check_engine(_args: argparse.Namespace) -> None:
    # The engine stubs fire on call; invoke with None to trip the stub
    # without needing a constructed model.
    from freetoken.engine.engine import Engine

    try:
        Engine.generate(None, None)
    except NotYetImplemented:
        raise
    raise RuntimeError("engine layer changed: Engine.generate no longer raises — update the spine")


def _check_server(_args: argparse.Namespace) -> None:
    # Real layer: create_app now builds the FastAPI app (issue #25). The app
    # is created without binding a port, so the spine can confirm the wiring
    # exists without hijacking the requested port.
    from freetoken.server.api_server import create_app
    from freetoken.server.args import parse_args

    server_args = parse_args(_args.model, prog="ft serve")

    def _engine_holder():
        raise NotYetImplemented("engine loop is a stub — implement under `engine-loop` (#14)")

    try:
        create_app(server_args, _engine_holder)
    except NotYetImplemented as exc:
        raise NotYetImplemented(f"server: {exc}") from exc


LAYERS: tuple[Layer, ...] = (
    Layer("device", _check_device),
    Layer("args", _check_args),
    Layer("resolve", _check_resolve),
    Layer("loader", _check_loader),
    Layer("engine", _check_engine),
    Layer("server", _check_server),
)


def _parse_args(argv: list[str], prog: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Start the FreeToken-Intel API server on an Intel Arc Pro B70.",
    )
    parser.add_argument("model", help="model reference (HF repo id, FTW path, or registered name)")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="bind port (default: 8080)")
    return parser.parse_args(argv)


def _walk(layers: tuple[Layer, ...], args: argparse.Namespace, out: TextIO) -> int:
    for layer in layers:
        try:
            layer.check(args)
        except NotYetImplemented as exc:
            out.write(f"  [stub]  {layer.name:<8} {exc}\n")
            out.write(f"blocked: {_ISSUE_BY_LAYER[layer.name]}\n")
            return EXIT_BLOCKED
        except RuntimeError as exc:
            # A RuntimeError here means the machine is misconfigured (e.g. no
            # XPU visible), not that code is unimplemented — so no issue points
            # at it, and the message carries the remediation.
            out.write(f"  [error] {layer.name:<8} {exc}\n")
            return EXIT_BLOCKED
        out.write(f"  [ok]    {layer.name:<8} {'done' if layer.name != 'device' else device_report_line()}\n")
    out.write("all layers live — server starting\n")
    return EXIT_OK


def launch_server(argv: list[str] | None = None, prog: str = "ft serve", out: TextIO = None) -> int:
    """Entry point for ``ft serve``. Returns a process exit code.

    Never raises for stub layers: a ``NotYetImplemented`` at any layer is
    printed as a ``[stub]`` line and the walk stops, pointing at the issue
    that owns the gap. Usage errors exit 2; a blocked or misconfigured walk
    exits 1. Any other exception is a real bug and propagates.

    Note: the layer *checks* import torch-dependent modules lazily (inside the
    check functions), so importing this module — and therefore running the
    ``--help`` / usage-error paths — works on a CPU-only machine with no
    torch installed.
    """
    stream = out if out is not None else sys.stdout
    argv = list(argv) if argv is not None else []
    set_serve_argv(argv)
    # argparse prints to the real sys.stdout and raises SystemExit on both
    # --help (0) and usage errors (2); honor whatever it chose.
    try:
        args = _parse_args(argv, prog=prog)
    except SystemExit as exc:
        return int(exc.code if exc.code is not None else EXIT_OK)

    stream.write(f"ft serve {args.model}\n")
    result = _walk(LAYERS, args, stream)
    if result != EXIT_OK:
        from freetoken.version import __version__

        stream.write(f"(freetoken-intel {__version__})\n")
        return result

    # Every layer is live: hand over to the real server. This path is reached
    # only once loader (#17) and engine (#14) are implemented, so it imports
    # the torch-bound stack only now.
    stream.write("all layers live — starting server\n")
    from freetoken.server.args import parse_args
    from freetoken.server.api_server import run_api_server

    server_args = parse_args(_SERVE_ARGV, prog=prog)

    def _engine_holder():
        # Real wiring: load the model onto the XPU host banks and wrap it in
        # the engine loop. Both land under #17/#14; until then the spine
        # never reaches this line.
        from freetoken.engine.engine import Engine
        from freetoken.models import create_model
        from freetoken.models.config import ModelConfig

        create_model(ModelConfig())
        Engine()

    return run_api_server(server_args, _engine_holder)
