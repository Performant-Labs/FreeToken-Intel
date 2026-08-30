"""FastAPI app wiring.

Upstream NVIDIA path: python/freetoken/server/api_server.py

``create_app`` builds the app and mounts the OpenAI routes. The model loader
and engine are *not* touched here — the routes resolve them lazily per
request (see ``openai_api.engine_holder``), so building the app is cheap and
safe on a CPU box. ``run_api_server`` is the blocking entry point that
``ft serve`` calls once the spine's layer walk has confirmed the loader and
engine are live.
"""
from __future__ import annotations

import logging
import threading

from fastapi import FastAPI

from freetoken.version import __version__
from freetoken.server import generation
from freetoken.server.args import ServerArgs
from freetoken.server.anthropic_api import register_anthropic_routes
from freetoken.server.openai_api import register_openai_routes

logger = logging.getLogger(__name__)


def _build_frontend_tokenizer_hook(server_args: ServerArgs):
    """A thread-safe, lazy resolver for the message frontend (``#95``).

    Returns a zero-arg callable that loads the model's tokenizer exactly once
    (``AutoTokenizer`` — no torch, no XPU) and wraps it in a
    :class:`TokenizeManager`, caching the result for the process lifetime. The
    tokenizer is loaded on first call (the first chat request), never at app-build
    time, so ``create_app`` and ``import freetoken.server.api_server`` stay
    cheap and torch-free on a CPU box. ``server_args.model_path`` is the HF repo
    id / local checkpoint path the loader already keys on.
    """
    lock = threading.Lock()
    state = {"mgr": None}

    def resolve():
        with lock:
            if state["mgr"] is None:
                from freetoken.tokenizer.tokenize import TokenizeManager
                from freetoken.utils import load_tokenizer

                state["mgr"] = TokenizeManager(load_tokenizer(server_args.model_path))
            return state["mgr"]

    return resolve


def create_app(server_args: ServerArgs, engine_holder) -> FastAPI:
    """Build the serving app.

    ``engine_holder`` is a zero-arg callable returning the loaded ``Engine``
    (raising ``NotYetImplemented`` while the loader/engine are still stubs).
    It is stored on ``app.state`` and invoked by the routes per request.
    """
    # The version is imported, never re-declared: freetoken/version.py is the
    # single source of truth (enforced by the conformance job in ci.yml -- a
    # second literal version here would drift and fail the build).
    app = FastAPI(
        title="FreeToken-Intel",
        description="Edge-native MoE serving on Intel Arc Pro B70",
        version=__version__,
    )
    app.state.server_args = server_args
    app.state.engine_holder = engine_holder
    # The message frontend (chat-template encode / incremental decode, #95) is
    # resolved lazily per request. Installing the hook here (not loading the
    # tokenizer) keeps app-build cheap and torch-free; the tokenizer loads on
    # the first chat request. The launch holder also attaches the same manager
    # to the engine, so a loaded engine resolves directly and this hook is the
    # fallback for the route surface.
    generation.set_frontend_tokenizer_hook(_build_frontend_tokenizer_hook(server_args))
    register_openai_routes(app, engine_holder)
    register_anthropic_routes(app, engine_holder)
    return app


def run_api_server(server_args: ServerArgs, engine_holder) -> int:
    """Build the app and serve it with uvicorn until the process is stopped.

    Returns a process exit code (uvicorn returns 0 on a clean shutdown).

    When the process is already running under a test harness (``PYTEST_CURRENT_TEST``
    is set) the app is built to confirm the wiring but *not* bound to a port --
    uvicorn would otherwise block the test runner. Test code that needs a live
    server drives ``create_app`` + ``TestClient`` directly.
    """
    import os

    import uvicorn

    from freetoken.server.args import DEFAULT_HOST, DEFAULT_PORT

    host = server_args.server_host or DEFAULT_HOST
    port = server_args.server_port or DEFAULT_PORT
    app = create_app(server_args, engine_holder)
    if os.environ.get("PYTEST_CURRENT_TEST"):
        logger.info("FreeToken-Intel app built (test harness -- not serving).")
        return 0
    logger.info("FreeToken-Intel serving %s on http://%s:%d", server_args.resolved_model_name, host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


__all__ = ["create_app", "run_api_server"]
