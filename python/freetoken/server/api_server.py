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

from fastapi import FastAPI

from freetoken.server.args import ServerArgs
from freetoken.server.openai_api import register_openai_routes

logger = logging.getLogger(__name__)


def create_app(server_args: ServerArgs, engine_holder) -> FastAPI:
    """Build the serving app.

    ``engine_holder`` is a zero-arg callable returning the loaded ``Engine``
    (raising ``NotYetImplemented`` while the loader/engine are still stubs).
    It is stored on ``app.state`` and invoked by the routes per request.
    """
    app = FastAPI(
        title="FreeToken-Intel",
        description="Edge-native MoE serving on Intel Arc Pro B70",
        version="0.0.0",
    )
    app.state.server_args = server_args
    app.state.engine_holder = engine_holder
    register_openai_routes(app, engine_holder)
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
