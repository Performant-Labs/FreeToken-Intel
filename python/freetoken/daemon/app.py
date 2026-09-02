"""Daemon HTTP control plane.

Upstream NVIDIA path: python/freetoken/daemon/app.py
Issue: `shell-daemon` (#27, see docs/architecture.md).

A tiny FastAPI app -- ``/status`` (GET), ``/start`` / ``/stop`` (POST) --
in front of one :class:`freetoken.daemon.serve_manager.ServeManager`. This
is the daemon's own control channel, separate from the ``ft serve`` model
server it supervises (which exposes the OpenAI/Anthropic routes on its own
port once running -- see ``server/api_server.py``). Dependency-light like
``server/openai_api.py`` (FastAPI + pydantic, no torch), so building/testing
this app never touches torch either.
"""
from __future__ import annotations

import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .serve_manager import ServeManager


class StartRequest(BaseModel):
    model: str
    host: str = "127.0.0.1"
    port: int = 8080
    extra_args: list[str] = []


def _serve_argv(req: StartRequest) -> list[str]:
    return [
        sys.executable,
        "-m",
        "freetoken.cli",
        "serve",
        req.model,
        "--host",
        req.host,
        "--port",
        str(req.port),
        *req.extra_args,
    ]


def create_app(serve_manager: ServeManager | None = None) -> FastAPI:
    """Build the daemon's control-plane app. ``serve_manager`` defaults to a
    fresh :class:`ServeManager`; tests inject one to observe/stub start/stop."""
    app = FastAPI(title="freetoken-daemon", description="ft daemon control plane")
    app.state.serve_manager = serve_manager if serve_manager is not None else ServeManager()

    @app.get("/status")
    def status() -> dict:
        return app.state.serve_manager.status()

    @app.post("/start")
    def start(req: StartRequest) -> dict:
        try:
            app.state.serve_manager.start(
                _serve_argv(req),
                meta={"model": req.model, "host": req.host, "port": req.port},
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return app.state.serve_manager.status()

    @app.post("/stop")
    def stop() -> dict:
        app.state.serve_manager.stop()
        return app.state.serve_manager.status()

    return app
