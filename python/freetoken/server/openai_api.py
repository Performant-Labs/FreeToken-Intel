"""OpenAI-compatible /v1/chat/completions, /v1/responses, /v1/models.

Upstream NVIDIA path: python/freetoken/server/openai_api.py
Fill in: GitHub issue `server-openai` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


def register_openai_routes(*args, **kwargs):
    unimplemented("register_openai_routes", "server-openai")

