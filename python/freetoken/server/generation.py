"""Request -> engine generation path, streaming.

Upstream NVIDIA path: python/freetoken/server/generation.py

This is the seam between the HTTP surface and the engine. The OpenAI routes
call :func:`stream_chat`; it renders the messages to a prompt, streams decoded
tokens from the engine, and splits reasoning from content via the configured
reasoning parser.

Boundary: the engine loop (``#14``) is not implemented yet, so a call into it
raises :class:`EngineNotReady` (a ``NotYetImplemented``). The routes map that
to a clean 503 — the server stays up and ``/v1/models`` keeps answering —
which is exactly what the ``ft serve`` spine relies on to report
``blocked: engine-loop (#14)`` once the loader and model land.
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Iterator

from freetoken._stub import NotYetImplemented


class EngineNotReady(NotYetImplemented):
    """The HTTP surface is wired but the generation backend is still a stub."""


def _chat_completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]


def _prompt_from_messages(messages: list[dict]) -> str:
    """Flatten chat messages into the single prompt string the engine stub takes.

    The real message frontend / chat-template rendering (tokenizer process)
    replaces this once the tokenizer path exists; today it is a
    deterministic, dependency-free join so the route is testable on a CPU box.
    """
    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if isinstance(content, list):  # multimodal content parts
            content = " ".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _stream_tokens(engine, prompt: str, *, model: str, max_tokens: int | None) -> Iterator[str]:
    """Yield raw decoded token strings from the engine.

    The real implementation (issue #14) submits the prompt via
    ``engine.add_request`` and drains ``engine.step`` outputs. Until the
    engine loop lands, an engine without a ``generate`` method is the
    not-ready signal: it raises ``EngineNotReady`` (a ``NotYetImplemented``),
    which the routes map to a clean 503. A loaded engine exposes ``generate``
    and is streamed token by token.
    """
    generate = getattr(engine, "generate", None)
    if generate is None:
        raise EngineNotReady(
            "engine loop is not implemented — generation is blocked on "
            "engine-loop (#14); see docs/architecture.md"
        )
    for token in generate(prompt, model=model, max_tokens=max_tokens):
        yield token


def stream_chat(
    engine,
    messages: list[dict],
    *,
    model: str,
    max_tokens: int | None = None,
    reasoning_parser=None,
):
    """Yield ``(reasoning_delta, content_delta)`` per decoded token.

    ``reasoning_parser`` is stateful across the whole stream (it buffers
    partial tags), so each token is fed through it and ``flush()`` drains the
    remainder once the stream ends. With no parser, every token is content.
    """
    prompt = _prompt_from_messages(messages)
    if reasoning_parser is None:
        for token in _stream_tokens(engine, prompt, model=model, max_tokens=max_tokens):
            yield "", token
        return
    for token in _stream_tokens(engine, prompt, model=model, max_tokens=max_tokens):
        for reasoning_delta, content_delta in reasoning_parser.parse([token]):
            if reasoning_delta or content_delta:
                yield reasoning_delta, content_delta
    final_reasoning, final_content = reasoning_parser.flush()
    if final_reasoning or final_content:
        yield final_reasoning, final_content


def completion_id() -> str:
    return _chat_completion_id()


def now_timestamp() -> int:
    return int(time.time())


__all__ = ["EngineNotReady", "stream_chat", "completion_id", "now_timestamp"]
