"""Anthropic-compatible /v1/messages.

Upstream NVIDIA path: python/freetoken/server/anthropic_api.py

The second half of the serving contract (``docs/stack.md``: OpenAI
``/v1/chat/completions`` **and** Anthropic ``/v1/messages`` on 1919). Claude
Code and the other Anthropic-SDK coding agents point ``ANTHROPIC_BASE_URL``
here, so this endpoint is what lets the box double as a local Claude backend.

It is a thin, dependency-light translation over the same generation seam the
OpenAI routes use (``generation.stream_chat``), rendered into Anthropic's
en,
``message``, ``content_block_*`` wire shapes. Like the OpenAI routes it imports
no torch, so it is exercised on a CPU box; the only torch-bound seam is
generation, and a not-ready engine there yields a clean Anthropic-shaped ``503``
``api_error`` rather than a traceback.
"""
from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from freetoken._stub import NotYetImplemented
from freetoken.server import generation
from freetoken.server.reasoning_parser import get_parser as get_reasoning_parser


def _anthropic_message_id() -> str:
    return "msg_" + uuid.uuid4().hex[:24]


class MessagePart(BaseModel):
    type: str = "text"
    text: str


class AnthropicMessageRequest(BaseModel):
    """Anthropic ``/v1/messages`` request body (subset we serve).

    Anthropic models have a ``system`` prompt that lives *outside* the
    ``messages`` array, so it is a top-level field. ``messages`` entries carry a
    ``role`` (``user`` / ``assistant``) and a ``content`` that may be a string
    or a list of typed parts; only ``text`` parts are rendered to the engine.
    """

    model: str
    messages: list[dict]
    system: str | None = None
    max_tokens: int = Field(default=1024, ge=1)
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None

    model_config = {"populate_by_name": True}


def _to_chat_messages(system: str | None, messages: list[dict]) -> list[dict]:
    """Render the Anthropic request into the chat shape ``stream_chat`` consumes.

    The system prompt is prepended as a ``system`` message; each Anthropic
    message is passed through with its role and content (string or parts list)
    intact. ``generation._prompt_from_messages`` flattens parts, so only ``text``
    parts survive rendering to the prompt.
    """
    rendered: list[dict] = []
    if system:
        rendered.append({"role": "system", "content": system})
    for message in messages:
        rendered.append({"role": message.get("role", "user"), "content": message.get("content", "")})
    return rendered


def register_anthropic_routes(app: FastAPI, engine_holder) -> None:
    """Mount the Anthropic-compatible endpoints onto ``app``.

    ``engine_holder`` is a zero-arg callable returning the loaded ``Engine``
    (or raising ``NotYetImplemented`` if the loader is still a stub). It is
    called per-request, never at import time, so creating the app never touches
    torch — mirroring ``register_openai_routes``.
    """

    def _engine():
        try:
            return engine_holder()
        except NotYetImplemented as exc:
            # The OpenAI routes map a not-ready engine to a FastAPI 503 with a
            # ``detail`` body. Anthropic clients instead parse an
            # ``{"type": "error", "error": {...}}`` envelope, so the route
            # *returns* that shape as a JSONResponse (a Response bypasses
            # FastAPI's ``detail`` wrapper) while the spine's 503 contract
            # (which the serve-spine depends on) still holds.
            return JSONResponse(
                status_code=503,
                content={
                    "type": "error",
                    "error": {"type": "api_error", "message": str(exc)},
                },
            )

    @app.post("/v1/messages/count-tokens")
    def count_tokens(_request: AnthropicMessageRequest) -> dict:
        # Cheap, engine-free proxy for Anthropic's token-estimation endpoint.
        # Counting is over the rendered prompt text (whitespace-delimited), so
        # this stays dependency-light and never blocks on the engine.
        parts: list[str] = []
        if _request.system:
            parts.append(_request.system)
        for message in _request.messages:
            parts.append(str(message.get("content", "")))
        prompt = " ".join(parts)
        return {"input_tokens": len(prompt.split())}

    @app.post("/v1/messages")
    def create_message(request: AnthropicMessageRequest) -> object:
        if request.stream:
            return StreamingResponse(_message_stream(request), media_type="text/event-stream")
        # Non-streaming: collect the stream and emit one Anthropic ``message``.
        return _message_response(request)

    def _message_stream(request: AnthropicMessageRequest) -> Iterator[str]:
        yield from _sse(_message_events(request))

    def _message_events(request: AnthropicMessageRequest) -> Iterator[dict]:
        engine = _engine()
        if isinstance(engine, Response):  # not-ready engine: emit the 503, nothing else
            yield {"type": "error", "error": engine.body}
            return
        server_args = app.state.server_args
        reasoning_parser = get_reasoning_parser(server_args.reasoning_parser)
        model_name = server_args.resolved_model_name
        message_id = _anthropic_message_id()
        created = int(time.time())
        yield {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model_name,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }
        block_index = 0
        block_opened = False
        for reasoning_delta, content_delta in generation.stream_chat(
            engine,
            _to_chat_messages(request.system, request.messages),
            model=model_name,
            max_tokens=request.max_tokens,
            reasoning_parser=reasoning_parser,
        ):
            # The OpenAI routes surface the parser's "thinking" stream as a
            # separate ``reasoning_content`` key; Anthropic has that concept
            # natively. Route it to an ``thinking`` block (first), then the
            # visible ``text`` block, so a reasoning model's output is not
            # dropped or mis-filed.
            if reasoning_delta:
                if not block_opened or block_index != 0:
                    block_opened = True
                    yield {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "thinking", "thinking": ""},
                    }
                    block_index = 0
                yield {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": reasoning_delta}}
            if content_delta:
                if not block_opened or block_index != 1:
                    block_index = 1
                    block_opened = True
                    yield {
                        "type": "content_block_start",
                        "index": 1,
                        "content_block": {"type": "text", "text": ""},
                    }
                yield {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": content_delta}}
        yield {"type": "message_stop"}

    def _message_response(request: AnthropicMessageRequest):
        engine = _engine()
        if isinstance(engine, Response):  # not-ready engine: return the 503 envelope
            return engine
        server_args = app.state.server_args
        reasoning_parser = get_reasoning_parser(server_args.reasoning_parser)
        model_name = server_args.resolved_model_name
        content: list[dict] = []
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        for reasoning_delta, content_delta in generation.stream_chat(
            engine,
            _to_chat_messages(request.system, request.messages),
            model=model_name,
            max_tokens=request.max_tokens,
            reasoning_parser=reasoning_parser,
        ):
            if reasoning_delta:
                reasoning_parts.append(reasoning_delta)
            if content_delta:
                content_parts.append(content_delta)
        if reasoning_parts:
            content.append({"type": "thinking", "thinking": "".join(reasoning_parts)})
        if content_parts:
            content.append({"type": "text", "text": "".join(content_parts)})
        elif not content:
            content.append({"type": "text", "text": ""})
        return {
            "id": _anthropic_message_id(),
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": model_name,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": sum(len(p) for p in content_parts)},
        }

    def _sse(events: Iterator[dict]) -> Iterator[str]:
        for event in events:
            yield f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
        # Anthropic's SSE stream has no terminal "[DONE]" sentinel — the
        # ``message_stop`` event is the end.


__all__ = ["register_anthropic_routes", "AnthropicMessageRequest", "MessagePart"]
