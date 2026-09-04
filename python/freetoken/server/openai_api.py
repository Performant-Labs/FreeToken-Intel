"""OpenAI-compatible routes: /v1/chat/completions, /v1/responses, /v1/models.

Upstream NVIDIA path: python/freetoken/server/openai_api.py

These routes are the real HTTP surface. They are dependency-light (pydantic
models + FastAPI, no torch) so they can be exercised on a CPU box; the only
torch-bound seam is generation, and a not-yet-ready engine there yields a
clean 503 with a ``Retry-After: 0`` body pointing at the owning issue rather
than a traceback.
"""
from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from freetoken._stub import NotYetImplemented
from freetoken.server import generation
from freetoken.server.reasoning_parser import get_parser as get_reasoning_parser
from freetoken.server.function_call_parser import get_parser as get_tool_parser


class ChatMessage(BaseModel):
    role: str
    content: str | list | None = None


class ResponsesRequest(BaseModel):
    """The subset of the real OpenAI Responses API request this port
    honors (issue #200's audit found `/v1/responses` was a hardcoded
    empty-output stub -- looked wired up, never called generation at
    all). Scope: plain text in, plain text out, matching this port's own
    existing `/v1/chat/completions` capability -- tool-call-argument
    streaming and reasoning-item events are real upstream features
    (`responses_api.py`, ~750 lines using the ``openai`` SDK's typed
    response models) deliberately deferred as follow-up, not attempted
    here. ``input`` accepts either a bare string (a single user turn,
    the common case) or the real API's list-of-message-item shape."""

    model: str
    input: str | list[dict]
    instructions: str | None = None
    stream: bool = False
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    tools: list[dict] | None = None
    reasoning_effort: str | None = None

    model_config = {"extra": "allow"}


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int | None = Field(default=None, alias="max_completion_tokens")
    temperature: float | None = None
    top_p: float | None = None
    tools: list[dict] | None = None
    # Reasoning-effort dial (issue #97). Free of a static per-family table, the
    # value is quantized onto the checkpoint's probed effort vocabulary at encode
    # time, so any value is accepted here and mapped (or dropped) downstream.
    reasoning_effort: str | None = None

    model_config = {"populate_by_name": True}


def register_openai_routes(app: FastAPI, engine_holder) -> None:
    """Mount the OpenAI-compatible endpoints onto ``app``.

    ``engine_holder`` is a zero-arg callable returning the loaded
    ``Engine`` (or raising ``NotYetImplemented`` if the loader is still a
    stub). It is called per-request, never at import time, so creating the
    app never touches torch.
    """

    def _engine():
        try:
            return engine_holder()
        except NotYetImplemented as exc:
            raise HTTPException(
                status_code=503,
                detail=f"model not ready — {exc}",
            ) from exc

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models")
    def list_models() -> dict:
        name = _served_name(app)
        return {
            "object": "list",
            "data": [
                {
                    "id": name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "freetoken-intel",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> object:
        if request.stream:
            return StreamingResponse(_chat_stream(request), media_type="text/event-stream")
        # Non-streaming: collect the stream and emit one chat.completion.
        chunks = list(_chat_completions(request))
        return _merge_to_completion(chunks)

    @app.post("/v1/responses")
    def responses(request: ResponsesRequest) -> object:
        if request.stream:
            return StreamingResponse(_responses_stream(request), media_type="text/event-stream")
        text = "".join(content for _, content in _responses_generate(request))
        return _responses_object(text, status="completed")

    def _chat_stream(request: ChatCompletionRequest):
        yield from _sse(_chat_completions(request))

    def _chat_completions(request: ChatCompletionRequest) -> Iterator[dict]:
        engine = _engine()
        server_args = app.state.server_args
        reasoning_parser = get_reasoning_parser(server_args.reasoning_parser)
        model_name = server_args.resolved_model_name
        # The client's reasoning controls ride as chat-template kwargs. The
        # encode step quantizes ``reasoning_effort`` onto the checkpoint's probed
        # effort profile (issue #97); ``tools`` is forwarded separately.
        chat_template_kwargs: dict = {}
        if request.reasoning_effort is not None:
            chat_template_kwargs["reasoning_effort"] = request.reasoning_effort
        for reasoning_delta, content_delta in generation.stream_chat(
            engine,
            [m.model_dump() for m in request.messages],
            model=model_name,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            reasoning_parser=reasoning_parser,
            tools=request.tools,
            chat_template_kwargs=chat_template_kwargs,
        ):
            delta: dict = {}
            if content_delta:
                delta["content"] = content_delta
            if reasoning_delta:
                delta["reasoning_content"] = reasoning_delta
            yield {
                "id": generation.completion_id(),
                "object": "chat.completion.chunk",
                "created": generation.now_timestamp(),
                "model": model_name,
                "choices": [{"index": 0, "delta": delta or {"role": "assistant"}, "finish_reason": None}],
            }
        yield {
            "id": generation.completion_id(),
            "object": "chat.completion.chunk",
            "created": generation.now_timestamp(),
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }

    def _responses_messages(request: ResponsesRequest) -> list[dict]:
        """Normalize the real Responses API's ``input`` (a bare string, or
        a list of message-item dicts whose ``content`` may itself be a
        string or a list of ``{"type": "input_text", "text": ...}``-shaped
        parts) into this port's own plain chat-message shape
        (``generation.stream_chat`` already consumes this -- the same
        conversion `/v1/chat/completions` relies on)."""
        messages: list[dict] = []
        if request.instructions:
            messages.append({"role": "system", "content": request.instructions})
        if isinstance(request.input, str):
            messages.append({"role": "user", "content": request.input})
            return messages
        for item in request.input:
            role = item.get("role", "user")
            content = item.get("content", "")
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict) and "text" in part
                )
            messages.append({"role": role, "content": content})
        return messages

    def _responses_generate(request: ResponsesRequest) -> Iterator[tuple[str, str]]:
        """Yield ``(reasoning_delta, content_delta)`` per decoded token --
        the same primitive `/v1/chat/completions` streams from, reused
        as-is (see this endpoint's own scope note on ``ResponsesRequest``:
        plain text only, no tool-call/reasoning-item streaming yet)."""
        engine = _engine()
        server_args = app.state.server_args
        chat_template_kwargs: dict = {}
        if request.reasoning_effort is not None:
            chat_template_kwargs["reasoning_effort"] = request.reasoning_effort
        yield from generation.stream_chat(
            engine,
            _responses_messages(request),
            model=server_args.resolved_model_name,
            max_tokens=request.max_output_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            tools=request.tools,
            chat_template_kwargs=chat_template_kwargs,
        )

    def _responses_object(text: str, *, status: str, response_id: str | None = None) -> dict:
        """The real Responses API's top-level object shape (a plain dict,
        matching this file's own established style -- see
        `_merge_to_completion` -- rather than depending on the ``openai``
        SDK's typed models, which aren't a declared dependency here)."""
        return {
            "id": response_id or ("resp_" + uuid.uuid4().hex),
            "object": "response",
            "created_at": generation.now_timestamp(),
            "status": status,
            "model": app.state.server_args.resolved_model_name,
            "output": [
                {
                    "type": "message",
                    "id": "msg_" + uuid.uuid4().hex,
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "output_text": text,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        }

    def _responses_stream(request: ResponsesRequest) -> Iterator[str]:
        """Real Responses API SSE events -- ``event: <type>\\ndata:
        <json>\\n\\n`` per event (a different wire shape from
        `/v1/chat/completions`' own plain ``data:``-only frames). Emits
        the subset codex needs for plain-text streaming: ``created`` ->
        one ``output_text.delta`` per token -> ``completed``. Tool-call
        argument deltas and reasoning-item events are the real upstream
        feature this scope note already flags as deferred."""
        response_id = "resp_" + uuid.uuid4().hex
        seq = 0

        def frame(event_type: str, data: dict) -> str:
            nonlocal seq
            data = {"type": event_type, "sequence_number": seq, **data}
            seq += 1
            return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        in_progress = _responses_object("", status="in_progress", response_id=response_id)
        yield frame("response.created", {"response": in_progress})
        yield frame("response.in_progress", {"response": in_progress})

        parts: list[str] = []
        item_id = "msg_" + uuid.uuid4().hex
        yield frame(
            "response.output_item.added",
            {
                "output_index": 0,
                "item": {"type": "message", "id": item_id, "status": "in_progress", "role": "assistant", "content": []},
            },
        )
        yield frame(
            "response.content_part.added",
            {
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
        )
        for _reasoning_delta, content_delta in _responses_generate(request):
            if not content_delta:
                continue
            parts.append(content_delta)
            yield frame(
                "response.output_text.delta",
                {"item_id": item_id, "output_index": 0, "content_index": 0, "delta": content_delta},
            )
        text = "".join(parts)
        yield frame(
            "response.output_text.done",
            {"item_id": item_id, "output_index": 0, "content_index": 0, "text": text},
        )
        yield frame(
            "response.content_part.done",
            {
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": text, "annotations": []},
            },
        )
        final = _responses_object(text, status="completed", response_id=response_id)
        yield frame(
            "response.output_item.done",
            {"output_index": 0, "item": final["output"][0]},
        )
        yield frame("response.completed", {"response": final})

    def _served_name(app: FastAPI) -> str:
        return app.state.server_args.resolved_model_name

    def _sse(chunks: Iterator[dict]) -> Iterator[str]:
        for chunk in chunks:
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    def _merge_to_completion(chunks) -> dict:
        """Collapse a stream of deltas into one OpenAI chat.completion object."""
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish_reason = "stop"
        meta = chunks[0]
        for chunk in chunks:
            choice = chunk["choices"][0]
            delta = choice.get("delta", {})
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            if delta.get("content"):
                content_parts.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning_parts.append(delta["reasoning_content"])
        message: dict = {"role": "assistant", "content": "".join(content_parts)}
        if reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)
        return {
            "id": meta["id"],
            "object": "chat.completion",
            "created": meta["created"],
            "model": meta["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }


__all__ = ["register_openai_routes", "ChatCompletionRequest", "ChatMessage"]
