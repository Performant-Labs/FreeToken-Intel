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


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int | None = Field(default=None, alias="max_completion_tokens")
    temperature: float | None = None
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
    def responses() -> dict:
        _engine()  # 503 if the engine/loader is still a stub
        return {
            "id": "resp-" + uuid.uuid4().hex[:24],
            "object": "response",
            "status": "completed",
            "output": [],
        }

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
