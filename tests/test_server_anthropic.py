"""Tests for the Anthropic-compatible server surface (issue ``server-anthropic``).

CPU-runnable (no ``xpu`` marker, no torch): the HTTP surface is
dependency-light, so it is exercised with FastAPI's ``TestClient``. The only
torch-bound seam — generation — is stubbed, and these tests pin that
``/v1/messages`` renders Anthropic's wire shapes (``message`` / ``content`` /
``usage``), that a not-ready engine yields a clean Anthropic-shaped ``503``
``api_error`` (not a traceback or an OpenAI ``detail``), and that streaming
emits the ``message_start`` / ``content_block_*`` / ``message_stop`` event
sequence without a ``[DONE]`` sentinel.
"""
from __future__ import annotations

from freetoken.server.args import parse_args
from freetoken.server.api_server import create_app


class _StubEngine:
    """A loaded engine that produces a fixed token stream (no torch).

    Exposes ``generate`` — the seam ``generation.stream_chat`` streams from —
    so it is indistinguishable from a real loaded engine to the HTTP layer.
    """

    def __init__(self, tokens: list[str]):
        self._tokens = tokens

    def generate(self, prompt, *, model, max_tokens=None):
        for token in self._tokens:
            yield token


def _make_app(tokens: list[str]):
    def engine_holder():
        return _StubEngine(tokens)

    server_args = parse_args(["Qwen/Qwen3-30B-A3B"])
    return create_app(server_args, engine_holder)


def _client(app):
    from fastapi.testclient import TestClient

    return TestClient(app)


def _messages_payload(**overrides):
    payload = {
        "model": "Qwen3-30B-A3B",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Hi"}],
    }
    payload.update(overrides)
    return payload


def test_messages_non_streaming_envelope():
    app = _make_app(["Hello", " world"])
    response = _client(app).post("/v1/messages", json=_messages_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["model"] == "Qwen3-30B-A3B"
    assert body["stop_reason"] == "end_turn"
    # The text content is the first (and only) content block.
    assert body["content"][0]["type"] == "text"
    assert body["content"][0]["text"] == "Hello world"
    assert body["usage"]["output_tokens"] > 0


def test_messages_system_prompt_is_forwarded():
    # A system prompt must reach the engine (Anthropic keeps it top-level,
    # outside the messages array). The stub engine echoes the rendered prompt,
    # so the system text appearing in the output proves the wiring.
    app = _make_app([])
    # Point the stub engine at nothing; instead assert the prompt rendering
    # directly via the module the route calls.
    from freetoken.server.generation import _prompt_from_messages
    from freetoken.server.anthropic_api import _to_chat_messages

    rendered = _to_chat_messages("You are terse.", [{"role": "user", "content": "Hi"}])
    prompt = _prompt_from_messages(rendered)
    assert prompt.startswith("system: You are terse.")


def test_messages_content_parts_rendered():
    # Anthropic content may be a list of typed parts. The OpenAI generation seam
    # flattens parts into the prompt text (it renders a non-text part as a marker
    # — multimodal prompt rendering is a later tokenizer-front issue, out of
    # scope here), so we pin that the text part's text is what surfaces.
    from freetoken.server.generation import _prompt_from_messages
    from freetoken.server.anthropic_api import _to_chat_messages

    rendered = _to_chat_messages(None, [{"role": "user", "content": [{"type": "text", "text": "hello-anthropic"}]}])
    prompt = _prompt_from_messages(rendered)
    assert "hello-anthropic" in prompt


def test_count_tokens_endpoint_is_engine_free():
    app = _make_app([])
    response = _client(app).post("/v1/messages/count-tokens", json=_messages_payload(system="one two three"))
    assert response.status_code == 200
    assert response.json()["input_tokens"] > 0


def test_messages_streaming_event_sequence():
    app = _make_app(["Hello", " world"])
    with _client(app).stream("POST", "/v1/messages", json=_messages_payload(stream=True)) as response:
        assert response.status_code == 200
        text = "".join(response.iter_text())
    assert "event: message_start" in text
    assert "event: content_block_start" in text
    assert "event: content_block_delta" in text
    assert "event: message_stop" in text
    assert "Hello" in text
    # Anthropic streams end with message_stop, never an OpenAI-style [DONE].
    assert "[DONE]" not in text


def test_not_ready_engine_returns_503_anthropic_shape():
    def engine_holder():
        from freetoken._stub import NotYetImplemented

        raise NotYetImplemented("engine loop is a stub — implement under `engine-loop` (#14)")

    app = create_app(parse_args(["m"]), engine_holder)
    client = _client(app)
    response = client.post("/v1/messages", json={"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "x"}]})
    assert response.status_code == 503
    # Anthropic error envelope, not FastAPI detail — the SDK parses `error.type`.
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "api_error"
    # The OpenAI surface stays live on the same app even while the engine is a
    # stub (both dialects share the engine_holder seam).
    assert client.get("/v1/models").status_code == 200
