"""Tests for the OpenAI-compatible server surface (issue ``server-openai``).

CPU-runnable (no ``xpu`` marker, no torch): the HTTP surface is
dependency-light, so it is exercised with FastAPI's ``TestClient``. The only
torch-bound seam — generation — is stubbed, and these tests pin that the
endpoint converts a not-ready engine into a clean 503 rather than a
traceback (the behavior the ``ft serve`` spine depends on).
"""
from __future__ import annotations

import pytest

from freetoken.server.args import DEFAULT_HOST, DEFAULT_PORT, ServerArgs, parse_args
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


def test_parse_args_defaults():
    sa = parse_args(["Qwen/Qwen3-30B-A3B"])
    assert sa.model == "Qwen/Qwen3-30B-A3B"
    assert sa.server_host == DEFAULT_HOST
    assert sa.server_port == DEFAULT_PORT
    assert sa.resolved_model_name == "Qwen3-30B-A3B"
    assert sa.tool_call_parser == "qwen25"  # inferred from the qwen3 marker
    assert sa.reasoning_parser == "qwen3"


def test_parse_args_explicit_overrides():
    sa = parse_args(["some/repo", "--host", "0.0.0.0", "--port", "9000", "--served-model-name", "alias"])
    assert sa.server_host == "0.0.0.0"
    assert sa.server_port == 9000
    assert sa.resolved_model_name == "alias"


def test_parse_args_off_reasoning():
    sa = parse_args(["some/repo", "--reasoning-parser", "off"])
    assert sa.reasoning_parser is None


def test_parse_args_bad_port_rejected():
    with pytest.raises(ValueError):
        parse_args(["m", "--port", "99999"])


def test_health_endpoint_live():
    app = _make_app(["hello"])
    response = _client(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_models_endpoint_reports_served_name():
    app = _make_app(["hello"])
    response = _client(app).get("/v1/models")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data[0]["id"] == "Qwen3-30B-A3B"
    assert data[0]["owned_by"] == "freetoken-intel"


def test_chat_completions_non_streaming():
    app = _make_app(["Hello", " world"])
    response = _client(app).post(
        "/v1/chat/completions",
        json={"model": "Qwen3-30B-A3B", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "Hello world"
    assert body["choices"][0]["finish_reason"] == "stop"


def test_chat_completions_streaming_sse():
    app = _make_app(["Hello", " world"])
    with _client(app).stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "Qwen3-30B-A3B", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
    ) as response:
        assert response.status_code == 200
        text = "".join(response.iter_text())
    assert "data:" in text
    assert "data: [DONE]" in text
    assert "chat.completion.chunk" in text


def test_not_ready_engine_returns_503_not_traceback():
    def engine_holder():
        from freetoken._stub import NotYetImplemented

        raise NotYetImplemented("engine loop is a stub — implement under `engine-loop` (#14)")

    app = create_app(parse_args(["m"]), engine_holder)
    client = _client(app)
    response = client.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "x"}]})
    assert response.status_code == 503
    assert "detail" in response.json()
    # /v1/models stays live even while the engine is a stub.
    assert client.get("/v1/models").status_code == 200
