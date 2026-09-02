"""Shell TUI: run_tui loop (scripted I/O) + ChatClient (real HTTP) (issue `shell-daemon`, #27)."""
from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from freetoken.shell import main as shell_main
from freetoken.shell.tui import ChatClient, ChatClientError, run_tui


def _scripted_input(lines: list[str]) -> callable:
    it = iter(lines)

    def _input(_prompt: str) -> str:
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    return _input


class _StubClient:
    def __init__(self, replies: list[str]) -> None:
        self._replies = iter(replies)
        self.sent: list[list[dict]] = []

    def send(self, messages: list[dict]) -> str:
        self.sent.append([dict(m) for m in messages])
        return next(self._replies)


def test_run_tui_echoes_replies_and_exits_on_slash_exit():
    out = io.StringIO()
    client = _StubClient(["hi there"])
    code = run_tui("http://x", "m", client=client, input_fn=_scripted_input(["hello", "/exit"]), out=out)
    assert code == 0
    assert "hi there" in out.getvalue()
    assert client.sent == [[{"role": "user", "content": "hello"}]]


def test_run_tui_stops_on_eof_without_error():
    out = io.StringIO()
    client = _StubClient([])
    code = run_tui("http://x", "m", client=client, input_fn=_scripted_input([]), out=out)
    assert code == 0


def test_run_tui_skips_blank_lines():
    out = io.StringIO()
    client = _StubClient(["ok"])
    run_tui("http://x", "m", client=client, input_fn=_scripted_input(["", "  ", "hi", "/exit"]), out=out)
    assert len(client.sent) == 1


def test_run_tui_reports_client_error_and_keeps_going():
    class _FailThenSucceed:
        def __init__(self) -> None:
            self.calls = 0

        def send(self, messages):
            self.calls += 1
            if self.calls == 1:
                raise ChatClientError("boom")
            return "recovered"

    out = io.StringIO()
    run_tui("http://x", "m", client=_FailThenSucceed(), input_fn=_scripted_input(["a", "b", "/exit"]), out=out)
    assert "error: boom" in out.getvalue()
    assert "recovered" in out.getvalue()


def test_shell_help_exits_zero():
    assert shell_main(["--help"]) == 0


# ---------------------------------------------------------------------------
# ChatClient against a real (stdlib-only) HTTP endpoint.
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_openai_server():
    """A minimal stdlib HTTP server standing in for `ft serve`'s
    /v1/chat/completions -- proves ChatClient's real request/response wire
    format against a real socket, without needing the full engine stack."""

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # noqa: D401 -- silence test log spam
            pass

        def do_POST(self):
            length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(length))
            if body.get("model") == "error-model":
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"boom")
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            reply = f"echo: {body['messages'][-1]['content']}"
            self.wfile.write(json.dumps({"choices": [{"message": {"content": reply}}]}).encode())

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_chat_client_sends_and_parses_a_real_response(fake_openai_server):
    client = ChatClient(fake_openai_server, "m")
    reply = client.send([{"role": "user", "content": "hello"}])
    assert reply == "echo: hello"


def test_chat_client_raises_on_http_error(fake_openai_server):
    client = ChatClient(fake_openai_server, "error-model")
    with pytest.raises(ChatClientError, match="server returned 500"):
        client.send([{"role": "user", "content": "hi"}])


def test_chat_client_raises_on_connection_error():
    client = ChatClient("http://127.0.0.1:1", "m")  # nothing listens on port 1
    with pytest.raises(ChatClientError, match="could not reach server"):
        client.send([{"role": "user", "content": "hi"}])
