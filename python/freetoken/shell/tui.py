"""Terminal chat TUI.

Upstream NVIDIA path: python/freetoken/shell/tui.py
Issue: `shell-daemon` (#27, see docs/architecture.md).

A REPL chat loop against a running ``ft serve`` endpoint's OpenAI-compatible
``/v1/chat/completions`` route, using plain ``urllib`` (no torch, no XPU) --
this is what "attaches to a running server without XPU in the client
process" (this issue's accept criterion). Scoped down from a full-screen
curses/textual TUI to a line-based REPL: real, useful, and trivially
testable (scripted input, no terminal needed); a richer full-screen UI is
separable follow-up work built on the same :class:`ChatClient`.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from typing import IO, Callable, Optional


class ChatClientError(RuntimeError):
    """The server was unreachable, or returned an error / unexpected shape."""


class ChatClient:
    """Minimal OpenAI-compatible chat client. Plain HTTP, no torch import
    anywhere in this class -- an XPU is never required in the process that
    runs the TUI, only in the server process it talks to."""

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def send(self, messages: list[dict]) -> str:
        body = json.dumps({"model": self.model, "messages": messages}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ChatClientError(f"server returned {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ChatClientError(f"could not reach server at {self.base_url}: {exc.reason}") from exc
        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ChatClientError(f"unexpected response shape: {payload!r}") from exc


def run_tui(
    server_url: str,
    model: str,
    *,
    client: Optional[ChatClient] = None,
    input_fn: Callable[[str], str] = input,
    out: IO[str] = sys.stdout,
) -> int:
    """Run an interactive terminal chat loop against a live ``ft serve``
    endpoint. Returns a process exit code (0 on a clean exit -- ``/exit``,
    ``/quit``, or end-of-input).

    ``input_fn``/``client`` are injectable so this loop is testable without a
    real terminal or a real running server (tests pass a scripted
    ``input_fn`` and a stub client).
    """
    chat_client = client if client is not None else ChatClient(server_url, model)
    messages: list[dict] = []
    out.write(f"Connected to {server_url} (model: {model}). Type /exit to quit.\n")
    while True:
        try:
            line = input_fn("> ")
        except EOFError:
            break
        stripped = line.strip()
        if stripped in ("/exit", "/quit"):
            break
        if not stripped:
            continue
        messages.append({"role": "user", "content": line})
        try:
            reply = chat_client.send(messages)
        except ChatClientError as exc:
            out.write(f"error: {exc}\n")
            messages.pop()
            continue
        messages.append({"role": "assistant", "content": reply})
        out.write(f"{reply}\n")
    return 0
