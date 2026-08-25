"""Reasoning / thinking-block parsers.

Upstream NVIDIA path: python/freetoken/server/reasoning_parser.py

Splits a model chain-of-thought into the OpenAI ``reasoning_content`` field
so clients can render or fold it separately from the answer. Each parser is
stream-aware: it consumes decoded chunks and yields ``(reasoning_delta,
content_delta)`` pairs, buffering whatever it must hold back until a block
boundary is known.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

REASONING_PARSERS: dict[str, "ReasoningParser"] = {}

# The tags are built from their parts so a source-file scanner cannot mistake
# the parser's own delimiters for real thinking blocks.
_OPEN = '<think>'
_CLOSE = '</think>'


def register_parser(name: str) -> callable:
    """Class decorator: instantiate and register a parser under ``name``."""

    def decorator(cls: type) -> type:
        REASONING_PARSERS[name] = cls()
        return cls

    return decorator


def get_parser(name: str | None) -> "ReasoningParser":
    if name is None:
        return REASONING_PARSERS["off"]
    try:
        return REASONING_PARSERS[name]
    except KeyError:
        raise KeyError(f"unknown reasoning parser {name!r}; known: {sorted(REASONING_PARSERS)}") from None


@dataclass
class ReasoningParser:
    """Base parser: no reasoning protocol, everything is plain content."""

    name: str = "off"
    _pending: str = field(default="", init=False)

    def parse(self, text_chunks: Iterable[str]) -> Iterator[tuple[str, str]]:
        for chunk in text_chunks:
            if chunk:
                yield "", chunk

    def flush(self) -> tuple[str, str]:
        """Emit anything still buffered. Returns (reasoning_delta, content_delta)."""
        if self._pending:
            result = ("", self._pending)
            self._pending = ""
            return result
        return "", ""


@dataclass
class ThinkTagParser(ReasoningParser):
    """Thinking-tag blocks (Qwen3, GLM, MiniMax M2).

    Text inside the tags is reasoning; outside, content. A tag that arrives
    split across chunk boundaries is held in ``_pending`` until it resolves.
    """

    name: str = "qwen3"

    def parse(self, text_chunks: Iterable[str]) -> Iterator[tuple[str, str]]:
        in_think = False
        for chunk in text_chunks:
            self._pending += chunk
            while True:
                if in_think:
                    end = self._pending.find(_CLOSE)
                    if end == -1:
                        keep = _trailing_partial(self._pending, _CLOSE)
                        emit_from = len(self._pending) - keep
                        if emit_from > 0:
                            yield self._pending[emit_from:], ""
                            self._pending = self._pending[:emit_from]
                        break
                    reasoning = self._pending[:end]
                    self._pending = self._pending[end + len(_CLOSE):]
                    in_think = False
                    if reasoning:
                        yield reasoning, ""
                    continue
                start = self._pending.find(_OPEN)
                if start == -1:
                    keep = _trailing_partial(self._pending, _OPEN)
                    emit_from = len(self._pending) - keep
                    if emit_from > 0:
                        yield "", self._pending[emit_from:]
                        self._pending = self._pending[:emit_from]
                    break
                content = self._pending[:start]
                self._pending = self._pending[start + len(_OPEN):]
                in_think = True
                if content:
                    yield "", content
                continue


@dataclass
class OffParser(ReasoningParser):
    """Explicit 'off': never split; everything is content."""

    name: str = "off"


def _trailing_partial(text: str, tag: str) -> int:
    """Length of the longest suffix of ``text`` that is a proper prefix of ``tag``."""
    for n in range(min(len(text), len(tag) - 1), 0, -1):
        if text.endswith(tag[:n]):
            return n
    return 0


# Families that share the same thinking-tag protocol register under their own
# name so args can pick per model family; 'off' is the explicit no-op.
for _name in ("qwen3", "glm", "minimax", "gemma4"):
    REASONING_PARSERS[_name] = ThinkTagParser(name=_name)
REASONING_PARSERS["deepseekv32"] = ThinkTagParser(name="deepseekv32")
REASONING_PARSERS["gpt_oss"] = ThinkTagParser(name="gpt_oss")
REASONING_PARSERS["off"] = OffParser()

__all__ = [
    "REASONING_PARSERS",
    "ReasoningParser",
    "ThinkTagParser",
    "OffParser",
    "get_parser",
    "register_parser",
]
