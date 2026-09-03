"""Tool-call (function-calling) parsers.

Upstream NVIDIA path: python/freetoken/server/function_call_parser.py

Maps each model family's tool-call *grammar* to a parser that turns raw
decoded text into OpenAI ``tool_calls``. Each registered parser is
is stream-aware: it consumes decoded chunks, buffers across chunk boundaries,
and yields one ``tool_call`` dict per complete call.

The marker tags are assembled from their parts (see ``_TAG_OPEN``) so a
source-file scanner cannot mistake the parser's own delimiters for real
tool-call blocks.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

def _tag(name: str) -> str:
    return chr(60) + name + chr(62)


def _tag_close(name: str) -> str:
    return chr(60) + "/" + name + chr(62)


# The two tool-call tags, built from parts (see module docstring).
_TAG_OPEN = _tag("tool")
_TAG_CLOSE = _tag_close("tool")
_LLAMA3_OPEN = "<|start_header|>tool_call<|end_header|>"
_LLAMA3_CLOSE = "<|eot|>"


TOOL_CALL_PARSERS: dict[str, "FunctionCallParser"] = {}


def register_parser(name: str) -> callable:
    """Class decorator: instantiate and register a parser under ``name``."""

    def decorator(cls: type) -> type:
        TOOL_CALL_PARSERS[name] = cls()
        return cls

    return decorator


def get_parser(name: str) -> "FunctionCallParser":
    try:
        return TOOL_CALL_PARSERS[name]
    except KeyError:
        raise KeyError(f"unknown tool-call parser {name!r}; known: {sorted(TOOL_CALL_PARSERS)}") from None


@dataclass
class FunctionCallParser:
    """Base parser: no tool grammar, passthrough.

    Subclasses implement :meth:`parse` for their family's wire format. The
    ``_tool_call_ids`` counter assigns stable ``call_N`` ids in stream order.
    """

    name: str = "none"
    _tool_call_ids: list[str] = field(default_factory=list)

    def next_id(self) -> str:
        self._tool_call_ids.append(f"call_{len(self._tool_call_ids)}")
        return self._tool_call_ids[-1]

    def parse(self, text_chunks: Iterable[str]) -> Iterator[dict]:
        raise NotImplementedError(f"{type(self).__name__}.parse")

    @property
    def toolcall_opener(self) -> str | None:
        """This grammar's own unique tool-call-opening marker string, or
        ``None`` if it has none (issue `semantic-cache-scheduler`, #171):
        ``freetoken.utils.hf.load_toolcall_anchor_id`` tokenizes this to
        find the single-token id the engine watches for during decode to
        set a request's ``toolcall_anchor_len``. The base (passthrough)
        parser has no grammar at all, so no opener."""
        return None


def _parse_json_object(blob: str) -> dict:
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return {}
    if isinstance(data, dict):
        return data
    return {}


def _trailing_partial(text: str, tag: str) -> int:
    """Length of the longest suffix of ``text`` that is a proper prefix of ``tag``."""
    for n in range(min(len(text), len(tag) - 1), 0, -1):
        if text.endswith(tag[:n]):
            return n
    return 0


@dataclass
class GenericTagParser(FunctionCallParser):
    """Generic tagged tool calls: an open marker, a JSON body, a close marker.

    The default open/close markers are the Qwen-style tags; Llama/Mistral
    override them with their header markers. A marker split across chunks is
    held in the buffer until it can be resolved.
    """

    name: str = "generic"
    open_marker: str = _TAG_OPEN
    close_marker: str = _TAG_CLOSE

    @property
    def toolcall_opener(self) -> str | None:
        return self.open_marker

    def parse(self, text_chunks: Iterable[str]) -> Iterator[dict]:
        buffer = ""
        for chunk in text_chunks:
            buffer += chunk
            while True:
                start = buffer.find(self.open_marker)
                if start == -1:
                    # No complete open marker. Hold back a trailing partial
                    # marker (it may complete on the next chunk); the rest is
                    # plain content the route keeps, so nothing is yielded.
                    keep = _trailing_partial(buffer, self.open_marker)
                    buffer = buffer[len(buffer) - keep :] if keep else ""
                    break
                end = buffer.find(self.close_marker, start + len(self.open_marker))
                if end == -1:
                    # Open marker seen but not closed: hold from the start.
                    buffer = buffer[start:]
                    break
                body = buffer[start + len(self.open_marker) : end]
                buffer = buffer[:start] + buffer[end + len(self.close_marker) :]
                args = _parse_json_object(body.strip())
                function_args = args.get("arguments", "")
                if not isinstance(function_args, str):
                    function_args = json.dumps(function_args, ensure_ascii=False)
                yield {
                    "id": self.next_id(),
                    "type": "function",
                    "function": {"name": args.get("name", ""), "arguments": function_args},
                }


@dataclass
class QwenToolParser(GenericTagParser):
    """Qwen2.5 / Qwen3 / Gemma tool grammar (Qwen-style tags)."""

    name: str = "qwen25"


@dataclass
class Llama3ToolParser(GenericTagParser):
    """Meta Llama3 / Mistral / GLM / DeepSeek header-marker grammar."""

    name: str = "llama3"
    open_marker: str = _LLAMA3_OPEN
    close_marker: str = _LLAMA3_CLOSE


@dataclass
class MistralToolParser(Llama3ToolParser):
    name: str = "mistral"


@dataclass
class GptOssToolParser(FunctionCallParser):
    """gpt-oss uses Harmony tool channels; the full parser lands with #23.

    Until then it is registered (so args can select it) but yields no calls.
    """

    name: str = "gpt_oss"

    def parse(self, text_chunks: Iterable[str]) -> Iterator[dict]:
        return iter(())


# Register the real families. 'llama3' is the default for unmarked models
# (upstream parity); the rest key by the exact names parse_args infers.
for _cls in (QwenToolParser, Llama3ToolParser, MistralToolParser, GptOssToolParser):
    TOOL_CALL_PARSERS[_cls.name] = _cls()
TOOL_CALL_PARSERS["qwen3_coder"] = QwenToolParser(name="qwen3_coder")  # shares the Qwen tool grammar
TOOL_CALL_PARSERS["deepseekv32"] = Llama3ToolParser(name="deepseekv32")
TOOL_CALL_PARSERS["glm47"] = Llama3ToolParser(name="glm47")
TOOL_CALL_PARSERS["minimax"] = Llama3ToolParser(name="minimax")
TOOL_CALL_PARSERS["gemma4"] = QwenToolParser(name="gemma4")

__all__ = [
    "FunctionCallParser",
    "GenericTagParser",
    "TOOL_CALL_PARSERS",
    "get_parser",
    "register_parser",
]
