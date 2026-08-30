"""Prompt-side encoding: render a chat into ids with the model's chat template.

Upstream NVIDIA path: python/freetoken/tokenizer/tokenize.py

The Intel port drops the ZMQ tokenizer *process* and the DeepSeek-V4/GGUF custom
encoder: on this B70 the hero model (Qwen3.5/3.6) ships a plain HuggingFace Jinja
chat template, so ``encode`` runs ``apply_chat_template`` directly on the
transformers ``AutoTokenizer`` in-process. This is the same template the engine's
worker would run, so a rendered prompt tokenizes identically to a generated one.
"""

from __future__ import annotations

from typing import Any

# ``apply_chat_template(..., return_dict=True)`` returns a ``BatchEncoding``
# (a ``UserDict`` in transformers 5.x) whose ``input_ids`` is a plain ``list[int]``. We take those
# ids directly rather than rendering a string and re-splitting: the string
# round-trip would lose the template's exact tokenization boundaries.
_CHAT_KWARGS = {"add_generation_prompt": True, "return_dict": True}


class TokenizeManager:
    """Own the chat-template encode path for one model.

    Mirrors the upstream ``TokenizeManager`` minus the worker-process / ZMQ
    machinery: ``encode`` and ``count`` are synchronous in-process calls.
    """

    def __init__(self, tokenizer: Any, eos_token_id: int | None = None) -> None:
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self._template = getattr(tokenizer, "chat_template", None)

    def render(self, messages: list[dict[str, Any]]) -> str:
        """Human-readable rendering of the templated prompt (diagnostic only)."""
        return self.tokenizer.decode(self.encode(messages), skip_special_tokens=False)

    def encode(self, messages: list[dict[str, Any]]) -> list[int]:
        """Render the chat through the model's chat template and return the ids."""
        if not self._template:
            raise RuntimeError(
                "the model ships no chat template — cannot render a chat prompt"
            )
        return list(self.tokenizer.apply_chat_template(messages, **_CHAT_KWARGS)["input_ids"])

    def count(self, messages: list[dict[str, Any]]) -> int:
        """Number of prompt tokens a chat renders to, without keeping the ids."""
        return len(self.encode(messages))

    @property
    def supports_tools(self) -> bool:
        """Whether the chat template understands a ``tools`` kwarg (function calling)."""
        return "tools" in (self._template or "")


__all__ = ["TokenizeManager"]
