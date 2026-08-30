"""Prompt-side encoding: render a chat into ids with the model's chat template.

Upstream NVIDIA path: python/freetoken/tokenizer/tokenize.py

The Intel port drops the ZMQ tokenizer *process* and the DeepSeek-V4/GGUF custom
encoder: on this B70 the hero model (Qwen3.5/3.6) ships a plain HuggingFace Jinja
chat template, so ``encode`` runs ``apply_chat_template`` directly on the
transformers ``AutoTokenizer`` in-process. This is the same template the engine's
worker would run, so a rendered prompt tokenizes identically to a generated one.

On top of that, ``encode`` accepts the client's reasoning controls (issue #97):
``reasoning_effort`` is quantized onto the checkpoint's *probed* effort
vocabulary (a client's "max" maps to the nearest gear this template actually
grades; an unsupported value drops to the template default instead of
failing), and the thinking toggle (``enable_thinking`` / ``thinking_mode`` /
``tools``) is resolved through the same probed thinking profile. Both profiles
are learned from the template itself on first use (never a static per-family
table) and cached, so the per-request path only ever does a cheap dict lookup.
The probe renders to a prompt *string*, so this module stays torch-free.
"""

from __future__ import annotations

from typing import Any

import threading

from freetoken.tokenizer.effort import (
    EffortProfile,
    ThinkingProfile,
    probe_effort_profile,
    probe_thinking_profile,
    quantize_effort,
)

# ``apply_chat_template(..., return_dict=True)`` returns a ``BatchEncoding``
# (a ``UserDict`` in transformers 5.x) whose ``input_ids`` is a plain ``list[int]``. We take those
# ids directly rather than rendering a string and re-splitting: the string
# round-trip would lose the template's exact tokenization boundaries.
_CHAT_KWARGS = {"add_generation_prompt": True, "return_dict": True}

#: The minimal conversation the profile probes render (the probe only needs the
#: template to accept the shape; the prompt text is never served).
_PROBE_MESSAGES = [{"role": "user", "content": "ping"}]


def resolve_thinking_mode(chat_template_kwargs: dict[str, Any] | None, tools: Any | None) -> str:
    """The single source of truth for whether a request is in thinking mode.

    The encode side (:meth:`TokenizeManager.encode`) uses it to pick the prompt
    the model sees, and the frontend parse side (``server/openai_api.py``)
    imports it to decide whether the model's output begins inside a reasoning
    block. Keeping one implementation prevents the two sides from disagreeing.
    Thinking is on when tools are offered (a dsv4-style encoder only emits
    well-formed tool calls in thinking mode) or when the caller requests it via
    a thinking kwarg.
    """
    ctk = chat_template_kwargs or {}
    mode = str(ctk.get("thinking_mode") or "chat")
    if tools or ctk.get("enable_thinking") or ctk.get("thinking"):
        mode = "thinking"
    if mode not in ("chat", "thinking"):
        mode = "chat"
    return mode


class TokenizeManager:
    """Own the chat-template encode path for one model.

    Mirrors the upstream ``TokenizeManager`` minus the worker-process / ZMQ
    machinery: ``encode`` and ``count`` are synchronous in-process calls. The
    effort / thinking profiles are probed from the template on first use (never
    a static table) and cached, so the per-request path only does a cheap dict
    lookup.
    """

    def __init__(self, tokenizer: Any, eos_token_id: int | None = None) -> None:
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self._template = getattr(tokenizer, "chat_template", None)
        self._effort_profile: EffortProfile | None = None
        self._thinking_profile: ThinkingProfile | None = None
        self._profile_lock = threading.Lock()
        self._logged_effort_maps: set[tuple[Any, str | None]] = set()

    def render(self, messages: list[dict[str, Any]]) -> str:
        """Human-readable rendering of the templated prompt (diagnostic only)."""
        return self.tokenizer.decode(self.encode(messages), skip_special_tokens=False)

    def encode(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> list[int]:
        """Render the chat through the model's chat template and return the ids.

        ``tools`` and ``chat_template_kwargs`` carry the client's reasoning
        controls (issue #97): ``reasoning_effort`` is quantized onto this
        checkpoint's probed effort vocabulary, and the thinking toggle is
        resolved through the probed thinking profile. With no kwargs the render
        is byte-identical to the pre-#97 path.
        """
        kwargs = dict(chat_template_kwargs or {})
        if "reasoning_effort" in kwargs:
            kwargs = self._sanitize_effort(kwargs)
        if tools is not None:
            kwargs = {**kwargs, "tools": tools}
        if not self._template:
            raise RuntimeError(
                "the model ships no chat template — cannot render a chat prompt"
            )
        return list(
            self.tokenizer.apply_chat_template(
                messages,
                **{**kwargs, **_CHAT_KWARGS},
            )["input_ids"]
        )

    def count(self, messages: list[dict[str, Any]], **kwargs: Any) -> int:
        """Number of prompt tokens a chat renders to, without keeping the ids."""
        return len(self.encode(messages, **kwargs))

    @property
    def supports_tools(self) -> bool:
        """Whether the chat template understands a ``tools`` kwarg (function calling)."""
        return "tools" in (self._template or "")

    # -- Probed dialect profiles (issue #97) ---------------------------------

    def effort_profile(self) -> EffortProfile:
        """The checkpoint's effort vocabulary, probed on first use and cached
        for the process lifetime."""
        with self._profile_lock:
            if self._effort_profile is None:
                self._effort_profile = probe_effort_profile(self._probe_render)
            return self._effort_profile

    def thinking_profile(self) -> ThinkingProfile:
        """The checkpoint's thinking controls (toggle behavior + effort
        vocabulary), probed on first use and cached for the process lifetime."""
        efforts = self.effort_profile()
        with self._profile_lock:
            if self._thinking_profile is None:
                self._thinking_profile = probe_thinking_profile(self._probe_render, efforts)
            return self._thinking_profile

    def _probe_render(self, kwargs: dict[str, Any], tools: list[dict[str, Any]] | None) -> str:
        """Render the probe conversation to a *string* (not a BatchEncoding),
        since ``_renderings_differ`` compares the probe round-trips by string.
        Jinja ignores undeclared template variables, so the raw probe kwargs
        (including unsupported effort / thinking values) are safe to pass."""
        if tools is not None:
            kwargs = {**kwargs, "tools": tools}
        rendered = self.tokenizer.apply_chat_template(_PROBE_MESSAGES, tokenize=False, **kwargs)
        return rendered if isinstance(rendered, str) else str(rendered)

    def _sanitize_effort(self, chat_template_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Quantize a client's ``reasoning_effort`` onto the probed vocabulary.

        Every render path (``encode``, the count proxy, the frontend) must
        quantize identically, so the value reaching the template is the one the
        checkpoint actually grades. Unsupported values map to the nearest
        supported gear (or are dropped to the template default)."""
        if "reasoning_effort" not in chat_template_kwargs:
            return chat_template_kwargs
        raw = chat_template_kwargs.get("reasoning_effort")
        mapped = quantize_effort(raw, self.effort_profile())
        if mapped == raw:
            return chat_template_kwargs
        # raw is client-controlled and may be unhashable (a JSON list/dict).
        key = (raw if isinstance(raw, str) else repr(raw), mapped)
        if key not in self._logged_effort_maps:
            self._logged_effort_maps.add(key)
        sanitized = dict(chat_template_kwargs)
        if mapped is None:
            del sanitized["reasoning_effort"]
        else:
            sanitized["reasoning_effort"] = mapped
        return sanitized


__all__ = ["TokenizeManager", "resolve_thinking_mode"]
