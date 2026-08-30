"""Request -> engine generation path, streaming.

Upstream NVIDIA path: python/freetoken/server/generation.py

This is the seam between the HTTP surface and the engine. The OpenAI routes
call :func:`stream_chat`; it encodes the messages to prompt token ids, admits
them to the loaded engine, and streams the generated token ids back, splitting
reasoning from content via the configured reasoning parser.

Boundary: the engine loop (``#14``) is real, so :func:`_stream_tokens` drives
the real ``Engine`` (``add_request`` + ``generate``). The only remaining not-ready
signal is a genuinely unloaded engine (no ``generate`` / no ``add_request``, e.g.
the ``ft serve`` spine's pre-load holder), which raises
:class:`EngineNotReady` (a ``NotYetImplemented``) that the routes map to a clean
503 — the server stays up and ``/v1/models`` keeps answering.

Token-id <-> text: the message frontend is now real (``#95``) — prompts are
rendered through the model's chat template (:class:`TokenizeManager`) and
generated ids are decoded incrementally (:class:`DetokenizeManager`), both
attached lazily via ``frontend_tokenizer()`` so building the app and importing
this module stay torch-free; the tokenizer loads on the first request. If no
tokenizer is attached (e.g. the pre-load spine holder) the prompt encoder falls
back to a single in-vocab id and the decoder to a readable ``<tok-<id>>``
placeholder, so the seam stays exercisable end to end.
"""
from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Iterator
from typing import Any

from freetoken._stub import NotYetImplemented
from freetoken.tokenizer.detokenize import DetokenizeManager
from freetoken.tokenizer.tokenize import TokenizeManager

# The reference engine's sampler stops on eos_token_id=-1, so a request never
# hits an end-of-sequence id; the decode budget comes from max_tokens (or the
# engine's per-request cap when max_tokens is None). This cap bounds a
# tokenizer-less smoke test so a mis-set request cannot spin the decode loop.
_FALLBACK_MAX_TOKENS = 64
# The prompt encoder's fallback hashes the prompt into the model's *own* vocab
# (see _prompt_token_ids), not a fixed id space: a fixed 4096 would mint ids far
# past a small model's vocab, and the embedding layer would gather out of bounds
# (the engine's KV / embedding math only stays well-formed when every prompt id
# is < vocab_size).
_FALLBACK_ID_SPACE_MAX = 4096

# App-level resolver for the message frontend (chat-template encode / decode),
# installed by api_server.create_app. The per-request path prefers the loaded
# engine's own ``frontend_tokenizer`` (attached by the launch holder); this
# hook is the fallback for route tests that drive ``stream_chat`` with a plain
# engine object that never gets the attribute. A zero-arg callable returning a
# ``TokenizeManager`` or ``None`` (not loaded yet / no model path); ``None``
# when no app is installed (pure CPU unit tests of this module).
_frontend_tokenizer_hook: "Any" = None


class EngineNotReady(NotYetImplemented):
    """The HTTP surface is wired but the generation backend is still a stub."""


def set_frontend_tokenizer_hook(hook: "Any") -> None:
    """Install the app-level message-frontend resolver (called by create_app).

    ``hook`` is a zero-arg callable returning a :class:`TokenizeManager`
    (or ``None`` when no model path is configured / not loaded yet). Passing
    ``None`` clears the hook (module reset for tests). The per-request encode /
    decode path prefers the loaded engine's own ``frontend_tokenizer`` and only
    falls back to this hook.
    """
    global _frontend_tokenizer_hook
    _frontend_tokenizer_hook = hook


def _resolve_tokenize_manager(engine) -> Any:
    """The message frontend for this request.

    Prefers the loaded engine's own ``frontend_tokenizer`` (attached by the
    launch holder); falls back to the app-level hook (for route tests that drive
    ``stream_chat`` with a plain engine); returns ``None`` when neither exists
    (tokenizer-less fallback path).
    """
    mgr = getattr(engine, "frontend_tokenizer", None)
    if mgr is not None:
        return mgr
    if _frontend_tokenizer_hook is not None:
        return _frontend_tokenizer_hook()
    return None


def _chat_completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]


def _prompt_from_messages(messages: list[dict]) -> str:
    """Flatten chat messages into the single prompt string the engine takes.

    The real message frontend / chat-template rendering (tokenizer process)
    replaces this once the tokenizer path exists; today it is a
    deterministic, dependency-free join so the route is testable on a CPU box.
    """
    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if isinstance(content, list):  # multimodal content parts
            content = " ".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _fallback_id_space(engine) -> int:
    """The id space the fallback prompt encoder hashes into.

    Bounded by the model's own vocab size (read from the loaded engine's
    ``config.model_config.vocab_size``) so the hash-derived id is always a valid
    embedding-row index: ``embed_tokens`` gathers ``k_buffer[input_ids]``, so an
    id >= vocab_size reads past the embedding table (an out-of-bounds gather that
    the XPU runtime aborts on). A fixed 4096 was wrong for small models. When the
    vocab is unknown (not yet parsed) we fall back to a conservative ceiling.
    """
    vocab = getattr(getattr(getattr(engine, "config", None), "model_config", None), "vocab_size", None)
    if vocab is None:
        return _FALLBACK_ID_SPACE_MAX
    return max(2, min(int(vocab), _FALLBACK_ID_SPACE_MAX))


def _prompt_token_ids(
    engine,
    prompt: str,
    messages: list[dict] | None = None,
    *,
    tools: list[dict] | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[int]:
    """Encode the request to the token ids the engine consumes.

    With the real message frontend (``#95``) the prompt is the chat rendered
    through the model's chat template, so the ids come from
    :class:`TokenizeManager.encode` (the same template the model was trained
    with) — not from re-tokenizing a flattened string, which would not match the
    template's exact boundaries. The caller passes ``messages`` so the chat is
    re-rendered here; ``prompt`` is kept only as the no-tokenizer fallback input.

    ``tools`` and ``chat_template_kwargs`` carry the client's reasoning controls
    (issue #97); they are forwarded to ``encode``, which quantizes effort onto
    the checkpoint's probed profile and resolves the thinking toggle. They are
    no-ops on the no-tokenizer fallback path.
    """
    tokenize_mgr = _resolve_tokenize_manager(engine)
    if tokenize_mgr is not None and hasattr(tokenize_mgr, "encode"):
        return tokenize_mgr.encode(
            messages if messages is not None else [{"role": "user", "content": prompt}],
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
        )
    # No tokenizer attached (pre-load holder): keep the seam exercisable with a
    # single in-vocab id so the engine's prefill + embedding / KV math still run.
    # The client's reasoning controls (reasoning_effort / thinking) only apply
    # through the real chat template, so they are ignored on this path.
    vocab = getattr(getattr(getattr(engine, "config", None), "model_config", None), "vocab_size", None)
    space = max(2, min(int(vocab), _FALLBACK_ID_SPACE_MAX)) if vocab else _FALLBACK_ID_SPACE_MAX
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    return [int.from_bytes(digest[:4], "big") % space]


def _decode_stream(engine, token_ids: list[int]) -> Iterator[str]:
    """Yield printable text deltas for a generated id sequence.

    With the real tokenizer (``#95``) the ids feed a per-request
    :class:`DetokenizeManager`: each id is folded in and the newly *stable* text
    is yielded, so the concatenated deltas equal a single greedy decode (the
    trailing-token rollback holds back whatever might still change). Without a
    tokenizer (pre-load holder) each id yields a ``<tok-<id>>`` placeholder so
    the HTTP seam stays exercisable end to end.
    """
    tokenize_mgr = _resolve_tokenize_manager(engine)
    if tokenize_mgr is not None and hasattr(tokenize_mgr, "tokenizer"):
        stop_strs = getattr(tokenize_mgr, "stop_strs", None)
        detok = DetokenizeManager(tokenize_mgr.tokenizer, stop_strs=stop_strs)
        uid = uuid.uuid4().hex[:16]
        detok.create(uid)
        try:
            for token_id in token_ids:
                delta = detok.update(uid, [token_id])
                if delta:
                    yield delta
        finally:
            detok.delete(uid)
        return
    for token_id in token_ids:
        yield f"<tok-{token_id}>"


def _stream_tokens(
    engine,
    prompt: str,
    *,
    model: str,
    max_tokens: int | None,
    messages: list[dict] | None = None,
    tools: list[dict] | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> Iterator[str]:
    """Admit ``prompt`` to the engine and yield each generated token's text.

    Two engine shapes are supported, selected by what the engine exposes:

    * A real :class:`~freetoken.engine.engine.Engine` (``#14`` / ``#93``)
      exposes ``add_request`` + a no-arg ``generate``. It is driven by the id
      path: render the chat through the model's template to ids, admit it with
      the requested decode budget, run ``generate``, and decode the emitted ids
      incrementally (``#95``).

    * A lighter engine (the CPU route-test stub) exposes only
      ``generate(prompt, *, model, max_tokens)`` and yields token *text*. It is
      driven by the text path — the original ``#14``-era contract — so the HTTP
      seam is testable without torch / a real checkpoint.

    An engine with no ``generate`` at all is the not-loaded signal: raise
    :class:`EngineNotReady` so the routes map it to a clean 503.
    """
    generate = getattr(engine, "generate", None)
    if generate is None:
        raise EngineNotReady(
            "engine loop is not loaded — generation is blocked on "
            "engine-loop (#14); see docs/architecture.md"
        )
    add_request = getattr(engine, "add_request", None)

    if add_request is None:
        # Lightweight engine: generate() takes the prompt and yields token text.
        for token in generate(prompt, model=model, max_tokens=max_tokens):
            yield token
        return

    from freetoken.core import Req, SamplingParams

    budget = max_tokens if max_tokens and max_tokens > 0 else _FALLBACK_MAX_TOKENS
    engine.add_request(
        Req(
            input_ids=_prompt_token_ids(engine, prompt, messages, tools=tools, chat_template_kwargs=chat_template_kwargs),
            table_idx=0,  # add_request reassigns the real slot index
            cached_len=0,
            output_len=budget,
            uid=0,
            sampling_params=SamplingParams(max_tokens=budget),
            cache_handle=None,
        )
    )
    token_lists = generate()
    emitted = token_lists[0] if token_lists else []
    yield from _decode_stream(engine, list(emitted))


def stream_chat(
    engine,
    messages: list[dict],
    *,
    model: str,
    max_tokens: int | None = None,
    reasoning_parser=None,
    tools: list[dict] | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
):
    """Yield ``(reasoning_delta, content_delta)`` per decoded token.

    ``reasoning_parser`` is stateful across the whole stream (it buffers
    partial tags), so each token is fed through it and ``flush()`` drains the
    remainder once the stream ends. With no parser, every token is content.

    ``tools`` and ``chat_template_kwargs`` carry the client's reasoning controls
    (issue #97): they are forwarded to the encode step, where effort is
    quantized onto the checkpoint's probed profile and the thinking toggle is
    resolved, so the rendered prompt reflects the request's effort / thinking.
    """
    prompt = _prompt_from_messages(messages)
    if reasoning_parser is None:
        for token in _stream_tokens(
            engine,
            prompt,
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
        ):
            yield "", token
        return
    for token in _stream_tokens(
        engine,
        prompt,
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        tools=tools,
        chat_template_kwargs=chat_template_kwargs,
    ):
        for reasoning_delta, content_delta in reasoning_parser.parse([token]):
            if reasoning_delta or content_delta:
                yield reasoning_delta, content_delta
    final_reasoning, final_content = reasoning_parser.flush()
    if final_reasoning or final_content:
        yield final_reasoning, final_content


def completion_id() -> str:
    return _chat_completion_id()


def now_timestamp() -> int:
    return int(time.time())


__all__ = ["EngineNotReady", "stream_chat", "completion_id", "now_timestamp"]
