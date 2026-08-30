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

Token-id <-> text: the message frontend (chat-template encode / decode) is still
a stub (``server-openai`` tokenizer seam), so :func:`_prompt_token_ids` falls
back to a deterministic hash mapping and :func:`_decode_token` falls back to a
readable placeholder until the real tokenizer path lands.
"""
from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Iterator

from freetoken._stub import NotYetImplemented

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


class EngineNotReady(NotYetImplemented):
    """The HTTP surface is wired but the generation backend is still a stub."""


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


def _prompt_token_ids(engine, prompt: str) -> list[int]:
    """Encode ``prompt`` to the token ids the engine consumes.

    Uses the model's real tokenizer / chat template when the engine exposes one
    (``tokenizer`` with ``encode``/``apply_chat_template``); until the message
    frontend (``server-openai`` tokenizer seam) lands, it falls back to a
    deterministic hash of the prompt bound to the model's own vocab, so the
    request stays well-formed (a single in-vocab id) and the engine's prefill +
    embedding / KV math is exercised end to end without a tokenizer on the seam.
    """
    tokenizer = getattr(engine, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(prompt))
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    return [int.from_bytes(digest[:4], "big") % _fallback_id_space(engine)]


def _decode_token(engine, token_id: int) -> str:
    """Decode one generated token id to text (or a readable placeholder).

    Mirrors :func:`_prompt_token_ids`: a real ``tokenizer.decode`` when one is
    attached, else a stable ``<tok-<id>>`` placeholder so the stream is
    non-empty and the HTTP seam is exercisable before the tokenizer lands.
    """
    tokenizer = getattr(engine, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "decode"):
        return tokenizer.decode([token_id])
    return f"<tok-{token_id}>"


def _stream_tokens(engine, prompt: str, *, model: str, max_tokens: int | None) -> Iterator[str]:
    """Admit ``prompt`` to the engine and yield each generated token's text.

    Two engine shapes are supported, selected by what the engine exposes:

    * A real :class:`~freetoken.engine.engine.Engine` (``#14`` / ``#93``)
      exposes ``add_request`` + a no-arg ``generate``. It is driven by the id
      path: encode the prompt to ids, admit it with the requested decode budget,
      run ``generate``, and decode each emitted id.

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
            input_ids=_prompt_token_ids(engine, prompt),
            table_idx=0,  # add_request reassigns the real slot index
            cached_len=0,
            output_len=budget,
            uid=0,
            sampling_params=SamplingParams(max_tokens=budget),
            cache_handle=None,
        )
    )
    token_lists = generate()
    for token_id in token_lists[0] if token_lists else []:
        yield _decode_token(engine, token_id)


def stream_chat(
    engine,
    messages: list[dict],
    *,
    model: str,
    max_tokens: int | None = None,
    reasoning_parser=None,
):
    """Yield ``(reasoning_delta, content_delta)`` per decoded token.

    ``reasoning_parser`` is stateful across the whole stream (it buffers
    partial tags), so each token is fed through it and ``flush()`` drains the
    remainder once the stream ends. With no parser, every token is content.
    """
    prompt = _prompt_from_messages(messages)
    if reasoning_parser is None:
        for token in _stream_tokens(engine, prompt, model=model, max_tokens=max_tokens):
            yield "", token
        return
    for token in _stream_tokens(engine, prompt, model=model, max_tokens=max_tokens):
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
