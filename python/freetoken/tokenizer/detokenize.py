"""Incremental decoding: token ids -> printable deltas for the OpenAI seam.

Upstream NVIDIA path: python/freetoken/tokenizer/detokenize.py

The per-uid offset state machine is ported faithfully (single-trailing-token
rollback, U+FFFD printable-text trimming, ``stop_strs`` prefix hold-back,
ASCII/CJK boundary space insertion). Two upstream concerns are dropped for the
in-process seam: the ``DetokenizeReply`` ZMQ round-trip (``update`` simply
returns the new delta text) and the ``finished``/``matched_stop`` flush
(the engine owns sequence termination, not the decoder).
"""

from __future__ import annotations

import re
from typing import Any

# Matches a leading run of non-printable / whitespace characters.
# Ported from upstream tokenizer/detokenize.py.
_NON_PRINTABLE_RE = re.compile(r"^[ \t\n\r\f\v\x85\u1c2f\u1c3f\u200a-\u200c\u2028\u2029]+")


def find_printable_text(text: str) -> str:
    """Return ``text`` with leading whitespace / control chars stripped.

    A U+FFFD (here and downstream) may be preceded by a run of whitespace that
    the tokenizer emitted but should not be surfaced at the start of a stream.
    """
    if not text:
        return ""
    match = _NON_PRINTABLE_RE.match(text)
    if match:
        return text[match.end():]
    return text


class _DecodeState:
    __slots__ = ("decoded_ids", "decoded_str", "read_offset", "surr_offset", "sent_offset")

    def __init__(self) -> None:
        self.decoded_ids: list[int] = []
        self.decoded_str = ""
        self.read_offset = 0
        self.surr_offset = 0
        self.sent_offset = 0


def _stop_prefix_holdback(text: str, stop_strs: list[str]) -> int:
    """Length of a trailing suffix of ``text`` that is a strict prefix of a stop string.

    We hold that suffix back so the client does not paint it and then have to
    repaint when the stop string (or a longer token) arrives. Zero when no
    suffix of ``text`` is a proper prefix of any stop string.
    """
    if not stop_strs or not text:
        return 0
    max_len = min(len(text), max(len(s) for s in stop_strs))
    for length in range(max_len, 0, -1):
        suffix = text[-length:] if length < len(text) else text
        for stop in stop_strs:
            if len(stop) > length and stop.startswith(suffix):
                return length
    return 0


class DetokenizeManager:
    """Incrementally decode token-id sequences into printable text deltas.

    ``create(uid)`` mints a decode context; ``update(uid, ids)`` appends ids and
    returns the newly printable slice (empty until a printable boundary is
    reached); ``delete(uid)`` frees it. The trailing-token rollback means a
    delta is only emitted once it is known to be stable, so the concatenated
    stream equals the result of a single greedy decode of the full id sequence.
    """

    def __init__(self, tokenizer: Any, stop_strs: list[str] | None = None) -> None:
        self.tokenizer = tokenizer
        self.stop_strs = list(stop_strs or [])
        self.decode_map: dict[str, _DecodeState] = {}

    def create(self, uid: str) -> _DecodeState:
        state = _DecodeState()
        self.decode_map[uid] = state
        return state

    def delete(self, uid: str) -> None:
        self.decode_map.pop(uid, None)

    def update(self, uid: str, ids: list[int]) -> str:
        s = self.decode_map.get(uid)
        if s is None:
            raise KeyError(f"no decode context for uid={uid!r}; call create() first")
        s.decoded_ids.extend(ids)
        read_ids = s.decoded_ids[s.surr_offset :]
        surr_ids = s.decoded_ids[s.surr_offset : s.read_offset]

        read_str = self.tokenizer.decode(read_ids, skip_special_tokens=False)
        surr_str = self.tokenizer.decode(surr_ids, skip_special_tokens=False)

        new_text = read_str[len(surr_str) :]
        if len(new_text) > 0 and not new_text.endswith("\ufffd"):
            output_str = s.decoded_str + new_text
            s.decoded_str = output_str
            s.surr_offset = s.read_offset
            s.read_offset = len(s.decoded_ids)
        else:
            new_text = find_printable_text(new_text)
            output_str = s.decoded_str + new_text

        prev_sent = s.sent_offset
        if self.stop_strs:
            emit_end = len(output_str) - _stop_prefix_holdback(output_str, self.stop_strs)
        else:
            emit_end = len(output_str)
        incremental_output = output_str[prev_sent:emit_end] if emit_end > prev_sent else ""
        s.sent_offset = max(prev_sent, emit_end)

        return incremental_output


__all__ = ["DetokenizeManager", "find_printable_text"]
