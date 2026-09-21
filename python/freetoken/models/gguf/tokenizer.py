"""GGUF's embedded tokenizer -> a HF ``AutoTokenizer``-compatible object.

Upstream NVIDIA path: python/freetoken/models/gguf/
Issue: `models-gguf-config-tokenizer` (#272, see docs/architecture.md). Parent
epic: `models-gguf` (#199).

GGUF ships no separate HF tokenizer directory (``tokenizer.json`` /
``vocab.json`` / ``merges.txt``): the vocab, merges, per-token scores, and
special-token ids are embedded directly in the checkpoint's own KV-metadata
store under the ``tokenizer.ggml.*`` namespace (see
:func:`freetoken.models.gguf.reader.load_gguf_metadata`). ``freetoken.
tokenizer.tokenize`` (the port's only tokenizer-consuming code path) drives a
transformers ``AutoTokenizer`` instance via ``apply_chat_template`` /
``encode`` / ``decode`` and knows nothing about GGUF -- its own docstring
flags GGUF's custom tokenizer as a known divergence from the ZMQ-process
design this port drops.

The approach taken here (the simpler of the two the issue allows): rebuild
the embedded vocab as a real ``tokenizers`` library ``Tokenizer``, save it as
a standard HF ``tokenizer.json`` + ``tokenizer_config.json`` pair in a fresh
temp directory, and hand that directory to ``transformers.AutoTokenizer.
from_pretrained`` -- the exact same call ``freetoken.utils.hf.load_tokenizer``
already makes for every non-GGUF checkpoint. Downstream code (``tokenize.py``,
``TokenizeManager``) therefore needs zero changes: it receives a real
``PreTrainedTokenizerFast`` either way.

Two ``tokenizer.ggml.model`` values are supported (llama.cpp's own
``tokenizer_model`` string, see ``src/llama-vocab.cpp``):

* ``"gpt2"`` -- a byte-level BPE vocab + merges (Qwen's own tokenizer family,
  including the real target Qwen3.6-35B-A3B checkpoint per epic #199's "Why").
* ``"llama"`` -- a SentencePiece Unigram vocab + per-token scores (the format
  llama.cpp's own tiny CI fixtures -- e.g. ``tinyllamas/stories260K.gguf``,
  this issue's test fixture -- ship). Byte-fallback tokens (``tokenizer.ggml.
  token_type`` == BYTE, spelled ``"<0x1A>"`` etc.) are wired through
  ``Unigram(..., byte_fallback=True)`` so raw bytes never lose round-trip
  fidelity.

Any other ``tokenizer.ggml.model`` value (``"bert"``/``"t5"``/``"rwkv"``/...)
raises ``NotImplementedError`` naming the unsupported model -- this issue's
accept criteria only requires the real target's family (gpt2 BPE) and the
test fixture's own family (llama SPM), not every vocab type llama.cpp support.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from .reader import load_gguf_metadata

# llama.cpp's llama_token_type enum (src/llama-vocab.h) -- only BYTE and
# UNKNOWN matter here (BYTE tokens need byte-fallback wiring; UNKNOWN gives
# the Unigram model's ``unk_id``); the rest (NORMAL/CONTROL/USER_DEFINED/
# UNUSED) don't change how this module builds the vocab.
_TOKEN_TYPE_UNKNOWN = 2
_TOKEN_TYPE_BYTE = 6

_METASPACE_REPLACEMENT = "▁"  # "▁", SentencePiece's own space marker


class GGUFTokenizerError(ValueError):
    """Raised when a GGUF file's embedded tokenizer metadata can't be built
    into a real HF tokenizer (an unsupported ``tokenizer.ggml.model``, or
    missing required fields for the model type it does claim)."""


def _metadata_for(path_or_metadata: "str | Dict[str, Any]") -> Dict[str, Any]:
    """Accept either a GGUF file path or an already-parsed metadata dict, so
    a caller that already called :func:`load_gguf_metadata` (or
    :func:`freetoken.models.gguf.load_gguf`) never re-parses the file."""
    if isinstance(path_or_metadata, dict):
        return path_or_metadata
    return load_gguf_metadata(path_or_metadata)


def _build_gpt2_bpe_tokenizer(meta: Dict[str, Any]):
    """A byte-level BPE ``tokenizers.Tokenizer`` from ``tokenizer.ggml.tokens``
    + ``tokenizer.ggml.merges`` -- the vocab type Qwen's own tokenizer family
    (including the real Qwen3.6-35B-A3B target) uses. Matches GPT-2's own
    byte-level pre-tokenizer/decoder (the convention every ``"gpt2"``-model
    GGUF checkpoint's vocab is already encoded in -- each vocab entry is a
    GPT-2 byte-alphabet string, not raw UTF-8 text).
    """
    from tokenizers import Tokenizer, decoders, pre_tokenizers
    from tokenizers.models import BPE

    tokens: List[str] = meta.get("tokenizer.ggml.tokens") or []
    raw_merges = meta.get("tokenizer.ggml.merges") or []
    if not tokens:
        raise GGUFTokenizerError("GGUF metadata has no tokenizer.ggml.tokens (gpt2 vocab)")
    vocab = {tok: i for i, tok in enumerate(tokens)}
    # Each merge is stored as a single "left right" string (llama.cpp's own
    # on-disk spelling, see gguf-py's tokenizer writer); split on the first
    # space only, since a piece can itself legitimately contain the
    # byte-level space marker "Ġ" (never a literal ASCII space -- GPT-2's
    # byte alphabet never maps a raw byte to 0x20).
    merges: List[Tuple[str, str]] = [tuple(m.split(" ", 1)) for m in raw_merges]  # type: ignore[misc]

    unk_token = None
    unk_id = meta.get("tokenizer.ggml.unknown_token_id")
    if isinstance(unk_id, int) and 0 <= unk_id < len(tokens):
        unk_token = tokens[unk_id]

    tok = Tokenizer(BPE(vocab, merges, unk_token=unk_token, byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    return tok


def _build_llama_unigram_tokenizer(meta: Dict[str, Any]):
    """A SentencePiece-Unigram ``tokenizers.Tokenizer`` from ``tokenizer.
    ggml.tokens`` + ``.scores`` (+ ``.token_type`` for byte-fallback and the
    unknown-token id) -- the vocab type llama.cpp's own tiny CI fixtures ship
    (this issue's test fixture, ``tinyllamas/stories260K.gguf``).
    """
    from tokenizers import Tokenizer, decoders, pre_tokenizers
    from tokenizers.models import Unigram

    tokens: List[str] = meta.get("tokenizer.ggml.tokens") or []
    scores: List[float] = meta.get("tokenizer.ggml.scores") or []
    token_types: List[int] = meta.get("tokenizer.ggml.token_type") or []
    if not tokens:
        raise GGUFTokenizerError("GGUF metadata has no tokenizer.ggml.tokens (llama/SPM vocab)")
    if not scores:
        # A checkpoint that never set per-token scores (rare, but the KV type
        # is technically optional): every score defaults to 0.0, matching
        # llama.cpp's own behavior when the array is absent.
        scores = [0.0] * len(tokens)

    unk_id = meta.get("tokenizer.ggml.unknown_token_id")
    if not isinstance(unk_id, int):
        unk_id = next((i for i, t in enumerate(token_types) if t == _TOKEN_TYPE_UNKNOWN), 0)

    vocab = list(zip(tokens, (float(s) for s in scores)))
    has_byte_tokens = any(t == _TOKEN_TYPE_BYTE for t in token_types)
    tok = Tokenizer(Unigram(vocab, unk_id=unk_id, byte_fallback=has_byte_tokens))
    tok.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement=_METASPACE_REPLACEMENT, prepend_scheme="always"
    )
    tok.decoder = decoders.Metaspace(replacement=_METASPACE_REPLACEMENT, prepend_scheme="always")
    return tok


# tokenizer.ggml.model -> builder. Any other value raises (see module docstring).
_BUILDERS = {
    "gpt2": _build_gpt2_bpe_tokenizer,
    "llama": _build_llama_unigram_tokenizer,
}


def _special_token(tokens: List[str], token_id: Any) -> Optional[str]:
    """Resolve a ``tokenizer.ggml.*_token_id`` KV entry to its token string.

    GGUF spells "unset" as the max uint32 (``4294967295``, i.e. ``-1`` stored
    unsigned) rather than omitting the key -- ``seperator_token_id`` /
    ``padding_token_id`` on the llama-format fixture both do this. Treat any
    id outside the real vocab range as unset.
    """
    if not isinstance(token_id, int) or not (0 <= token_id < len(tokens)):
        return None
    return tokens[token_id]


def materialize_hf_tokenizer_dir(
    path_or_metadata: "str | Dict[str, Any]", out_dir: Optional[str] = None
) -> str:
    """Build the embedded GGUF tokenizer into a standard HF tokenizer
    directory (``tokenizer.json`` + ``tokenizer_config.json``) and return its
    path, ready for ``transformers.AutoTokenizer.from_pretrained``.

    ``out_dir`` is created (``tempfile.mkdtemp``) when not given -- the
    caller owns cleanup (this mirrors every other temp-artifact convention in
    this port; nothing here registers an atexit handler). Passing an existing
    empty directory is also fine (a test fixture's own ``tmp_path``, e.g.).
    """
    meta = _metadata_for(path_or_metadata)
    model_type = meta.get("tokenizer.ggml.model")
    builder = _BUILDERS.get(model_type)
    if builder is None:
        raise NotImplementedError(
            f"GGUF tokenizer.ggml.model={model_type!r} is not supported "
            f"(supported: {sorted(_BUILDERS)})"
        )
    tok = builder(meta)

    tokens: List[str] = meta.get("tokenizer.ggml.tokens") or []
    bos_token = _special_token(tokens, meta.get("tokenizer.ggml.bos_token_id"))
    eos_token = _special_token(tokens, meta.get("tokenizer.ggml.eos_token_id"))
    unk_token = _special_token(tokens, meta.get("tokenizer.ggml.unknown_token_id"))
    pad_token = _special_token(tokens, meta.get("tokenizer.ggml.padding_token_id"))

    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="freetoken-gguf-tokenizer-")
    os.makedirs(out_dir, exist_ok=True)
    tok.save(os.path.join(out_dir, "tokenizer.json"))

    from transformers import PreTrainedTokenizerFast

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=bos_token,
        eos_token=eos_token,
        unk_token=unk_token,
        pad_token=pad_token,
    )
    # add_bos_token / add_eos_token (GGUF's own convention: whether the
    # chat/encode path should automatically prepend/append them) are stashed
    # verbatim in tokenizer_config.json so a downstream reader can see the
    # checkpoint's own intent even though this port's own encode path
    # (freetoken.tokenizer.tokenize) drives everything through the chat
    # template instead of these flags.
    fast.add_bos_token = bool(meta.get("tokenizer.ggml.add_bos_token", False))
    fast.add_eos_token = bool(meta.get("tokenizer.ggml.add_eos_token", False))
    chat_template = meta.get("tokenizer.chat_template")
    if chat_template:
        fast.chat_template = chat_template
    fast.save_pretrained(out_dir)
    return out_dir


def load_gguf_tokenizer(path_or_metadata: "str | Dict[str, Any]", out_dir: Optional[str] = None):
    """Materialize the embedded tokenizer and load it back through
    ``transformers.AutoTokenizer`` -- the exact loader every other checkpoint
    format in this port uses (:func:`freetoken.utils.hf.load_tokenizer`),
    so nothing downstream (``freetoken.tokenizer.tokenize``) needs to know
    the tokenizer came from a GGUF file rather than a real HF directory.
    """
    from transformers import AutoTokenizer

    tokenizer_dir = materialize_hf_tokenizer_dir(path_or_metadata, out_dir=out_dir)
    return AutoTokenizer.from_pretrained(tokenizer_dir)


__all__ = [
    "GGUFTokenizerError",
    "materialize_hf_tokenizer_dir",
    "load_gguf_tokenizer",
]
