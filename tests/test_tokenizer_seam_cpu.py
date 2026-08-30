"""CPU round-trip for the message frontend seam (issue ``server-tokenizer``, #95).

The tokenizer seam is torch-free (``AutoTokenizer`` — text only), so its core
invariant is proven here on a CPU box in ``.venv`` (no XPU, no torch needed for
the encode/decode math): the chat-template *encode* ids and the incremental
*decode* deltas must round-trip exactly, so the stream a client sees equals what
a single greedy ``tokenizer.decode(all_ids)`` would yield. The full live engine
driving these ids on XPU is covered separately by ``test_serve_live_engine_xpu``.

Skipped cleanly where the hero tokenizer is not cached (no network in CI).
"""
from __future__ import annotations

import os

import pytest

# Resolve the tokenizer offline: a CPU box without the cache skips rather than
# triggering a hub download (the dual-venv contract keeps .venv lean).
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL = "Qwen/Qwen3-0.6B"

pytest.importorskip("transformers")

from pathlib import Path  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402

from freetoken.server import generation  # noqa: E402
from freetoken.tokenizer.detokenize import DetokenizeManager  # noqa: E402
from freetoken.tokenizer.tokenize import TokenizeManager  # noqa: E402


def _tokenizer_cached() -> bool:
    """True when the model's tokenizer files are already in the local HF cache."""
    cache_root = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    repo_dir = Path(cache_root) / "hub" / ("models--" + MODEL.replace("/", "--"))
    return repo_dir.exists() and any(repo_dir.glob("snapshots/*/tokenizer*.json"))


pytestmark = pytest.mark.skipif(
    not _tokenizer_cached(), reason=f"{MODEL} tokenizer not cached (offline CPU box)"
)


@pytest.fixture(scope="module")
def mgr():
    """The model's message frontend: chat-template encode + incremental decode."""
    return TokenizeManager(AutoTokenizer.from_pretrained(MODEL))


# A spread of strings that stress the BPE boundaries the incremental decoder must
# handle: short, long, punctuation, a long unbroken run (the rollback case), and
# text that spans many merges.
_MESSAGES = [
    [{"role": "user", "content": "Count from one."}],
    [{"role": "user", "content": "Hello"}],
    [{"role": "user", "content": "The answer is forty-two and it is not a placeholder."}],
    [{"role": "user", "content": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}],
    [{"role": "user", "content": "Mixed case, punctuation! 1234567890. Tab\there and a newline\nbelow."}],
    [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "One word."}],
]


def test_encode_is_deterministic_and_in_vocab(mgr):
    """Chat-template encode is stable and produces valid ids."""
    ids_a = mgr.encode(_MESSAGES[0])
    ids_b = mgr.encode(_MESSAGES[0])
    assert ids_a == ids_b, "encode must be deterministic for the same messages"
    assert ids_a, "encode must not be empty"
    # The ids must be valid embedding-row indices: < the model's config vocab (the
    # embedding-table width). NOTE: .tokenizers' ``vocab_size`` is a merge count
    # (151643) *below* the chat template's special-token ids (151644/151645), so
    # the bound must be the config vocab, not the tokenizer's. (The XPU gather
    # aborts on an out-of-bounds id — the whole point of this invariant.)
    for token_id in ids_a:
        assert 0 <= token_id < len(mgr.tokenizer), f"encoded id outside vocab: {token_id}"


def test_incremental_decode_equals_greedy_decode(mgr):
    """The concatenated incremental deltas equal a single greedy decode.

    This is the load-bearing invariant of ``#95``: streaming per-token deltas
    (each id folded through the ``DetokenizeManager`` with its trailing-token
    rollback) must reproduce exactly what ``tokenizer.decode(all_ids)`` yields,
    so a client reading the stream sees the same text as a batch decode.
    """
    for messages in _MESSAGES:
        ids = mgr.encode(messages)
        detok = DetokenizeManager(mgr.tokenizer, stop_strs=None)
        uid = "seam"
        detok.create(uid)
        try:
            streamed = "".join(
                delta for i in ids for delta in [detok.update(uid, [i])] if delta
            )
        finally:
            detok.delete(uid)
        expected = mgr.tokenizer.decode(ids)
        assert streamed == expected, (
            f"incremental decode diverged from greedy decode:\n"
            f"  streamed={streamed!r}\n  expected={expected!r}"
        )


def test_prompt_ids_from_stream_chat_match_encode(mgr):
    """The prompt ids the seam feeds the engine are exactly the template's ids."""
    # A stub engine that only records the admitted ids (no real generate needed
    # for this assertion; _prompt_token_ids is what under add_request consumes).
    class _Recorder:
        config = type("c", (), {"model_config": type("m", (), {"vocab_size": mgr.tokenizer.vocab_size})})()
        input_ids = None
        add_request = lambda self, r: setattr(self, "input_ids", r.input_ids)

    eng = _Recorder()
    eng.frontend_tokenizer = mgr
    got = generation._prompt_token_ids(eng, "ignored", _MESSAGES[0])
    assert got == mgr.encode(_MESSAGES[0])


def test_create_app_installs_resolver_hook(monkeypatch):
    """create_app installs the lazy frontend resolver (the #95 wiring)."""
    from freetoken.server.api_server import create_app
    from freetoken.server.args import parse_args

    monkeypatch.setattr(generation, "_frontend_tokenizer_hook", None)
    create_app(parse_args([MODEL]), lambda: (_ for _ in ()).throw(RuntimeError("no engine")))
    assert generation._frontend_tokenizer_hook is not None, "create_app must install the frontend hook"
    resolved = generation._frontend_tokenizer_hook()
    assert isinstance(resolved, TokenizeManager)
    monkeypatch.setattr(generation, "_frontend_tokenizer_hook", None)
