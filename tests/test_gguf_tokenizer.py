"""Tests for GGUF's embedded tokenizer -> HF AutoTokenizer (issue
`models-gguf-config-tokenizer`, #272).

Uses the same real GGUF fixture `test_gguf_reader.py`/`test_gguf_config.py`
already download (`ggml-org/models`' `tinyllamas/stories260K.gguf`) --
`tokenizer.ggml.model == "llama"` (a SentencePiece-Unigram vocab with
byte-fallback tokens), so this exercises the SPM path end to end. The
byte-level BPE ("gpt2") path -- the real target family's own tokenizer
family (Qwen3.6) -- is covered separately with a small synthetic vocab, since
no tiny real gpt2-vocab GGUF fixture is bundled by llama.cpp's own CI set.
"""
from __future__ import annotations

import json
import os

import pytest

from freetoken.models.gguf.tokenizer import (
    GGUFTokenizerError,
    load_gguf_tokenizer,
    materialize_hf_tokenizer_dir,
)

_HF_REPO = "ggml-org/models"
_F32_FIXTURE = "tinyllamas/stories260K.gguf"


def _hf_download(filename: str) -> str:
    huggingface_hub = pytest.importorskip("huggingface_hub")
    try:
        return huggingface_hub.hf_hub_download(_HF_REPO, filename)
    except Exception as exc:  # pragma: no cover - network/offline environment
        pytest.skip(f"could not download real GGUF fixture {filename} from the Hub: {exc}")


@pytest.fixture(scope="module")
def f32_gguf_path() -> str:
    return _hf_download(_F32_FIXTURE)


@pytest.mark.slow
def test_materialize_hf_tokenizer_dir_writes_standard_hf_files(f32_gguf_path, tmp_path):
    out = materialize_hf_tokenizer_dir(f32_gguf_path, out_dir=str(tmp_path / "tok"))
    assert os.path.isfile(os.path.join(out, "tokenizer.json"))
    assert os.path.isfile(os.path.join(out, "tokenizer_config.json"))


@pytest.mark.slow
def test_load_gguf_tokenizer_round_trips_a_real_string(f32_gguf_path):
    pytest.importorskip("transformers")
    tok = load_gguf_tokenizer(f32_gguf_path)
    s = "Once upon a time, Tom and Lily went to the park."
    ids = tok.encode(s, add_special_tokens=False)
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
    assert tok.decode(ids) == s


@pytest.mark.slow
def test_load_gguf_tokenizer_resolves_special_tokens_against_the_fixtures_own_vocab(f32_gguf_path):
    # Hand-verified in test_gguf_reader.py: bos_token_id=1 ("<s>"),
    # eos_token_id=2 ("</s>").
    pytest.importorskip("transformers")
    tok = load_gguf_tokenizer(f32_gguf_path)
    assert tok.bos_token_id == 1
    assert tok.bos_token == "<s>"
    assert tok.eos_token_id == 2
    assert tok.eos_token == "</s>"


@pytest.mark.slow
def test_load_gguf_tokenizer_accepts_a_pre_parsed_metadata_dict(f32_gguf_path):
    pytest.importorskip("transformers")
    from freetoken.models.gguf import load_gguf_metadata

    meta = load_gguf_metadata(f32_gguf_path)
    tok = load_gguf_tokenizer(meta)
    assert tok.encode("Tom", add_special_tokens=False)


def test_gpt2_bpe_vocab_round_trips(tmp_path):
    # The real target family's own tokenizer type (Qwen3.6): byte-level BPE,
    # not SentencePiece. No tiny real gpt2-vocab GGUF fixture is available in
    # this issue's fixture set, so a small real ``tokenizers`` BPE model is
    # trained here and its vocab/merges are round-tripped through GGUF's own
    # on-disk spelling (tokenizer.ggml.tokens / .merges, each merge a single
    # "left right" string) -- the exact shape _build_gpt2_bpe_tokenizer reads.
    bpe = pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer as _Tokenizer
    from tokenizers import decoders as _decoders
    from tokenizers import pre_tokenizers as _pre_tokenizers
    from tokenizers import trainers as _trainers
    from tokenizers.models import BPE as _BPE

    trained = _Tokenizer(_BPE(unk_token=None, byte_fallback=False))
    trained.pre_tokenizer = _pre_tokenizers.ByteLevel(add_prefix_space=False)
    trained.decoder = _decoders.ByteLevel()
    trained.train_from_iterator(
        ["a cat sat on a mat", "the cat sat", "a cat and a mat"],
        trainer=_trainers.BpeTrainer(vocab_size=300, special_tokens=["<|endoftext|>"]),
    )
    vocab_by_id = trained.get_vocab()
    tokens = [None] * len(vocab_by_id)
    for tok_str, idx in vocab_by_id.items():
        tokens[idx] = tok_str
    merges_raw = json.loads(trained.to_str())["model"]["merges"]
    merges = [" ".join(m) if isinstance(m, (list, tuple)) else m for m in merges_raw]

    meta = {
        "general.architecture": "qwen2",
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.merges": merges,
        "tokenizer.ggml.bos_token_id": 0,
        "tokenizer.ggml.eos_token_id": 0,
    }
    pytest.importorskip("transformers")
    tok = load_gguf_tokenizer(meta, out_dir=str(tmp_path / "tok"))
    s = "a cat sat on a mat"
    ids = tok.encode(s, add_special_tokens=False)
    assert tok.decode(ids) == s
    # Cross-check against the same trained tokenizer's own encode: GGUF's
    # tokens/merges round-trip byte-for-byte, not just to *some* valid tokenization.
    assert ids == trained.encode(s).ids
    assert tok.bos_token_id == 0


def test_unsupported_tokenizer_model_raises():
    with pytest.raises(NotImplementedError):
        materialize_hf_tokenizer_dir({"tokenizer.ggml.model": "rwkv", "tokenizer.ggml.tokens": ["a"]})


def test_missing_tokens_raises_gguf_tokenizer_error():
    with pytest.raises(GGUFTokenizerError):
        materialize_hf_tokenizer_dir({"tokenizer.ggml.model": "llama", "tokenizer.ggml.tokens": []})
