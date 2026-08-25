"""Tests for the Hugging Face helpers (issue ``models-loader``, #17).

All of these run offline on a CPU-only box: they read a locally fabricated
``config.json`` and never touch the network (the network path is isolated in
``download_hf_weight`` and is only exercised for non-local paths).
"""
from __future__ import annotations

import json

import pytest

from freetoken.utils import cached_load_hf_config
from freetoken.utils.hf import RawConfigShim, load_eos_token_ids, load_tokenizer

GPT2_LIKE_CONFIG = {
    "architectures": ["GPT2ForSequenceClassification"],
    "model_type": "gpt2",
    "vocab_size": 50257,
    "hidden_size": 16,
    "n_layer": 1,
    "n_head": 2,
}


def _write_config(tmp_path, config: dict) -> str:
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


def test_cached_load_hf_config_returns_pretrained_config(tmp_path):
    path = _write_config(tmp_path, GPT2_LIKE_CONFIG)
    cfg = cached_load_hf_config(path)
    assert cfg.architectures == ["GPT2ForSequenceClassification"]
    assert cfg.vocab_size == 50257
    # cached: same object for the same path
    assert cached_load_hf_config(path) is cfg


def test_cached_load_hf_config_falls_back_to_shim_on_unknown_type(tmp_path):
    # A model_type the installed transformers does not know -> AutoConfig raises
    # -> the raw-JSON shim is served instead (offline).
    path = _write_config(tmp_path, {"architectures": ["FooForCausalLM"], "model_type": "totally_unknown_x"})
    cfg = cached_load_hf_config(path)
    assert isinstance(cfg, RawConfigShim)
    assert "FooForCausalLM" in cfg.architectures
    assert cfg._name_or_path == path


def _write_vocab(tmp_path, extra: dict = {}) -> None:
    # GPT2's tokenizer needs a vocab.json + merges.json (both filenames, or both
    # in-memory). Provide both files so AutoTokenizer loads fully offline.
    vocab = {"<pad>": 0, "hello": 1, "world": 2}
    vocab.update(extra)
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    (tmp_path / "merges.txt").write_text("")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"model_type": "gpt2", "eos_token": "<pad>", "bos_token": "<pad>"})
    )


def test_load_tokenizer_reads_local_vocab(tmp_path):
    # Fabricate a real (tiny) vocab file alongside the config so AutoTokenizer
    # can load it offline.
    _write_vocab(tmp_path)
    _write_config(tmp_path, GPT2_LIKE_CONFIG)
    tok = load_tokenizer(str(tmp_path))
    assert tok.eos_token == "<pad>"
    assert tok.eos_token_id == 0


def test_load_eos_token_ids_unions_generation_config(tmp_path):
    _write_vocab(tmp_path, extra={"</s>": 3})
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [3]}))
    _write_config(tmp_path, GPT2_LIKE_CONFIG)
    tok = load_tokenizer(str(tmp_path))
    ids = load_eos_token_ids(str(tmp_path), tok)
    # tokenizer eos (0) unioned with the generation config's [3]
    assert ids == frozenset({0, 3})
