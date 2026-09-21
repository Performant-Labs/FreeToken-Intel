"""Tests for attention backend selection (issue ``attn-auto-is-torch``, #293).

On this port ``auto`` is a deliberate choice — pure-PyTorch GQA (the `torch`
backend) — not a fallback for a missing Triton-Intel install. See
docs/stack.md. These tests are CPU-safe: they never touch the XPU.
"""
from __future__ import annotations

import sys
import types

import pytest

# create_attention_backend / EngineConfig wiring pull in freetoken.attention.triton
# and freetoken.engine.engine, both of which import torch at module scope -- the
# CPU CI venv never has torch (see tests/conftest.py), so this whole module
# self-skips there, same as the other torch-dependent suites.
pytest.importorskip("torch")


def test_create_attention_backend_auto_is_torch():
    from freetoken.attention import create_attention_backend
    from freetoken.attention.triton import TritonAttentionBackend

    backend = create_attention_backend("auto", config=object())
    assert isinstance(backend, TritonAttentionBackend)


def test_auto_stays_torch_even_when_triton_intel_importable(monkeypatch):
    # Simulate Triton-Intel being importable in this environment: `auto` must
    # still resolve to the pure-PyTorch backend -- the choice is not gated on
    # what's installed.
    monkeypatch.setitem(sys.modules, "triton", types.ModuleType("triton"))

    from freetoken.attention import create_attention_backend
    from freetoken.attention.triton import TritonAttentionBackend

    backend = create_attention_backend("auto", config=object())
    assert isinstance(backend, TritonAttentionBackend)


def test_attention_backend_flows_into_engine_config(monkeypatch):
    """Issue #293 accept: --attention-backend sycl reaches
    EngineConfig.attention_backend unchanged."""
    import freetoken.engine.engine as engine_mod
    import freetoken.server.function_call_parser as fcp_mod
    import freetoken.server.launch as launch_mod
    import freetoken.utils.hf as hf_mod
    from freetoken.server.args import parse_args

    captured: dict = {}

    class _FakeEngine:
        def __init__(self, config):
            captured["config"] = config
            self.max_seq_len = 0
            self.frontend_tokenizer = None
            self.toolcall_anchor_id = None

    class _FakeFrontendTokenizer:
        tokenizer = object()

    class _FakeParser:
        toolcall_opener = None

    monkeypatch.setattr(engine_mod, "Engine", _FakeEngine)
    monkeypatch.setattr(launch_mod, "_frontend_tokenizer", lambda _server_args: _FakeFrontendTokenizer())
    monkeypatch.setattr(fcp_mod, "get_parser", lambda _name: _FakeParser())
    monkeypatch.setattr(hf_mod, "load_toolcall_anchor_id", lambda _tok, _opener: None)

    server_args = parse_args(["m", "--attention-backend", "sycl"])
    holder = launch_mod._build_engine_holder(server_args)
    holder()

    assert captured["config"].attention_backend == "sycl"
