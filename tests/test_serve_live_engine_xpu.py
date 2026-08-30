"""Live serve seam on the B70 (issue ``serve-live-engine``, #93).

Closes the gap between the three already-merged pieces — ``server-openai``
(#25, the HTTP surface), ``models-loader`` (#17, weights on XPU / host banks)
and ``engine-loop`` (#14, the real prefill/decode ``Engine``) — and the final
wiring step: the server's generation seam driving the *real* engine, not a
mock, so ``ft serve`` actually streams generated tokens instead of failing
loud at a stub.

XPU-only: skipped cleanly where there is no XPU. Runs in ``.venv-xpu`` on the
B70 (Intel Arc Pro B70, Battlemage). The model is the same tiny
Qwen3.5/3.6 that the offload suite proves correct (host-RAM expert banks +
XPU LRU), so this exercises the real 35B-shape architecture end to end.

Thread note: the offload MoE path does a device->host sync per token (the
host-side routing plan in the forward) and the XPU runtime faults on that sync
when it is issued from a non-main thread (FastAPI's ``TestClient`` runs the
route on an anyio portal thread — see the comment at
``qwen3_5_moe/__init__.py``). So these tests drive the real engine on the
*main* thread through the same ``stream_chat`` seam the route uses; that keeps
the XPU stream bound to the thread that built the engine. A real ``uvicorn``
serve is covered separately by the ``--smoke`` integration test.
"""
from __future__ import annotations

import json
import re
import pytest

import tests.test_qwen35_offload_xpu as _offload

from tests.test_qwen35_offload_xpu import XPU

# The offload suite defines `qwen35_xpu_serve_ckpt` (the serve-seam checkpoint:
# vocab 50257 = the real GPT-2 tokenizer the seam renders through) as a *module*-
# scoped fixture. This suite builds a fresh engine per test (each installs its own
# global context + offload cache), so a module-scoped checkpoint shared across
# tests would let two engines collide on the global ctx. Re-expose it at the default
# (function) scope so every test gets an isolated checkpoint dir.
qwen35_xpu_ckpt = pytest.fixture(_offload.qwen35_xpu_serve_ckpt.__wrapped__)

from freetoken.core import reset_global_ctx
from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.engine.engine import Engine
from freetoken.server.api_server import create_app
from freetoken.server.args import parse_args
from freetoken.server.generation import stream_chat
from freetoken.server.launch import _frontend_tokenizer

# The real message frontend (``#95``) renders the chat through the model's
# template, so the prompt is a handful of tokens (not the stub's single token).
# The budget stays tiny so the test is fast and the offload LRU is exercised
# without heavy churn; the KV pool below has headroom for the templated prompt.
_PROMPT_TOKENS = 32
_DECODE_TOKENS = 4
# KV pool sizing. Must be large enough that the identity-mapped page table
# (slots 0..max_seq_len-1) gathers stay in-bounds across the *whole* prompt +
# decode: the pool has ``num_pages`` rows and read_kv gathers
# page_table[table_idx, pos] with pos up to max_seq_len-1. The offload suite's
# proven-correct sizing is num_pages=64 / max_seq_len=32 (a 2x margin over the
# 5-token prompt + 4-token decode), so the serve test reuses it. A pool pinned
# to just the prompt token (the #93 spine's original 1-row pool) overruns the
# gather the moment the request decodes past position 0.
_NUM_PAGES = 64
_MAX_SEQ_LEN = 32

_MESSAGES = [{"role": "user", "content": "Count from one."}]


def qwen35_xpu_ckpt_vocab(ckpt_path: str) -> int:
    """The vocab size the serve path must keep generated ids inside."""
    config = json.load(open(f"{ckpt_path}/config.json"))
    return config["text_config"]["vocab_size"]


def _serve_engine(ckpt_path: str, dev):
    """The real engine the server seam drives: same as the offload suite, with
    the KV pool sized for the serve path's single prompt token."""
    import torch

    return Engine(
        EngineConfig(
            model_path=ckpt_path,
            tp_info=DistributedInfo(0, 1),
            dtype=torch.float32,
            device=dev,
            attention_backend="auto",
            moe_backend="offload",
            max_running_req=1,
            page_size=1,
            max_seq_len_override=_MAX_SEQ_LEN,
            num_page_override=_NUM_PAGES,
        )
    )


@XPU
def test_serve_chat_completions_streams_real_engine_tokens(qwen35_xpu_ckpt):
    """Accept (#93): the OpenAI seam streams real (non-503) tokens from the engine.

    A tiny Qwen3.5/3.6 is loaded host-offload through the real ``Engine`` and
    driven by the exact seam the ``/v1/chat/completions`` route uses
    (``stream_chat`` -> ``_stream_tokens`` -> ``engine.add_request`` +
    ``engine.generate``). The pre-fix seam called ``engine.generate(prompt,
    model=, max_tokens=)`` — a signature the real ``Engine`` does not have — so
    the route raised ``EngineNotReady`` and the HTTP layer answered 503. With
    the real message frontend (``#95``) the same seam now yields *readable* text:
    the chat is rendered through the model's template and the generated ids are
    decoded back to characters, not placeholders.

    The engine is built and driven on the *main* thread (see module docstring):
    the offload path's per-token D2H sync faults off-thread on the XPU, and
    ``TestClient`` would otherwise run this on a worker thread. The app is still
    built exactly as the route sees it, so the seam under test is the real one.
    """
    import torch

    dev = torch.device("xpu")
    engine = _serve_engine(qwen35_xpu_ckpt, dev)
    # Attach the real message frontend the way the live holder does, so the
    # seam under test resolves the tokenizer directly from the engine.
    engine.frontend_tokenizer = _frontend_tokenizer(parse_args([qwen35_xpu_ckpt]))
    server_args = parse_args([qwen35_xpu_ckpt])

    # The seam the /v1/chat/completions route calls, driven on the main thread.
    deltas = list(stream_chat(engine, _MESSAGES, model=server_args.resolved_model_name, max_tokens=_DECODE_TOKENS))
    content = "".join(content_delta for _reasoning, content_delta in deltas)

    # A real (non-503) token stream came back, now as readable text (the #95
    # tokenizer seam is live: no <tok-N> placeholders, no raw token-id runs).
    assert content, "the stream must be non-empty — the engine generated tokens"
    assert "<tok-" not in content, f"decoder still emits placeholders: {content!r}"
    # The decoded stream must be plain, printable text (the whole point of the
    # tokenizer seam landing: the client gets characters, not id placeholders).
    assert content.isprintable() or content.strip() == "", f"undecoded binary leaked into the stream: {content!r}"
    # And it must look like generated prose, not an id blob (no long digit runs).
    assert not re.search(r"\d{4,}", content), f"decoded stream looks like raw ids: {content!r}"


@XPU
def test_serve_app_built_against_real_engine_holder(qwen35_xpu_ckpt):
    """Accept (#93, wiring): create_app wires the real engine holder into the
    routes without error, so the live server's route graph is the real one."""
    import torch

    dev = torch.device("xpu")
    engine = _serve_engine(qwen35_xpu_ckpt, dev)
    # The app must build against the real holder that returns the real engine (the
    # launch.server layer's contract). A stub holder would build too, so assert
    # the app is the OpenAI app and its engine is the real Engine.
    app = create_app(parse_args([qwen35_xpu_ckpt]), lambda: engine)
    assert app.state.engine_holder is not None
    assert app.state.engine_holder() is engine
    routes = {getattr(r, "path", None) for r in app.routes}
    assert "/v1/chat/completions" in routes
    assert "/health" in routes


@XPU
def test_serve_503_only_when_engine_not_loaded():
    """Accept (#93, negative control): a genuinely not-loaded engine is a 503.

    The 503 path is preserved for the case the pre-fix code *falsely* hit — an
    engine with no generation method (e.g. the spine's pre-load holder). This
    pins that the fix did not delete the not-ready path, only the stub that
    raised it unconditionally. This test needs no XPU work, but keeps the XPU
    marker so it runs in the same XPU suite as the rest of the seam.
    """
    from freetoken._stub import NotYetImplemented

    from fastapi.testclient import TestClient

    def not_loaded_holder():
        raise NotYetImplemented("engine loop is a stub — implement under `engine-loop` (#14)")

    app = create_app(parse_args(["m"]), not_loaded_holder)
    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "x"}]})
    assert response.status_code == 503
    assert "detail" in response.json()
    # /v1/models stays live even while the engine is not loaded.
    assert client.get("/v1/models").status_code == 200
