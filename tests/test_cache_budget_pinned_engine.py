"""Engine-level integration test for the pinned-mode VRAM fit-check (issue #260).

``Engine.__init__`` sizes the pinned (non-auto) KV pool as
``max_running_req * max_seq_len`` and, until this issue, never checked that
against actual VRAM -- the auto-planning path already got this treatment in
#246 (``resolve_served_context_len``). This test builds a tiny CPU engine
(same dummy-weight fixture ``test_engine_loop.py`` uses) with
``moe_cache_auto`` left off (the pinned/conventional path) and a
``max_running_req * max_seq_len`` demand engineered to exceed a monkeypatched
"available VRAM", and asserts the engine caps the pool (and logs a loud
warning) instead of either OOMing at ``create_kv_pool`` or silently building
an under-sized pool with no signal.

No live XPU is needed: ``xpu_total_memory`` is monkeypatched directly (the
same technique ``tests/test_moe_cache_budget_xpu.py`` uses for the auto path),
so this runs in the CPU-only per-PR suite.
"""
from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch")

from freetoken.core import reset_global_ctx
from freetoken.engine.engine import Engine
from tests.test_engine_loop import DEVICE, _engine_config, _write_tiny_checkpoint


@pytest.fixture(autouse=True)
def _clean_global_ctx():
    yield
    reset_global_ctx()


def test_pinned_pool_caps_and_warns_when_demand_exceeds_vram(tmp_path, monkeypatch, caplog):
    model_path = _write_tiny_checkpoint(tmp_path)

    # A tiny fake VRAM budget: far too small to hold
    # max_running_req(2) * max_seq_len(32) KV pages for this tiny model's real
    # per-token byte cost, so the pinned path's fit check must kick in.
    monkeypatch.setattr(
        "freetoken.utils.arch.xpu_total_memory", lambda: 4096
    )

    config = _engine_config(model_path, device=DEVICE)
    assert config.moe_cache_auto is False  # this test is specifically the pinned path

    with caplog.at_level(logging.WARNING):
        engine = Engine(config)

    # Capped: the pool actually built is smaller than the naive
    # max_running_req * max_seq_len conventional formula would have been.
    assert engine.kv_cache.num_pages < config.max_running_req * 32
    # Loud and operator-actionable (mirrors #246's own reason string).
    assert any("issue #260" in r.message for r in caplog.records)


def test_pinned_pool_unchanged_when_vram_is_plentiful(tmp_path, monkeypatch, caplog):
    model_path = _write_tiny_checkpoint(tmp_path)

    # A generous fake VRAM budget: the conventional pool comfortably fits, so
    # this must be a complete no-op (existing, working deployments unaffected).
    monkeypatch.setattr(
        "freetoken.utils.arch.xpu_total_memory", lambda: 32 * 1024**3
    )

    config = _engine_config(model_path, device=DEVICE)

    with caplog.at_level(logging.WARNING):
        engine = Engine(config)

    assert engine.max_seq_len == 32
    assert not any("issue #260" in r.message for r in caplog.records)
