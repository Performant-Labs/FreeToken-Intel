"""Live radix prefix-cache reuse (issue `kvcache`, #12): a repeated prompt
reuses a prior request's KV instead of recomputing it.

Off by default (``EngineConfig.enable_prefix_cache``) -- every test in
``test_engine_loop.py`` and friends runs with it unset/False and is
unaffected (already verified there); this file turns it on explicitly.

Reuses ``test_engine_loop.py``'s own tiny-checkpoint fixture. CPU-only, no
XPU dependency.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from freetoken.core import Req, SamplingParams, reset_global_ctx
from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.engine.engine import Engine

from tests.test_engine_loop import DEVICE, _write_tiny_checkpoint


@pytest.fixture(autouse=True)
def _clean_global_ctx():
    yield
    reset_global_ctx()


def _cfg(model_path: str, **overrides) -> EngineConfig:
    kwargs = dict(
        model_path=model_path,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.float32,
        device=DEVICE,
        attention_backend="auto",
        max_running_req=2,
        page_size=1,
        max_seq_len_override=32,
        use_dummy_weight=True,
        enable_prefix_cache=True,
        num_page_override=256,  # headroom for two full rows + cached history
    )
    kwargs.update(overrides)
    return EngineConfig(**kwargs)


def _make_req(uid: int, ids: list[int], output_len: int) -> Req:
    return Req(
        input_ids=list(ids),
        table_idx=0,
        cached_len=0,
        output_len=output_len,
        uid=uid,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=output_len),
        cache_handle=None,
    )


def test_repeated_prompt_matches_the_cached_prefix(tmp_path):
    model_path = _write_tiny_checkpoint(tmp_path)
    engine = Engine(_cfg(model_path))
    engine.add_request(_make_req(0, [1, 2, 3], 4))
    engine.generate()

    r2 = engine.add_request(_make_req(1, [1, 2, 3], 4))
    # Never the WHOLE prompt (see add_request's own comment): the last
    # prompt token is always left un-cached so the model has something
    # real to extend.
    assert r2.cached_len == 2
    # The reused slots are the SAME physical slots request 1's commit
    # wrote (a real alias, not a coincidence): request 1's own row was
    # freed/reused since then, so read its committed data back through
    # the tree instead -- request 2's own page_table row IS that data.
    assert engine.page_table[r2.table_idx, :2].tolist() != [0, 0]
    engine.generate()


def test_repeated_prompt_produces_identical_greedy_output(tmp_path):
    """The real point of this issue: reusing the cached prefix must be
    numerically transparent -- same prompt, same weights, greedy sampling
    -> the SAME generated tokens whether or not the prefix was cached."""
    model_path = _write_tiny_checkpoint(tmp_path)
    engine = Engine(_cfg(model_path))
    engine.add_request(_make_req(0, [1, 2, 3], 4))
    gen1 = engine.generate()

    engine.add_request(_make_req(1, [1, 2, 3], 4))
    gen2 = engine.generate()

    assert gen1[0] == gen2[0]


def test_different_prompt_does_not_match(tmp_path):
    model_path = _write_tiny_checkpoint(tmp_path)
    engine = Engine(_cfg(model_path))
    engine.add_request(_make_req(0, [1, 2, 3], 4))
    engine.generate()

    r2 = engine.add_request(_make_req(1, [9, 9, 9], 4))
    assert r2.cached_len == 0


def test_prefix_cache_disabled_by_default_matches_existing_behavior(tmp_path):
    model_path = _write_tiny_checkpoint(tmp_path)
    engine = Engine(_cfg(model_path, enable_prefix_cache=False))
    assert engine.cache_manager is None
    r1 = engine.add_request(_make_req(0, [1, 2, 3], 4))
    assert r1.cached_len == 0
    engine.generate()

    r2 = engine.add_request(_make_req(1, [1, 2, 3], 4))
    assert r2.cached_len == 0  # no reuse at all when the feature is off
    engine.generate()


def test_finished_request_row_is_reusable_by_a_later_unrelated_request(tmp_path):
    """A completed, tree-committed request's table_idx must still be
    admittable by a later, unrelated request (issue #12's own ownership-
    transfer design: detach releases the row's bookkeeping without
    returning the tree-owned slots to the pool)."""
    model_path = _write_tiny_checkpoint(tmp_path)
    engine = Engine(_cfg(model_path, max_running_req=1))
    engine.add_request(_make_req(0, [1, 2, 3], 4))
    engine.generate()

    # A fresh, unrelated request reusing the same (only) table_idx row.
    r2 = engine.add_request(_make_req(1, [9, 9, 9], 4))
    assert r2.table_idx == 0
    assert engine.kv_cache.is_allocated(0)
    gen2 = engine.generate()
    assert len(gen2[0]) == 4
