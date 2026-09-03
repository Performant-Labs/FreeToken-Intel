"""Tool-call anchor detection + GDN ping-pong snapshot (issue
`semantic-cache-scheduler`, #171, part of the `semantic-cache` epic, #32).

Three independent pieces, each tested in isolation:
  1. ``toolcall_opener`` on the tool-call parser classes.
  2. Live anchor DETECTION in ``Engine.step()`` -- ``req.toolcall_anchor_len``
     is set exactly once, at the right decode position, when the engine
     watches for a configured ``toolcall_anchor_id``.
  3. ``CacheManager.snapshot_toolcall_anchor`` -- freezing a request's live
     GDN state into an idle ping-pong track slot, standalone (a synthetic
     ``LinearStatePool`` + ``Req``s, no real model).

Actually wiring a hybrid-GDN model's forward to read/write through
``LinearStatePool`` (so ``mamba_ping_pong``/``linear_slot_idx`` are ever set
by live serving) is issue #172's own scope -- (3) below exercises the method
directly against hand-built ``Req`` state, matching how #169/#170 tested
their own new structures before any model wiring existed.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from freetoken.core import Req, SamplingParams, reset_global_ctx
from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.engine.engine import Engine
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.scheduler.cache import CacheManager
from freetoken.server.function_call_parser import (
    FunctionCallParser,
    GenericTagParser,
    Llama3ToolParser,
    MistralToolParser,
    QwenToolParser,
    TOOL_CALL_PARSERS,
)

from tests.test_engine_loop import DEVICE, _engine_config, _write_tiny_checkpoint


# --- (1) toolcall_opener -----------------------------------------------------


def test_base_parser_has_no_opener():
    assert FunctionCallParser().toolcall_opener is None


def test_generic_tag_parser_opener_is_its_own_open_marker():
    parser = GenericTagParser(open_marker="<foo>", close_marker="</foo>")
    assert parser.toolcall_opener == "<foo>"


def test_qwen_and_llama3_and_mistral_openers_match_their_registered_markers():
    assert QwenToolParser().toolcall_opener == QwenToolParser().open_marker
    assert Llama3ToolParser().toolcall_opener == Llama3ToolParser().open_marker
    assert MistralToolParser().toolcall_opener == MistralToolParser().open_marker


def test_gpt_oss_parser_has_no_opener():
    # Harmony tool channels aren't a simple tag -- the base's None stands.
    assert TOOL_CALL_PARSERS["gpt_oss"].toolcall_opener is None


# --- (2) live anchor detection -----------------------------------------------


@pytest.fixture(autouse=True)
def _clean_global_ctx():
    yield
    reset_global_ctx()


def _add_prompt(engine: Engine, output_len: int) -> Req:
    req = Req(
        input_ids=[1, 2, 3],
        table_idx=0,
        cached_len=0,
        output_len=output_len,
        uid=0,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=output_len),
        cache_handle=None,
    )
    engine.add_request(req)
    return req


def _drive(engine: Engine, uid: int):
    """Run ``engine.step()`` to completion (mirrors ``Engine.generate``'s own
    loop), returning ``(generated_tokens, last_seen_req)``.

    ``Engine.add_request`` hands the raw ``Req`` to the scheduler, which
    wraps it in its own ``PendingReq``/``Req`` bookkeeping -- the object
    mutated step-by-step (``req.toolcall_anchor_len`` included) is NOT the
    caller's original object, so tests that need to observe it must read it
    back off ``ForwardOutput.reqs`` (exactly what ``generate()`` itself
    does to accumulate tokens), not off the object passed to
    ``add_request``.
    """
    from freetoken.engine.engine import ChunkedReq

    generated: list[int] = []
    last_req = None
    for _ in range(64):
        out = engine.step()
        if out.next_token_ids is None or len(out.next_token_ids) == 0:
            break
        for i, req in enumerate(out.reqs):
            if isinstance(req, ChunkedReq) or req.uid != uid:
                continue
            generated.append(int(out.next_token_ids[i]))
            last_req = req
    return generated, last_req


def test_anchor_id_none_by_default_is_a_silent_no_op(tmp_path):
    model_path = _write_tiny_checkpoint(tmp_path)
    engine = Engine(_engine_config(model_path, device=DEVICE))
    assert engine.toolcall_anchor_id is None
    _add_prompt(engine, output_len=3)
    _, last_req = _drive(engine, uid=0)
    assert last_req.toolcall_anchor_len is None


def test_anchor_len_is_set_once_at_the_first_occurrence(tmp_path):
    model_path = _write_tiny_checkpoint(tmp_path)

    # First pass: learn the (deterministic, greedy) generated sequence for
    # this dummy checkpoint -- see test_engine_greedy_is_deterministic for
    # why two builds of the same checkpoint are bit-identical.
    probe = Engine(_engine_config(model_path, device=DEVICE))
    _add_prompt(probe, output_len=4)
    generated, _ = _drive(probe, uid=0)
    reset_global_ctx()

    anchor_id = generated[0]
    prompt_len = 3  # see _add_prompt's fixed prompt
    first_occurrence = generated.index(anchor_id)
    expected_anchor_len = prompt_len + first_occurrence + 1

    engine = Engine(_engine_config(model_path, device=DEVICE))
    engine.toolcall_anchor_id = anchor_id
    _add_prompt(engine, output_len=4)
    out, last_req = _drive(engine, uid=0)

    assert out == generated  # same deterministic weights + prompt -> same tokens
    assert last_req.toolcall_anchor_len == expected_anchor_len

    # Once-only: a later occurrence of the same id (if any) must not move it.
    if generated.count(anchor_id) > 1:
        second_occurrence = generated.index(anchor_id, first_occurrence + 1)
        assert expected_anchor_len != prompt_len + second_occurrence + 1


# --- (3) CacheManager.snapshot_toolcall_anchor -------------------------------


def _pool(num_slots: int = 8) -> LinearStatePool:
    return LinearStatePool(
        num_layers=2,
        num_key_heads=4,
        num_value_heads=4,
        key_head_dim=8,
        value_head_dim=8,
        conv_kernel_dim=4,
        num_slots=num_slots,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def _cache_manager(page_size: int = 1) -> CacheManager:
    return CacheManager(torch.device("cpu"), page_size)


def _req_with_state(uid: int, *, anchor_len, ping_pong, last_track_seqlen=None) -> Req:
    return Req(
        input_ids=[0],
        table_idx=0,
        cached_len=0,
        output_len=1,
        uid=uid,
        sampling_params=SamplingParams(),
        cache_handle=None,
        linear_slot_idx=None,
        toolcall_anchor_len=anchor_len,
        mamba_ping_pong=ping_pong,
        mamba_last_track_seqlen=last_track_seqlen,
    )


def test_snapshot_copies_live_slot_into_the_next_ping_pong_track_slot():
    pool = _pool()
    live, track0, track1 = pool.alloc(3)
    pool.conv_state(0, live)[:] = 5.0
    pool.recurrent_state(0, live)[:] = 7.0

    req = _req_with_state(0, anchor_len=4, ping_pong=(track0, track1))
    req.linear_slot_idx = live

    mgr = _cache_manager(page_size=1)
    mgr.snapshot_toolcall_anchor([req], pool)

    torch.testing.assert_close(pool.conv_state(0, track0), pool.conv_state(0, live))
    torch.testing.assert_close(pool.recurrent_state(0, track0), pool.recurrent_state(0, live))
    assert req.mamba_last_track_seqlen == 4
    assert req.mamba_next_track_idx == 1  # flipped, ready for the next snapshot


def test_snapshot_is_a_real_copy_surviving_later_overwrites_of_the_live_slot():
    pool = _pool()
    live, track0, track1 = pool.alloc(3)
    pool.conv_state(0, live)[:] = 1.0

    req = _req_with_state(0, anchor_len=2, ping_pong=(track0, track1))
    req.linear_slot_idx = live
    _cache_manager(page_size=1).snapshot_toolcall_anchor([req], pool)

    pool.conv_state(0, live)[:] = 99.0
    assert pool.conv_state(0, track0).max().item() == 1.0


def test_snapshot_skips_a_request_with_no_anchor_yet():
    pool = _pool()
    live, track0, track1 = pool.alloc(3)
    req = _req_with_state(0, anchor_len=None, ping_pong=(track0, track1))
    req.linear_slot_idx = live
    _cache_manager(page_size=1).snapshot_toolcall_anchor([req], pool)
    assert req.mamba_last_track_seqlen is None


def test_snapshot_skips_a_request_not_opted_into_ping_pong_tracking():
    pool = _pool()
    live = pool.alloc(1)[0]
    req = _req_with_state(0, anchor_len=4, ping_pong=None)
    req.linear_slot_idx = live
    _cache_manager(page_size=1).snapshot_toolcall_anchor([req], pool)
    assert req.mamba_last_track_seqlen is None


def test_snapshot_is_idempotent_once_already_tracked():
    pool = _pool()
    live, track0, track1 = pool.alloc(3)
    req = _req_with_state(0, anchor_len=4, ping_pong=(track0, track1), last_track_seqlen=4)
    req.linear_slot_idx = live
    mgr = _cache_manager(page_size=1)
    mgr.snapshot_toolcall_anchor([req], pool)
    # No flip happened -- the guard skipped this request outright.
    assert req.mamba_next_track_idx == 0


def test_snapshot_skips_an_anchor_not_page_aligned():
    pool = _pool()
    live, track0, track1 = pool.alloc(3)
    req = _req_with_state(0, anchor_len=5, ping_pong=(track0, track1))
    req.linear_slot_idx = live
    _cache_manager(page_size=4).snapshot_toolcall_anchor([req], pool)  # 5 % 4 != 0
    assert req.mamba_last_track_seqlen is None
