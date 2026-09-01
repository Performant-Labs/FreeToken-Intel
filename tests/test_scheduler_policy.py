"""Tests for the scheduler (issue ``scheduler``, #13).

Two layers, matching the issue's acceptance criteria:

* **Policy tests** (this file, ``test_scheduler_policy.py``) exercise the
  scheduler against a *dummy engine* -- no torch, no model -- so they run in
  the CPU venv (the dual-venv contract: ``import freetoken.scheduler`` never
  pulls in torch). They pin the scheduling *decisions*: chunked prefill respects
  the per-step token budget, prefill preempts decode, decode batches are
  uid-ordered and pruned on completion, and abort frees a slot.
* **Integration tests** (``test_scheduler_integration.py``, torch-marked) run
  the scheduler wired into the real :class:`~freetoken.engine.engine.Engine`
  over a tiny dummy-weight model, on the CPU, to prove the engine's
  ``add_request`` / ``step`` / ``generate`` hand off to the scheduler correctly.
"""
from __future__ import annotations

from freetoken.core import Batch, Req, SamplingParams
from freetoken.scheduler import (
    ChunkedReq,
    DecodeManager,
    PendingReq,
    PrefillAdder,
    PrefillManager,
    Scheduler,
    SchedulerConfig,
    make_pending_req,
)


def _config(**overrides) -> SchedulerConfig:
    # EngineConfig requires model_path / tp_info / dtype; the policy tests never
    # run a model, so dummies stand in. Only the scheduling fields matter here.
    base = dict(
        model_path="/dummy",
        tp_info=None,
        dtype=None,
        max_running_req=4,
        page_size=1,
        max_extend_tokens=8192,
    )
    base.update(overrides)
    return SchedulerConfig(**base)


def _pending(prompt_len: int, output_len: int, uid: int, table_idx: int) -> PendingReq:
    pr = PendingReq(
        uid=uid,
        input_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=output_len),
    )
    # The Scheduler normally assigns these at add(); the unit tests set them
    # directly so PrefillAdder can read them without a full Scheduler.
    pr._table_idx = table_idx  # noqa: SLF001
    pr._next_table_idx = -1  # noqa: SLF001
    return pr


def _scheduler(*, max_extend_tokens: int = 8, max_running_req: int = 4) -> Scheduler:
    config = _config(max_extend_tokens=max_extend_tokens, max_running_req=max_running_req)
    # A generous pool budget so the request cap (not the pool) is the binding
    # constraint in the policy tests; the cap is max_running_req.
    return Scheduler(config, max_pages=256, cache_budget=4096)


# -- chunked prefill ---------------------------------------------------------


def test_chunked_prefill_splits_to_budget():
    """A 16-token prompt with an 8-token budget splits across two prefill steps.

    Step 1 extends [0,8) (a ChunkedReq, can_decode False). Step 2 extends
    [8,16) (a full Req, can_decode True) and the prompt leaves the pending queue
    -- the budget is respected on every step (8 tokens each, never 16).
    """
    sched = _scheduler(max_extend_tokens=8)
    sched.add(_to_pending(_full_req(prompt_len=16, output_len=2)))
    # One pending prompt, nothing running yet.
    assert len(sched.prefill_manager.pending_list) == 1

    batch1 = sched.schedule()
    assert batch1 is not None
    assert batch1.phase == "prefill"
    (req1,) = batch1.reqs
    # First chunk: 8 tokens, not finished, so it is a ChunkedReq.
    assert isinstance(req1, ChunkedReq)
    assert req1.extend_len == 8  # exactly the budget
    assert req1.can_decode is False
    # The prompt is not done: it stays in the pending queue (1 continuation).
    assert len(sched.prefill_manager.pending_list) == 1

    # Capture step 1's extension up front: the lines below mutate req1's
    # cached_len/device_len to simulate the engine having run step 1, which
    # would otherwise zero out req1.extend_len (device_len - cached_len) by
    # step 113. The per-step extensions are what the budget is measured against.
    step1_extend = req1.extend_len

    # The engine "ran" step 1: extend [0,8) -> device_len 8.
    req1.cached_len = 8
    req1.device_len = 8
    sched.complete(batch1)
    # A ChunkedReq is not yet decodable, so it is NOT in the decode set.
    assert len(sched.decode_manager.running_reqs) == 0

    batch2 = sched.schedule()
    assert batch2 is not None
    assert batch2.phase == "prefill"
    (req2,) = batch2.reqs
    # Second chunk extends the remaining 8 tokens and completes the prompt.
    assert isinstance(req2, Req)
    assert not isinstance(req2, ChunkedReq)
    assert req2.extend_len == 8  # exactly the budget (16 - 8 remaining)
    # The prompt left the pending queue (fully prefilled).
    assert len(sched.prefill_manager.pending_list) == 0
    # The prompt is NOT yet in the decode set: promotion happens in complete()
    # (after the step runs), not in schedule(). can_decode is already True
    # (prompt done, 2 output tokens remain), but the decode set only sees it
    # once the step finishes.
    assert req2.can_decode is True
    assert len(sched.decode_manager.running_reqs) == 0
    # Budget respected across both steps: 8 + 8 = 16 = the whole prompt.
    assert step1_extend + req2.extend_len == 16

    # Simulate the engine running step 2: extend [8,16), sample first token.
    req2.device_len = 16
    req2.input_ids = req2.input_ids + [100]  # first generated token
    sched.complete(batch2)
    # Now the prompt is in the decode set (promoted in complete()).
    assert len(sched.decode_manager.running_reqs) == 1


def test_short_prompt_not_chunked():
    """A prompt that fits the budget is prefilled whole in one step."""
    sched = _scheduler(max_extend_tokens=8)
    sched.add(_to_pending(_full_req(prompt_len=4, output_len=2)))
    batch = sched.schedule()
    (req,) = batch.reqs
    assert isinstance(req, Req) and not isinstance(req, ChunkedReq)
    assert req.extend_len == 4
    req.device_len = 4
    sched.complete(batch)
    assert len(sched.decode_manager.running_reqs) == 1
    assert len(sched.prefill_manager.pending_list) == 0


def test_budget_is_per_step_not_per_request():
    """Two 6-token prompts with an 8-token budget: step 1 takes one fully, the
    other is deferred (8 < 6+6) and runs alone in step 2 -- the budget is a
    per-step limit, not a per-request one."""
    sched = _scheduler(max_extend_tokens=8)
    sched.add(_to_pending(_full_req(prompt_len=6, output_len=1, uid=0, table_idx=0)))
    sched.add(_to_pending(_full_req(prompt_len=6, output_len=1, uid=1, table_idx=1)))
    batch1 = sched.schedule()
    # Only the first prompt fits (6 <= 8); the second would need 6 more (8 < 12).
    assert len(batch1.reqs) == 1
    assert batch1.reqs[0].uid == 0
    (batch1.reqs[0]).device_len = 6
    sched.complete(batch1)
    # Step 2 now runs the deferred second prompt.
    batch2 = sched.schedule()
    assert batch2 is not None
    assert len(batch2.reqs) == 1
    assert batch2.reqs[0].uid == 1


# -- prefill preempts decode -------------------------------------------------


def test_prefill_preempts_decode():
    """With a prompt pending and a request decoding, the next step prefills."""
    sched = _scheduler(max_extend_tokens=8)
    decoding = _full_req(prompt_len=2, output_len=10)
    # Put `decoding` directly in the decode set (it already finished its prompt).
    sched.decode_manager.running_reqs.update({decoding: None})
    # A new prompt is admitted.
    sched.add(_to_pending(_full_req(prompt_len=3, output_len=2, uid=50, table_idx=2)))
    batch = sched.schedule()
    # Prefill wins over decode even though a request is already decoding.
    assert batch.phase == "prefill"
    assert batch.reqs[0].uid == 50


# -- decode batching ---------------------------------------------------------


def test_decode_batch_uid_ordered():
    """The decode batch runs the live requests in uid order."""
    sched = _scheduler()
    for uid, table_idx in [(7, 0), (3, 1), (11, 2)]:
        decoding = _full_req(prompt_len=1, output_len=5, uid=uid, table_idx=table_idx)
        decoding.device_len = 1
        sched.decode_manager.running_reqs.update({decoding: None})
    batch = sched.schedule()
    assert batch.phase == "decode"
    assert [req.uid for req in batch.reqs] == [3, 7, 11]


def test_decode_prunes_completed():
    """A decode request that hits max_tokens is pruned and its slot freed."""
    sched = _scheduler()
    done = _full_req(prompt_len=1, output_len=1, uid=1, table_idx=0)
    alive = _full_req(prompt_len=1, output_len=5, uid=2, table_idx=1)
    done.device_len = 1
    alive.device_len = 1
    sched.decode_manager.running_reqs.update({done: None, alive: None})
    free_before = sorted(sched._free_slots)
    # Step 1: `done` emits its 1st (final) token -> device_len 2 == max, aborts.
    batch = sched.schedule()
    assert batch.phase == "decode" and len(batch.reqs) == 2
    for req in batch.reqs:
        req.device_len += 1  # one decode token each
        if req.device_len >= req.max_device_len:
            req.aborted = True
    sched.complete(batch)
    # `done` is pruned (device_len == max_device_len -> not can_decode), `alive`
    # remains. The freed slot is back in the free list.
    assert {req.uid for req in sched.decode_manager.running_reqs} == {2}
    assert sorted(sched._free_slots) == sorted(free_before + [0])


def test_idle_when_nothing_pending_or_decoding():
    sched = _scheduler()
    assert sched.schedule() is None
    assert sched.idle is True


# -- abort -------------------------------------------------------------------


def test_abort_pending_frees_slot():
    sched = _scheduler()
    req = _full_req(prompt_len=4, output_len=2, uid=9, table_idx=0)
    sched.add(_to_pending(req))
    assert 0 in sched._free_slots or len(sched._free_slots) < 4
    assert sched.abort(9) is True
    # The slot is released.
    assert 0 in sched._free_slots
    # Nothing left to schedule.
    assert sched.schedule() is None


def test_abort_decoding_frees_slot():
    sched = _scheduler()
    decoding = _full_req(prompt_len=1, output_len=5, uid=4, table_idx=2)
    decoding.device_len = 1
    sched.decode_manager.running_reqs.update({decoding: None})
    assert sched.abort(4) is True
    assert 2 in sched._free_slots
    assert len(sched.decode_manager.running_reqs) == 0


def _full_req(prompt_len: int, output_len: int, uid: int = 0, table_idx: int = 0) -> Req:
    # The caller (server path) sets uid; the scheduler assigns the page-table
    # row at add(). table_idx is a placeholder here. sampling_params is required
    # -- the scheduler wraps the Req in a PendingReq at admission.
    return Req(
        input_ids=list(range(prompt_len)),
        table_idx=table_idx,
        cached_len=0,
        output_len=output_len,
        uid=uid,
        sampling_params=SamplingParams(max_tokens=output_len),
        cache_handle=None,
    )


def _to_pending(req: Req) -> PendingReq:
    """Wrap a raw Req in a PendingReq, as the engine's add_request does.

    The scheduler's add() takes a PendingReq (which carries the caller's uid);
    this mirrors the engine's wrap so the policy tests exercise the same path
    the integration tests will.
    """
    return make_pending_req(req.uid, req.input_ids, req.sampling_params, req.cache_handle)
