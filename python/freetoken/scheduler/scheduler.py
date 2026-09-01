"""The scheduler: per-step batch selection for the Intel loop.

Upstream NVIDIA path: python/freetoken/scheduler/scheduler.py
Filled in: GitHub issue ``scheduler`` (see docs/architecture.md).

Upstream the scheduler is a multi-worker daemon: it talks to the model over
ZMQ, owns its own process group, and runs an ``while True`` receive/schedule/
forward loop. The Intel port is the *scheduling policy* of that daemon,
exposed as a plain object the single-process
:class:`~freetoken.engine.engine.Engine` drives:

* ``add(req)`` / ``add_all(reqs)`` -- admit a prompt into the pending queue
  (allocating a page-table row + a unique uid).
* ``abort(uid)`` -- drop a pending or in-flight request, freeing its row.
* ``schedule()`` -- pick this step's batch: a prefill batch (chunked to the
  token budget) if there is a pending prompt, else a decode batch.
* ``complete(batch)`` -- record that a batch just ran: move finished prefills
  into the decode set, refresh the decode set, and free rows that hit their
  stop condition.

The scheduler owns the request uid sequence and the page-table rows
(``table_idx``); the engine owns the token pool and does the actual
``model.forward`` + sampling, handing the scheduler a :class:`Batch` back via
:meth:`complete`. This split keeps the scheduler testable against a dummy
engine (the issue's acceptance criterion) and keeps torch out of the import
path (torch is only needed by the managers when they build a batch tensor).
"""
from __future__ import annotations

from typing import List, Optional

from freetoken.core import Batch, Req
from freetoken.scheduler.config import SchedulerConfig
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.prefill import PendingReq, PrefillManager, make_pending_req


class Scheduler:
    """Selects each step's prefill/decode batch for a single-process engine."""

    def __init__(
        self,
        config: SchedulerConfig,
        *,
        max_pages: int,
        cache_budget: int,
    ):
        self.config = config
        self.max_pages = max_pages
        self.cache_budget = cache_budget
        self._next_uid = 0
        # Free page-table rows (table_idx), kept sorted. The engine owns the
        # rows (it is the one that allocates them in the pool); the scheduler
        # borrows this list to hand requests a table_idx. add() pops the lowest
        # free index and complete() / abort() append freed ones back (sorted),
        # so admission is deterministic (lowest index first).
        self._free_slots: List[int] = sorted(range(config.max_running_req))
        self.decode_manager = DecodeManager(page_size=config.page_size)
        self.prefill_manager = PrefillManager(
            max_running_req=config.max_running_req,
            cache_budget=cache_budget,
            page_size=config.page_size,
            decode_manager=self.decode_manager,
        )

    # -- admission -----------------------------------------------------------

    def add(self, pending: PendingReq) -> int:
        """Admit a prompt: hand it a page-table row and queue it for prefill.

        ``pending.uid`` must already be set (the engine assigns it at
        ``add_request``); the scheduler only assigns the page-table row. Returns
        the uid (``pending.uid``). Raises :class:`RuntimeError` when the request
        cap is reached (no free page-table row) so the engine can reject the
        request cleanly. ``pending._table_idx`` is overwritten here.
        """
        if not self._free_slots:
            raise RuntimeError(
                f"request cap reached (max_running_req={self.config.max_running_req}); "
                "cannot admit a new request"
            )
        table_idx = self._free_slots.pop(0)
        pending._table_idx = table_idx  # noqa: SLF001
        pending._next_table_idx = -1  # noqa: SLF001 - assigned post-step
        self.prefill_manager.add_one_req(pending)
        # Keep the uid sequence past this admission so a later internal
        # allocation (none today) never re-issues a uid the engine already used.
        self._next_uid = max(self._next_uid, pending.uid + 1)
        return pending.uid

    # -- abort ---------------------------------------------------------------

    def abort(self, uid: int) -> bool:
        """Free a request's row and drop it from whichever set it is in.

        Returns ``True`` if the request was found and freed (pending, chunked,
        or decoding). The engine uses this to clear a row when a request is
        cancelled or hits a stop condition.
        """
        # Pending (not yet admitted, or a chunk awaiting continuation).
        pending = self.prefill_manager.pending_list
        for i, pr in enumerate(pending):
            if pr.uid == uid:
                # The row was allocated at add() (recorded on the PendingReq);
                # free it now. A prompt not yet split has no ChunkedReq, so the
                # table_idx comes from the pending entry, not from a chunk.
                table_idx = pr.chunked_req.table_idx if pr.chunked_req is not None else pr._table_idx
                self.prefill_manager.pending_list.pop(i)
                self._free_slots.append(table_idx)
                self._free_slots.sort()
                return True
        # In-flight decode.
        if (req := self.decode_manager.abort_req(uid)) is not None:
            self._free_slots.append(req.table_idx)
            self._free_slots.sort()
            return True
        return False

    # -- per-step ------------------------------------------------------------

    def schedule(self) -> Optional[Batch]:
        """Pick this step's batch: prefill if anything is queued, else decode.

        Mirrors the upstream policy -- a pending prompt always preempts a decode
        step (prefill-first), and a chunked prompt's continuation is always
        scheduled ahead of new prompts (PrefillManager keeps continuations at
        the front of the queue).

        The prefill batch is returned *as-is*; promoting its fully-prefilled
        requests into the decode set happens in :meth:`complete`, *after* the
        step runs. Promoting at schedule time (before the forward) would drop a
        just-prefilled request -- its first token has not been sampled yet, so ``device_len``
        still equals the prompt length, ``remain_len`` is 0 and ``can_decode`` is
        False -- and the prefill step's first token would be lost. Deferring the
        promotion to :meth:`complete` lets the step bump ``device_len`` first
        (``remain_len`` goes positive again), so the request is admitted with its
        first token intact. A partial (:class:`ChunkedReq`) prefill is never
        promoted: ``ChunkedReq.can_decode`` is False (see prefill.py), so
        :meth:`complete` keeps it out until a later step finishes the prompt.
        """
        batch = self.prefill_manager.schedule_next_batch(self.config.max_extend_tokens)
        if batch is None:
            batch = self.decode_manager.schedule_next_batch()
        if batch is None:
            # No pending prefill and no live decode: nothing to run this step.
            # The decode set is empty too (it would have scheduled), so the
            # loop is idle.
            return None
        return batch

    def complete(self, batch: Batch) -> None:
        """Record that ``batch`` just executed: update the bookkeeping.

        Called by the engine after ``model.forward`` + sampling + id append.
        Two things happen:

        * A request whose prompt was *fully* extended by this step (a plain
          :class:`Req` -- a whole prompt, or the final chunk of a split one) is
          admitted into the decode set here, *after* the step ran. The step has
          already sampled its first token and bumped ``device_len``, so
          ``remain_len > 0`` and ``can_decode`` is True. Promoting at
          :meth:`schedule` time instead would drop it and lose that first token
          (see :meth:`schedule`).
        * A request that hit its stop condition during this step (max_tokens /
          EOS / abort) is pruned from the decode set and its row freed. A
          *partial* (:class:`ChunkedReq`) prefill is neither admitted nor freed:
          it stays queued (PrefillManager re-queued its PendingReq) to be
          continued next step; ``ChunkedReq.can_decode`` is False so
          ``filter_reqs`` never puts it in the decode set, and it is not in
          ``running_before`` so it is not freed here.

        ``filter_reqs`` is idempotent (a request already present is moved to the
        end), so a decode request that survived the step is re-admitted (a no-op
        for its presence) while any just-finished ones are dropped.
        """
        running_before = self.decode_manager.running_reqs  # OrderedDict
        self.decode_manager.filter_reqs(batch.reqs)
        running_after = self.decode_manager.running_reqs  # OrderedDict
        # Free the rows of requests that were running before the step but are
        # no longer decodable after it. Both sides are OrderedDict keyed by Req
        # (identity hash), so the key-difference is well-defined. A fully
        # prefill request admitted *here* is not in running_before, so it is
        # never freed; a decode request that just hit its stop condition was in
        # running_before and is dropped from running_after, so its row is freed.
        freed = [req.table_idx for req in running_before.keys() - running_after.keys()]
        if freed:
            self._free_slots.extend(freed)
            self._free_slots.sort()

    # -- introspection --------------------------------------------------------

    @property
    def idle(self) -> bool:
        return not self.schedule()

    @property
    def next_table_idx_of(self, req: Req) -> int:
        """The next free row ``req`` will be moved to once it completes.

        Used by the engine to assign ``req._next_table_idx`` at admission /
        step boundaries so a completing request can be compacted in place.
        """
        return self._free_slots[0] if self._free_slots else -1
