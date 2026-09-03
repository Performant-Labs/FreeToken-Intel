"""Chunked-prefill admission.

Upstream NVIDIA path: python/freetoken/scheduler/prefill.py
Filled in: GitHub issue ``scheduler`` (see docs/architecture.md).

The prefill manager holds the queue of not-yet-extended requests and, each
step, decides how much of each prompt can be extended *within the per-step
token budget* (``max_extend_tokens``). A prompt that fits in the budget is
prefilled whole and moves on to decode; a prompt that does not is *chunked* --
only ``min(budget, remaining)`` tokens are extended this step and the request
stays in the queue to continue next step. This is the core acceptance criterion
for the scheduler: a long prompt is split across steps and the budget is
respected on every step.

The upstream file is coupled to the radix / SWA cache manager (prefix caching,
the ``kvcache`` issue) and to CUDA streams / pinned-memory copies. None of that
exists in the Intel loop yet, so the two currency gates upstream adds -- the
prefix-cache ``match_req`` and the sliding-window pool reservation -- collapse
to a single gate: the number of admitted requests must fit ``max_running_req``
(page slots). The budget / chunk / continuation logic, which is the actual
scheduling policy being ported, is carried over verbatim.

This module is torch-free at import time. :class:`PendingReq` holds the
prompt's token ids, and the upstream stores those as a ``torch.Tensor``; the
Intel engine stores prompt ids as a plain Python ``list[int]`` (see
:class:`~freetoken.core.Req`), so ``PendingReq`` stores a list too.
:mod:`torch` is imported only inside :meth:`PrefillManager.schedule_next_batch`
to build the batch's id tensor -- keeping ``import freetoken.scheduler` (and
the CPU venv) clean, per the dual-venv contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

from freetoken.core import Batch, Req

if TYPE_CHECKING:
    from freetoken.core import SamplingParams


@dataclass(eq=False)
class PendingReq:
    """A queued prompt awaiting prefill.

    Mirrors the upstream ``PendingReq``: the raw prompt ids, the sampling
    settings, and -- when the prompt was chunked -- the partial
    :class:`ChunkedReq` that tracks how far the chunked prefill has progressed.
    ``chunked_req`` is ``None`` for a prompt that has not yet been split.

    ``eq=False`` (identity equality + identity hash), matching :class:`Req`:
    a queued prompt is an identity (its uid), not a value. With field-wise
    equality, a chunked continuation and its original entry (identical
    prompt/sampling fields) hash *and* compare equal, so the ``set`` bookkeeping
    in :class:`DecodeManager.filter_reqs` would collapse them and the scheduler
    would free a slot that is still in use. Identity hashing keeps the two
    distinct.
    """

    uid: int
    input_ids: List[int]
    sampling_params: "SamplingParams"
    chunked_req: "ChunkedReq | None" = None
    # How much of input_ids is already cached (issue `kvcache`, #12): a
    # page-aligned prefix length from CacheManager.match, or 0 for a cache
    # miss / prefix caching disabled. Only the [cached_len, input_len) tail
    # counts against the per-step token budget and pool footprint -- the
    # prefix's KV is already resident, reused via the matched slot indices
    # on ``_cache_handle``, not recomputed.
    cached_len: int = 0
    # Assigned by Scheduler.add when the request is queued: the page-table row
    # (table_idx) and the next free row to hand the request once it finishes
    # (next_table_idx). ``_``-prefixed so they stay out of the dataclass's
    # public surface; the adder reads them at admission time. ``cache_handle``
    # is the prefix-cache ticket (kvcache issue's scope) -- the matched
    # RadixCacheHandle when cached_len > 0, else None.
    _table_idx: int = 0
    _next_table_idx: int = 0
    _cache_handle: object = None

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens


def make_pending_req(
    uid: int,
    input_ids: List[int],
    sampling_params: "SamplingParams",
    cache_handle=None,
    cached_len: int = 0,
) -> PendingReq:
    """Build a :class:`PendingReq` from raw request fields.

    The engine's ``add_request`` takes a :class:`~freetoken.core.Req` (the
    server path's shape) and wraps it this way before handing the
    :class:`~freetoken.scheduler.Scheduler` a :class:`PendingReq`. It lives here
    (not in the engine) so the wrap stays in the torch-free scheduler package --
    the engine imports it on the XPU path but the CPU-venv policy tests never do.
    ``cached_len`` (issue #12) is the page-aligned prefix length the engine's
    own ``CacheManager.match`` already found before calling this.
    """
    return PendingReq(
        uid=uid,
        input_ids=list(input_ids),
        sampling_params=sampling_params,
        _cache_handle=cache_handle,
        cached_len=cached_len,
    )


class ChunkedReq(Req):
    """A prompt's in-progress chunk: prefilling but not yet sampling.

    A chunk is a prefill with a known "not finished" status, so it must never be
    added to the decode set (that would hand it a next token before its prompt
    is fully extended). Overriding :attr:`Req.can_decode` with ``False`` keeps
    the decode manager's ``filter_reqs`` from admitting it, exactly as upstream.

    Sampling is enforced at the type level the same way: :meth:`append_host`
    raises, so a chunk can never be appended a generated token. The engine's
    step loop therefore samples (and appends) only the *final* chunk -- the one
    that fully extends the prompt and is admitted as a plain :class:`Req` -- and
    skips every intermediate chunk.
    """

    @property
    def can_decode(self) -> bool:
        return False  # avoid being added to the decode manager

    def append_host(self, next_token) -> None:
        raise NotImplementedError("ChunkedReq should not be sampled")


@dataclass
class PrefillAdder:
    """Admits requests into one prefill batch within the token budget.

    ``token_budget`` is the per-step extend budget (``max_extend_tokens``) and
    is decremented by each admitted chunk; ``reserved_size`` is the token count
    the decode manager is already holding so a new admission cannot push the
    step past the pool the decode set is depending on (upstream folds this in
    from ``decode_manager.inflight_tokens``). ``step_budget`` snapshots the
    step's starting budget so the admission loop can tell "the first prompt in
    this step" (``token_budget == step_budget``) from "a later prompt" -- the
    former may use the whole budget (and chunk if longer), the latter must fit
    the remainder whole.
    """

    token_budget: int
    step_budget: int
    reserved_size: int
    max_running_req: int
    cache_budget: int
    running_count: int

    # -- admission -----------------------------------------------------------

    def _try_allocate_one(self, req: PendingReq):
        # Request-cap gate: a live request needs a page-table row (table_idx).
        # running_count counts running + admitted-this-pass, so this is the
        # only currency the Intel pool exposes (the upstream's radix / SWA
        # currency gates are the kvcache issue's scope).
        if self.running_count >= self.max_running_req:
            return None
        # KV-pool gate: this request's REMAINING extend (prompt tail past any
        # cache-matched prefix, plus requested output) must fit the pool
        # budget, with headroom for the in-flight decode set. cached_len
        # (issue #12 -- CacheManager.match, 0 when there is no match / prefix
        # caching is disabled) is already resident and reused, not
        # recomputed, so it costs no budget here.
        extend_len = req.input_len - req.cached_len
        estimated_len = extend_len + req.output_len
        if self.reserved_size + estimated_len > self.cache_budget:
            return None
        # Reserve the request's footprint + a running slot up front so the
        # per-step admission is an atomic decide-then-commit: a later gate
        # (the all-or-nothing budget check in _add_one_req) can defer the
        # request and _undo_allocation rolls these back. This mirrors upstream,
        # whose try_add_one unlocks / frees the row when a continuation or a
        # not-yet-admitted request is turned down after passing the pool gates.
        self.reserved_size += extend_len + req.output_len
        self.running_count += 1
        return req.cached_len

    def _undo_allocation(self, req: PendingReq) -> None:
        """Roll back the reservation :meth:`_try_allocate_one` made.

        The all-or-nothing step-budget check in :meth:`_add_one_req` can defer a
        fresh prompt that already passed the pool gates; without this undo the
        deferred prompt would keep its running slot and pool reservation for the
        rest of the pass (and every later pass, since the PendingReq stays
        queued), starving later prompts and inflating the in-flight estimate the
        decode manager is sized against. The table row itself is untouched: it
        is owned by the scheduler (reserved at add()) and the PendingReq stays
        queued with it, so a deferred prompt neither leaks a row nor needs one.
        """
        self.reserved_size -= (req.input_len - req.cached_len) + req.output_len
        self.running_count -= 1

    # -- sizing + construction ------------------------------------------------

    def _add_one_req(
        self,
        pending_req: PendingReq,
        table_idx: int,
        cached_len: int,
        next_table_idx: int,
        cache_handle,
        chunked: bool = False,
    ) -> Req | None:
        remain_len = pending_req.input_len - cached_len
        if chunked:
            # A continuation resumes an in-flight chunk: extend the next
            # budget-sized slice of the remaining prompt (a prompt longer than
            # the budget is split across steps). The token budget is a per-step
            # limit, so only the *new* tokens this slice extends are charged.
            chunk_size = min(self.token_budget, remain_len)
            # No room for even one token this pass (budget spent, or nothing
            # left of the prompt): leave the request in the queue for the next step.
            if chunk_size <= 0:
                return None
        else:
            # A fresh prompt is admitted only if its *whole* prompt fits the
            # remaining step budget. The per-step budget is an all-or-nothing
            # admission gate for a not-yet-started request (a prompt that does
            # not fully fit is deferred to a step with room for it, rather than
            # being split behind another prompt in the same step); only an
            # already-started continuation is chunked (see ``chunked=True``).
            # The caller checks ``token_budget`` before calling, so
            # remain_len <= token_budget is the precise "fits whole" test.
            if remain_len > self.token_budget:
                return None
            chunk_size = remain_len
        # For a continuation, chunk_size < remain_len means the prompt is split:
        # this step extends only [cached_len, cached_len + chunk_size) and the
        # remainder continues next step. A fresh prompt is never split (see the
        # chunked=False branch: it is admitted whole or deferred), so this is a
        # no-op there. The class swap (Req vs ChunkedReq) is what stops a partial
        # prompt from being sampled.
        is_chunked = chunk_size < remain_len
        CLS = ChunkedReq if is_chunked else Req
        # The per-step token budget is spent only by the *new* tokens this
        # chunk extends (chunk_size), not by the already-prefilled prefix
        # (cached_len) -- a continuation must not re-pay for prefix tokens.
        # (The pool footprint + running slot were already reserved by
        # _try_allocate_one on the fresh path; a continuation skips both.)
        self.token_budget -= chunk_size
        # input_ids carries the prompt prefix extended so far: [0, chunk_size)
        # for a first admit, or [0, cached_len + chunk_size) for a continuation
        # (the prior chunk's ids plus the new slice). device_len then becomes
        # cached_len + chunk_size, so extend_len (device_len - cached_len) is
        # exactly this step's new chunk, not the whole row.
        input_ids = pending_req.input_ids[: cached_len + chunk_size]
        req = CLS(
            input_ids=input_ids,
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
        )
        req._next_table_idx = next_table_idx  # noqa: SLF001 - engine fills after forward
        return req

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        if self.token_budget <= 0:
            return None
        # A continuation resumes the in-flight chunk: it keeps the original
        # table_idx and cached_len (the already-extended prefix) and extends the
        # next budget-sized slice. The table slot was already allocated when the
        # first chunk was admitted, so only the budget / running-count are spent.
        if chunked_req := pending_req.chunked_req:
            table_idx = chunked_req.table_idx
            # A continuation resumes the in-flight chunk. device_len is the
            # field that reflects how far the prompt has been extended: the
            # engine extends the chunk this step (device_len = cached_len +
            # chunk_size) and, for a ChunkedReq, does NOT append a sampled token
            # (final-chunk-only -- the chunk's append_host raises and step()
            # skips it), so device_len is the pure extended prefix (0,1,2,3).
            # Using it as the resume offset keeps remain_len = input_len -
            # cached_len shrinking (5,4,3,2,1) so the final chunk completes the
            # prompt, and -- because _add_one_req builds the new slice as
            # input_ids[:cached_len + chunk_size] -- each continuation reads the
            # RIGHT slice (positions [prefix, prefix+chunk)), not the first
            # chunk again. (cached_len is NOT used here: for an intermediate
            # chunk it is the slice's START, not its end, so it would re-read
            # chunk 0 forever. Only the final chunk's cached_len is snapped to
            # the prompt boundary, and that one is a plain Req, not a
            # ChunkedReq, so it never reaches this branch.)
            cached_len = chunked_req.device_len
            next_table_idx = chunked_req._next_table_idx  # noqa: SLF001
            cache_handle = chunked_req.cache_handle
            req = self._add_one_req(
                pending_req, table_idx, cached_len, next_table_idx, cache_handle, chunked=True
            )
            # A continuation must not be dropped for lack of budget (it already
            # occupies a slot): the caller re-queues it and retries next step.
            return req
        resource = self._try_allocate_one(pending_req)
        if resource is None:
            return None
        # _try_allocate_one returns the request's cached_len (issue #12 --
        # 0 for a cache miss / prefix caching disabled); the table index is
        # assigned by the scheduler at admission time (it owns the free list).
        # The adder only decides *whether* to admit and *how much* to extend.
        table_idx = pending_req._table_idx  # noqa: SLF001 - set by Scheduler.add
        next_table_idx = pending_req._next_table_idx  # noqa: SLF001
        cache_handle = pending_req._cache_handle  # noqa: SLF001
        # The per-step budget is an all-or-nothing gate *between* prompts: the
        # first prompt admitted into an empty step may use the whole budget (and,
        # if it is longer than that, is chunked and continues next step); any
        # prompt admitted *after* one is admitted only if it fits the *remaining*
        # budget whole (otherwise it defers to a step with room, rather than
        # splitting behind another prompt in the same step). token_budget == the
        # step's budget iff nothing has been admitted this step, so "first in"
        # is exactly token_budget == the step's max_extend_tokens.
        first_in_step = self.token_budget == self.step_budget
        # A fresh prompt admitted as the *first* prompt in the step may use the
        # whole budget and, if longer than it, be chunked (the original
        # budget-splitting policy). Any prompt admitted *after* one must fit the
        # remaining budget whole or defer (chunked=False -> whole-or-defer).
        # _try_allocate_one above already reserved this prompt's pool footprint
        # + running slot, so a deferral must roll those back (undo) or the
        # deferred prompt would keep them for the rest of the pass and every
        # later pass. The table row is unaffected (scheduler-owned, pending
        # stays queued with it).
        req = self._add_one_req(
            pending_req, table_idx, resource, next_table_idx, cache_handle, chunked=first_in_step
        )
        if req is None:
            self._undo_allocation(pending_req)
        return req


@dataclass
class PrefillManager:
    """The pending-queue plus the per-step prefill admission.

    ``schedule_next_batch`` walks the queue (continuations first, then fresh
    requests in admission order) and admits as many as the budget and the pool
    allow, mirroring upstream. A request that fits whole is removed from the
    queue (it moves to decode after its step); a chunked request stays in the
    queue with its :class:`ChunkedReq` so the next step continues it.
    """

    max_running_req: int
    cache_budget: int
    page_size: int
    decode_manager: object
    pending_list: List[PendingReq] = field(default_factory=list)

    def add_one_req(self, pending_req: PendingReq) -> None:
        self.pending_list.append(pending_req)

    def schedule_next_batch(self, prefill_budget: int):
        # The policy itself is torch-free: it decides *which* requests to run and how many
        # tokens each extends, and the batch carries plain-Python id lists. The
        # engine (which *does* have torch) turns those lists into the model's
        # device tensors in step(). This keeps `import freetoken.scheduler` and
        # the policy testable on the torch-free CPU venv (dual-venv contract).
        if len(self.pending_list) == 0:
            return None

        # Estimated offset due to in-flight decode (the decode set is reserving
        # pool capacity for its remaining tokens).
        adder = PrefillAdder(
            token_budget=prefill_budget,
            step_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,
            max_running_req=self.max_running_req,
            cache_budget=self.cache_budget,
            running_count=len(self.decode_manager.running_reqs),
        )
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []
        admitted_pending: List[PendingReq] = []
        prompt_admissions: list[tuple[int, int, int]] = []
        log_new_tokens = 0
        for pending_req in self.pending_list:
            is_continuation = pending_req.chunked_req is not None
            if req := adder.try_add_one(pending_req):
                # The admitted chunk is this step's row. If the prompt is not
                # finished the PendingReq keeps the ChunkedReq so the next step
                # resumes it (and it stays in the queue); if it is finished the
                # PendingReq is dropped from the queue (removed below).
                pending_req.chunked_req = req if isinstance(req, ChunkedReq) else None
                if isinstance(req, ChunkedReq):
                    chunked_list.append(pending_req)
                reqs.append(req)
                admitted_pending.append(pending_req)
                if not is_continuation:
                    # First chunk: record the prompt size + prefix hit (0 today)
                    # once at admission, as upstream. Continuations contribute 0
                    # to the cache-hit stat (the prefix was already counted), as upstream.
                    prompt_admissions.append((req.uid, pending_req.input_len, 0))
                log_new_tokens += req.extend_len
            else:
                if is_continuation:
                    # A continuation already owns a row and must not be starved
                    # by a not-fitting fresh prompt ahead of it: stop the pass.
                    break
                # A fresh prompt was not admitted because its *whole* prompt does
                # not fit the remaining step budget (the all-or-nothing policy).
                # That does NOT mean the budget is exhausted -- a *shorter* prompt
                # later in the queue can still fit -- so skip it (it stays queued
                # and its row is untouched) and keep admitting later requests.
                continue
        if len(reqs) == 0:
            return None
        # Rebuild the queue by *identity*: every admitted PendingReq is removed
        # from the tail (it is either a chunked continuation -- re-inserted at
        # the front via chunked_list -- or a fully-prefilled prompt -- dropped
        # entirely, having moved to the decode set). The un-admitted tail keeps
        # its original order. Rebuilding from identity (rather than a positional
        # slice) is what makes this correct when the admitted set is not a clean
        # prefix of pending_list: a continuation re-uses its PendingReq in place
        # (same object in pending_list and chunked_list), so a slice would
        # double-count it.
        admitted_ids = {id(p) for p in admitted_pending}
        unadmitted = [p for p in self.pending_list if id(p) not in admitted_ids]
        self.pending_list = chunked_list + unadmitted
        batch = Batch(reqs=reqs, phase="prefill")
        batch.log_new_tokens = log_new_tokens
        batch.log_cached_tokens = 0
        batch.prompt_admissions = prompt_admissions
        return batch

    def abort_req(self, uid: int) -> Req | None:
        for i, req in enumerate(self.pending_list):
            if req.uid == uid:
                self.pending_list.pop(i)
                return req.chunked_req  # None for a not-yet-chunked prompt
        return None

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
