"""Main engine loop: prefill, decode, sampling.

Upstream NVIDIA path: python/freetoken/engine/engine.py
Fill in: GitHub issue `engine-loop` (see docs/architecture.md).

This is the minimal *functional* engine the B70 port runs. It wires the pieces
that are already real -- the model's forward, the paged KV pool, the reference
attention backend, and the sampler -- into a prefill/decode loop:

    Engine(config)
        .add_request(Req)      -> admit into the scheduler (assigns a slot)
        .generate()             -> prefill the prompt (chunked to the token
                                   budget), then decode until every request
                                   hits its stop condition (eos or max_tokens)
                                   and return the generated ids

Since issue ``scheduler`` the engine no longer picks the batch by hand. Each
:meth:`step` asks the :class:`~freetoken.scheduler.Scheduler` which batch to run
(a prefill batch, chunked to the per-step token budget, or a decode batch), runs
that one batch through the model + sampler, and then hands the result back to
the scheduler via :meth:`~freetoken.scheduler.Scheduler.complete` so it can move
finished prefills into the decode set and free the rows of requests that hit
their stop condition. The engine keeps ownership of the token pool (it is the
one that actually materializes and frees ``table_idx`` rows) and of the device
tensors; the scheduler owns the scheduling decision (which requests, which
phase, which chunk) and the uid / free-slot bookkeeping.

Each step still builds the batch's device tensors (``input_ids`` /
``positions`` / ``out_loc``) from the request state, sets them on the global
context, runs ``model(...)`` once, and samples a next token per request.
Prefill and decode differ only in how the tensors are assembled (a request's
whole extend during prefill; one new token during decode), which the model's
forward already branches on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import List

import torch

from freetoken.attention import create_attention_backend
from freetoken.core import Batch, Context, Req, set_global_ctx
from freetoken.engine.config import EngineConfig
from freetoken.kvcache import create_kv_pool
from freetoken.models.loader import load_model
from freetoken.scheduler import Scheduler, SchedulerConfig, make_pending_req
from freetoken.scheduler.prefill import ChunkedReq
from freetoken.utils.arch import is_xpu_available


@dataclass
class ForwardOutput:
    """Per-request results of one engine step."""

    # Next-token id per request, in batch order (int64 [bs]).
    next_token_ids: torch.Tensor
    # Which requests finished this step (hit eos / max_tokens).
    finished: List[Req]
    # The batch this step ran (reqs in batch order, aligned to next_token_ids).
    reqs: List[Req]


class Engine:
    """A minimal functional inference engine (issue `engine-loop`, #14).

    ``config`` is an :class:`~freetoken.engine.config.EngineConfig`. On
    construction the engine builds the model (loading weights from the
    checkpoint, or fabricating dummy expert banks when ``use_dummy_weight`` is
    set), creates the paged KV pool + page table + attention backend + sampler,
    and installs them on the global context so the model's ``forward`` can read
    them. It also builds the :class:`~freetoken.scheduler.Scheduler`, which owns
    per-step batch selection; the engine drives it from :meth:`add_request` /
    :meth:`step` / :meth:`generate`.
    """

    def __init__(self, config) -> None:
        self.config = config
        model_config = config.model_config

        # Where the dense weights / KV pool / attention run. An explicit
        # ``config.device`` wins (the tests pin CPU); otherwise default to the
        # XPU when present, else CPU.
        if getattr(config, "device", None) is not None:
            device = (
                config.device
                if isinstance(config.device, torch.device)
                else torch.device(config.device)
            )
        else:
            device = torch.device("xpu") if is_xpu_available() else torch.device("cpu")
        self.device = device

        # Build the model and place its weights (dense on `device`; MoE experts
        # on host offload banks). use_dummy_weight fabricates the expert banks offline so
        # the engine is testable without a checkpoint on disk.
        dtype = config.dtype if isinstance(config.dtype, torch.dtype) else None
        self.model, _expert_sources = load_model(
            config.model_path,
            device,
            dtype=dtype,
            dummy=bool(getattr(config, "use_dummy_weight", False)),
            moe_backend=getattr(config, "moe_backend", None),
            moe_cpu_layers=getattr(config, "moe_cpu_layers", None),
        )

        # Size the paged KV pool. The pool is indexed by token slot
        # (page_size==1 in the reference path), one row per (request, position).
        # The override is a *floor*, not a cap: it raises the pool for small
        # test models (which otherwise get a 1-row pool for a 1-row page table
        # and overrun the gather on decode) but never shrinks a large model's
        # ``max_running_req * max_seq_len`` pool below what its context needs.
        max_seq_len = config.max_seq_len
        default_num_pages = config.max_running_req * max_seq_len
        num_pages = (
            max(config.num_page_override, default_num_pages) if config.num_page_override else default_num_pages
        )
        self.page_size = config.page_size
        self.max_seq_len = max_seq_len
        self.max_running_req = config.max_running_req
        self.kv_cache = create_kv_pool(
            model_config,
            page_size=self.page_size,
            num_pages=num_pages,
            device=device,
            dtype=dtype or torch.bfloat16,
        )

        # Page table: [max_running_req+1, max_seq_len] identity slot map
        # (slot `pos` holds the token at position `pos`); +1 row so table_idx
        # 0..max_running_req are all valid.
        self.page_table = torch.zeros(
            (self.max_running_req + 1, max_seq_len), dtype=torch.int64, device=device
        )
        for r in range(self.page_table.shape[0]):
            self.page_table[r, :max_seq_len] = torch.arange(max_seq_len, device=device)
        self.kv_cache.attach_page_table(self.page_table)

        # Attention backend (reference pure-torch GQA under "auto").
        self.attn_backend = create_attention_backend(config.attention_backend, config)

        # Sampler. The reference path has no single eos id to stop on, so we
        # stop purely on max_tokens (eos_token_id=-1 never matches); a request
        # can still opt out via ignore_eos.
        self.sampler = self._build_sampler(model_config, device)

        # Global context the model's forward reads. The model also resolves its
        # own reference (ctx.model) so the MoE blocks can reach the offload
        # cache / layer map without the engine reaching into the model.
        self.ctx = Context(page_size=self.page_size)
        self.ctx.model = self.model
        self.ctx.kv_cache = self.kv_cache
        self.ctx.attn_backend = self.attn_backend
        self.ctx.page_table = self.page_table
        # ADR 0002: when the MoE experts are host-offloaded, the model's forward
        # serves them through the LRU slot pool the loader attached.
        if getattr(self.model, "moe_offload", False) and getattr(self.model, "moe_cache", None) is not None:
            self.ctx.moe_offload_cache = self.model.moe_cache
        set_global_ctx(self.ctx)

        # Scheduler: owns uid assignment, page-slot (table_idx) bookkeeping, and
        # per-step batch selection (chunked prefill + decode batching). It is
        # built over the same pool the engine owns: max_pages rows (the page
        # table) and the pool's token budget. A plain EngineConfig is wrapped in
        # a SchedulerConfig view (a SchedulerConfig is itself an EngineConfig, so
        # passing one through works unchanged).
        self._pool_num_pages = num_pages
        self._pool_budget = num_pages * self.page_size
        if not isinstance(config, SchedulerConfig):
            # Build the SchedulerConfig view from the *declared* EngineConfig
            # fields only. config.__dict__ also carries the resolved
            # cached_property instances (hf_config, model_config) once they have
            # been accessed above, and those are not dataclass fields -- splatting
            # them would raise TypeError and would defeat the lazy (torch-gated)
            # load. Constructing the subclass directly (not dataclasses.replace,
            # which re-instantiates type(config) and would drop the subclass
            # fields like max_extend_tokens) is what actually yields a
            # SchedulerConfig.
            field_values = {f.name: getattr(config, f.name) for f in fields(EngineConfig)}
            config = SchedulerConfig(**field_values)
        self.scheduler = Scheduler(config, max_pages=num_pages, cache_budget=self._pool_budget)

    def _build_sampler(self, model_config, device) -> "object":
        from freetoken.engine.sample import Sampler

        # No eos id in the reference path; -1 is outside any real vocab.
        return Sampler(eos_token_id=-1, device=device)

    # -- request admission ---------------------------------------------------

    def add_request(self, req: Req) -> Req:
        """Admit a request: hand the scheduler a page slot and queue it.

        The caller sets ``req.uid`` (the server path assigns it at the API
        boundary); the scheduler assigns the page-table row (freeing it when the
        request later completes) and returns the uid. The raw
        :class:`~freetoken.core.Req` is wrapped in the scheduler's
        :class:`~freetoken.scheduler.prefill.PendingReq` -- the caller's uid is
        preserved -- and the assigned row is stored back on ``req``. Raises
        :class:`RuntimeError` when the request cap is reached so a caller can
        reject the request cleanly.
        """
        pending = make_pending_req(req.uid, req.input_ids, req.sampling_params, req.cache_handle)
        uid = self.scheduler.add(pending)
        req.uid = uid
        req.table_idx = pending._table_idx  # noqa: SLF001
        return req

    def abort_request(self, uid: int) -> bool:
        """Free a request's page slot and drop it from the scheduler (any phase)."""
        return self.scheduler.abort(uid)

    def _free_slot(self, table_idx: int) -> None:
        # Restore the identity slot map for a freed row (no-op on the reference
        # pool, which re-derives the slot from the position; kept for symmetry
        # with a paged pool that may need to clear a row on release).
        pass

    # -- the loop -------------------------------------------------------------

    def step(self) -> ForwardOutput:
        """Run one engine step (one model forward + one sample) over the batch
        the :class:`~freetoken.scheduler.Scheduler` selected.

        The scheduler picks the batch (a chunked-prefill batch if a prompt is
        waiting, else a decode batch). Each request in the batch extends by its
        own count: a whole-prompt or chunk extend during prefill, one new token
        during decode. After the forward + sample the result is handed back to
        the scheduler (:meth:`~freetoken.scheduler.Scheduler.complete`) so it can
        promote finished prefills into the decode set and free completed rows.

        Returns a :class:`ForwardOutput` with no batch (empty tensor) when the
        scheduler has nothing to run this step (no pending prefill, no live
        decode) -- the :meth:`generate` loop treats that as "idle" and stops.
        """
        batch = self.scheduler.schedule()
        if batch is None:
            return ForwardOutput(
                next_token_ids=torch.empty((0,), device=self.device), finished=[], reqs=[]
            )

        input_ids: List[int] = []
        positions: List[int] = []
        out_locs: List[int] = []
        extend_lens: List[int] = []
        # The scheduler emits a uniform per-step phase (a prefill batch or a
        # decode batch, never a mix -- see Scheduler.schedule), so batch.phase is
        # the authoritative phase for every request in this batch. The old
        # per-request shape test (device_len < len(input_ids)) misfired for a
        # FRESH prefill: at prefill time the scheduler has already bumped
        # device_len up to the prompt length, so device_len == len(input_ids) and
        # the strict "<" is False -- every whole-prompt (and final-chunk) prefill
        # was misrouted into the decode branch and ran as a 1-token step,
        # truncating the prompt's attention context.
        is_prefill_batch = batch.phase == "prefill"
        for req in batch.reqs:
            if is_prefill_batch:
                # Prefill: extend by this request's own extend_len (the whole
                # prompt for a non-chunked one, or the chunk's slice for a
                # continuation), reading the not-yet-extended prompt slice
                # [cached_len, cached_len + ext).
                ext = req.extend_len
                start = req.cached_len
                ids = req.input_ids
                for t in range(ext):
                    pos = start + t
                    input_ids.append(int(ids[pos]))
                    positions.append(pos)
                    out_locs.append(pos)
            else:  # decode: one new token per request
                ext = 1
                # The token being positioned is the one just sampled, whose
                # absolute position is ``device_len - 1``: the prefill step that
                # ran already did ``device_len += 1`` to include it, so
                # ``device_len`` is the history length (position + 1) and the
                # new token sits at ``device_len - 1`` (prompt len 5 -> decode
                # positions 5,6,7,...).
                #
                # Do NOT use ``len(req.input_ids) - 1``: the decode branch of the
                # post-step bookkeeping *replaces* ``input_ids`` with just the
                # new token (``[tok]``) rather than appending, so from the second
                # decode step on ``len(input_ids)`` is 1 and that formula pins
                # every decode position to 0 -- leaving pool slots unwritten and
                # exploding attention. ``device_len`` is chunk-size-independent
                # (the prefill's single ``+= 1`` is the same whether the prompt
                # was one 5-token chunk or five 1-token chunks), so this is what
                # keeps chunked and non-chunked runs bit-identical.
                pos = req.device_len - 1
                last = req.input_ids
                input_ids.append(int(last[-1]))
                positions.append(pos)
                out_locs.append(pos)
            extend_lens.append(ext)

        device = self.device
        batch.input_ids = torch.tensor(input_ids, dtype=torch.int64, device=device)
        batch.positions = torch.tensor(positions, dtype=torch.int64, device=device)
        batch.out_loc = torch.tensor(out_locs, dtype=torch.int64, device=device)
        # Per-request new-token counts (extend_len in prefill, 1 in decode) in
        # request order. The model's forward slices the token tensors by these
        # instead of a single batch-level phase flag, so a step that mixes
        # prefill and decode requests slices each request's tokens by its own count.
        batch.extend_lens = torch.tensor(extend_lens, dtype=torch.int64, device=device)

        with self.ctx.forward_batch(batch):
            self.attn_backend.prepare_metadata(batch)
            logits = self.model(batch.input_ids, batch.positions, batch.out_loc)

        from freetoken.engine.sample import BatchSamplingArgs

        sampling_args = BatchSamplingArgs([req.sampling_params for req in batch.reqs])
        next_ids = self.sampler.sample(logits, sampling_args)

        finished: List[Req] = []
        for i, req in enumerate(batch.reqs):
            if isinstance(req, ChunkedReq):
                # An intermediate prefill chunk is not sampled: only the chunk
                # that fully extends the prompt (a plain Req) emits the first
                # generated token. The chunk is not appended a token and is not
                # marked finished here; it stays queued (the scheduler re-queues
                # its PendingReq) and is continued next step. Mirrors upstream
                # _process_last_data, which skips ChunkedReq entirely.
                continue
            tok = int(next_ids[i])
            # This step's token is the one that may trip the stop condition, so
            # append it *before* marking the request finished (see generate()).
            # The batch phase is uniform, so the append rule is uniform too.
            req.input_ids = [tok] if batch.phase == "decode" else list(req.input_ids) + [tok]
            req.device_len += 1
            # The FINAL chunk of a chunked prefill -- the one that just
            # completed the prompt -- is a plain Req (promoted out of the
            # ChunkedReq chain) whose cached_len is > 0 (it resumed from a
            # prior ChunkedReq via the adder). A full, non-chunked prefill is
            # also a plain Req in a prefill batch, but its cached_len is 0, so
            # the "cached_len > 0" test is exactly "this is the final chunk of a
            # chunked prompt". Snap its cached prefix to the prompt boundary so
            # the post-prefill state is coherent: the attention backend keys its
            # phase off batch.phase and reads device_len as the attended length;
            # with cached_len == prompt_len the decode step's extend_len
            # (device_len - cached_len) is 1 (just the sampled token) and
            # positions [cached_len, cached_len+1) align with that token. A full
            # prefill must NOT be snapped (cached_len 0 -> the decode step reads
            # the whole history via device_len). Intermediate chunks never reach
            # here (step() skips ChunkedReq sampling above), so their
            # cached_len keeps tracking raw prefix progress.
            if batch.phase == "prefill" and not isinstance(req, ChunkedReq) and req.cached_len > 0:
                req.cached_len = len(req.input_ids) - 1
            sp = req.sampling_params
            done = tok == self.sampler.eos_token_id
            if not sp.ignore_eos and done:
                req.aborted = True
            if req.device_len >= req.max_device_len:
                req.aborted = True
            if req.aborted:
                finished.append(req)

        # Record the step in the scheduler: promote finished prefills into the
        # decode set, refresh the decode set, and free the rows of requests that
        # completed (hit max_tokens / eos).
        self.scheduler.complete(batch)
        return ForwardOutput(next_token_ids=next_ids, finished=finished, reqs=list(batch.reqs))

    def generate(self, max_steps: int | None = None) -> List[List[int]]:
        """Run admitted requests to completion; return generated ids per request.

        Each :meth:`step` runs the batch the scheduler selected (a chunked
        prefill, then decodes) until the scheduler is idle -- no prompt waiting
        and no request still decoding -- or ``max_steps`` is reached. Returns a
        list, aligned to the admission order, of the *generated* token ids
        (prompt tokens excluded).

        With no admitted requests this is a no-op returning an empty list (the
        spine's ``Engine.generate(None)`` wiring probe relies on that).
        """
        if self.scheduler is None or not self.scheduler.prefill_manager.pending_list:
            # No admitted requests: the wiring probe (launch.py) calls
            # generate() on a freshly-constructed engine and expects [].
            return []
        # Each admitted request needs ceil(prompt_len / max_extend_tokens)
        # prefill steps plus one decode step per output token. Use that as the
        # default step budget when the caller does not pin one.
        budget = max_steps if max_steps is not None else sum(
            (
                math.ceil(len(pr.input_ids) / max(1, self.scheduler.config.max_extend_tokens))
                + pr.output_len
                for pr in self.scheduler.prefill_manager.pending_list
            )
        ) + len(self.scheduler.decode_manager.running_reqs)
        # Snapshot the admitted uids in admission order *before* the loop: as the
        # loop runs, each request leaves pending_list the moment its prompt is
        # fully prefilled (it is promoted into the decode set, then freed on
        # completion), so pending_list is empty by the time the loop ends.
        # Rebuilding the result from the live list would then drop every
        # completed request. The admission order is the contract the server relies
        # on to map results back to API calls.
        admitted_uids = [pr.uid for pr in self.scheduler.prefill_manager.pending_list]
        # ``tokens[uid]`` accumulates a request's generated tokens, in emission
        # order. Each step samples exactly one token per request in the batch
        # (the last-position logit). We capture it for every request that
        # participates in the step -- not just the one that trips the stop
        # condition -- because every step produces a real generated token.
        # Keys are uids (unique, scheduler-assigned) because batch order changes
        # across steps (prefill batches are uid-sorted, continuations first).
        tokens: dict[int, list[int]] = {}
        for pr in self.scheduler.prefill_manager.pending_list:
            tokens[pr.uid] = []
        steps = 0
        while steps < budget:
            step = self.step()
            if step.next_token_ids is None or len(step.next_token_ids) == 0:
                break  # scheduler idle: nothing pending, nothing decoding
            steps += 1
            # next_token_ids is aligned to batch.reqs order (one per request).
            # Record each request's token in its per-request accumulator. An
            # intermediate prefill chunk is skipped (step() does not append it a
            # token), so only the chunk that fully extends the prompt contributes
            # a generated token here -- the same final-chunk-only rule as above.
            for i, req in enumerate(step.reqs):
                if isinstance(req, ChunkedReq):
                    continue
                tokens.setdefault(req.uid, []).append(int(step.next_token_ids[i]))
        return [tokens[uid] for uid in admitted_uids]

    def rebuild_cache(self) -> None:
        # No radix / hybrid cache in the reference engine; nothing to rebuild.
        pass
