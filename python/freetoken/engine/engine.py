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
from freetoken.scheduler.cache import CacheManager
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

        dtype = config.dtype if isinstance(config.dtype, torch.dtype) else None

        # Issue #16 (elastic-memory): before loading, decide how the device's
        # VRAM is split between the MoE expert slot cache and the paged KV pool.
        # In auto mode the split is planned from the total VRAM
        # (memory_ratio budget, MoE-priority / KV-floor, fit assert); otherwise
        # the operator pinned moe_cache_size and the KV pool keeps its
        # conventional size. None = "use the loader's / conventional default".
        self._cache_budget = self._plan_cache_budget(model_config, dtype)
        planned_cache_size, planned_num_pages = self._cache_budget

        # Build the model and place its weights (dense on `device`; MoE experts
        # on host offload banks). use_dummy_weight fabricates the expert banks offline so
        # the engine is testable without a checkpoint on disk. The planned slot
        # count is threaded in so the loader sizes the MoE cache off the VRAM
        # budget instead of its conventional layer-count formula.
        self.model, _expert_sources = load_model(
            config.model_path,
            device,
            dtype=dtype,
            dummy=bool(getattr(config, "use_dummy_weight", False)),
            moe_backend=getattr(config, "moe_backend", None),
            moe_cpu_layers=getattr(config, "moe_cpu_layers", None),
            moe_cache_size=planned_cache_size,
        )

        # Issue #9 (moe-hybrid): the operator's per-step PCIe-fetch cap (the
        # --moe-hybrid-max-fetch serve flag, EngineConfig.moe_hybrid_max_fetch).
        # -1 (default) = fully profile-driven via the fetch fraction; a
        # non-negative int caps the routed experts fetched to the XPU each decode
        # step (the rest compute on the host CPU). Stashed on the model after
        # load so the block's _forward_hybrid reads it (it is a no-op on the
        # in-VRAM / cpu / offload backends, which never call _forward_hybrid).
        self.model.moe_hybrid_max_fetch = int(getattr(config, "moe_hybrid_max_fetch", -1) or -1)

        # Size the paged KV pool. The pool is indexed by token slot
        # (page_size==1 in the reference path), one row per (request, position).
        # The override is a *floor*, not a cap: it raises the pool for small
        # test models (which otherwise get a 1-row pool for a 1-row page table
        # and overrun the gather on decode) but never shrinks a large model's
        # ``max_running_req * max_seq_len`` pool below what its context needs.
        max_seq_len = config.max_seq_len
        default_num_pages = config.max_running_req * max_seq_len
        # When the budget planner sized the KV pool (auto mode), use the planned
        # count (it is at least the KV floor and reflects the real free VRAM).
        # Otherwise keep the conventional override-vs-default resolution.
        if planned_num_pages is not None:
            num_pages = planned_num_pages
        elif config.num_page_override:
            num_pages = max(config.num_page_override, default_num_pages)
        else:
            num_pages = default_num_pages
        # +1 page of slack: MHAKVCache (issue #173) reserves slot 0 as a
        # dummy/padding slot, so the pool's real allocatable capacity is one
        # page short of `num_pages * page_size` slots -- without this, the
        # LAST request admitted up to max_running_req would fail to allocate
        # its full max_seq_len-sized row (confirmed directly: a 2-running-req
        # pool sized to exactly fit 2 full rows raised "KV pool full" on the
        # second admission).
        num_pages += 1
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

        # Page table: [max_running_req+1, max_seq_len] slot map, +1 row so
        # table_idx 0..max_running_req are all valid (the scheduler only ever
        # hands out 0..max_running_req-1 -- see Scheduler._free_slots -- so
        # row max_running_req is a padding row, never allocated). Rows start
        # all-zero (pointing at MHAKVCache's reserved slot 0) and are filled
        # with a REAL, disjoint slot run per request at admission time
        # (add_request), not a shared identity map -- see issue
        # `engine-kv-addressing` (#173): every row pointing at the SAME slot
        # range let two concurrently-decoding requests silently corrupt each
        # other's KV.
        self.page_table = torch.zeros(
            (self.max_running_req + 1, max_seq_len), dtype=torch.int64, device=device
        )
        self.kv_cache.attach_page_table(self.page_table)

        # Radix prefix-cache reuse (issue `kvcache`, #12) -- off by default
        # (EngineConfig.enable_prefix_cache). When on, add_request matches a
        # new prompt against what's already cached (CacheManager.match) and
        # a finished request's full sequence is committed into the tree
        # (see step()) so a later request can reuse it instead of
        # recomputing. _full_ids accumulates each live request's full token
        # history (table_idx -> ids) across decode steps -- req.input_ids
        # itself is replaced with just the latest token during decode (see
        # step()'s own comment on that), so this is the only place the full
        # sequence a finished request should commit is available.
        self.cache_manager = CacheManager(device, self.page_size) if config.enable_prefix_cache else None
        self._full_ids: dict[int, list[int]] = {}
        # table_idx -> the (clamped, page-aligned) cached_len this row was
        # ADMITTED with -- req.cached_len itself gets mutated mid-lifetime
        # (snapped to the prompt boundary once the first prefill step
        # completes, see step()'s own comment on that), so this is the only
        # stable record of "how much of this row was an aliased reuse of
        # ANOTHER node's slots, never this row's own MHAKVCache allocation
        # at all" by the time step() needs it at completion.
        self._admitted_cached_len: dict[int, int] = {}

        # Tool-call anchor id (issue `semantic-cache-scheduler`, #171): the
        # single token id of this model's tool-call-opener grammar marker,
        # or None (default -- feature off). Engine itself never touches
        # tokenizers/tool-call parsers (that's the server layer's job, see
        # server/args.py + server/function_call_parser.py); this is set
        # post-construction by the server path (server/launch.py's
        # _build_engine_holder), mirroring the existing
        # ``engine.frontend_tokenizer = ...`` attach pattern there. When
        # set, step()'s decode loop watches for it and records
        # req.toolcall_anchor_len the first time it appears -- the deepest
        # reuse boundary that survives a client-side rewrite of the echoed
        # tool call. Actually FREEZING the GDN state at that boundary
        # (CacheManager.snapshot_toolcall_anchor) needs the model wired to
        # a real ping-pong-capable LinearStatePool, which qwen3_5_moe does
        # not do yet (it still uses its own, older, non-ping-pong
        # _LinearStatePool) -- that wiring is issue #172's own scope, so for
        # now, only ever gets read by tests and by whatever future PR does
        # that wiring, not by a live snapshot call yet.
        self.toolcall_anchor_id: int | None = None

        # Attention backend (reference pure-torch GQA under "auto").
        self.attn_backend = create_attention_backend(config.attention_backend, config)

        # Sampler.
        self.sampler = self._build_sampler(config, model_config, device)

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

    def _plan_cache_budget(self, model_config, dtype):
        """Decide the MoE cache / KV split (issue #16, elastic-memory).

        Returns a ``(moe_cache_size, kv_num_pages)`` pair consumed by the
        loader (MoE slot pool) and the engine (KV pool) respectively.

        * ``moe_cache_auto`` on  -> plan from the device's total VRAM
          (memory_ratio budget, MoE-priority / KV-floor, fit assert). ``moe_cache_rate``
          (when set) caps the MoE share to that fraction of the budget.
        * ``moe_cache_auto`` off  -> a pinned ``moe_cache_size`` (or the loader's
          conventional size when that is also 0) and the conventional KV size.

        Both entries are ``None`` to mean "fall back to the conventional
        formula" so a dense (non-MoE) model or a missing XPU never changes the
        pool the reference path has always used.
        """
        pinned = int(getattr(self.config, "moe_cache_size", 0) or 0)
        if not getattr(self.config, "moe_cache_auto", False):
            # Operator pinned the size (or nothing): no VRAM planning.
            return (pinned if pinned else None, None)

        # Auto: plan off the real VRAM. Need a live device with a total.
        from freetoken.utils.arch import xpu_total_memory

        total_vram = xpu_total_memory()
        if total_vram is None:
            # No XPU (CPU test box): nothing to plan against; keep conventional.
            return (None, None)

        num_experts = int(getattr(model_config, "num_experts", 0) or 0)
        # num_moe_layers may be unset in the HF config; derive it the way the
        # loader does (num_layers minus the dense prefix) so a config that only
        # carries num_layers still plans.
        num_moe_layers = int(
            getattr(model_config, "num_moe_layers", None)
            or (
                int(getattr(model_config, "num_layers", 0) or 0)
                - int(getattr(model_config, "first_k_dense_replace", 0) or 0)
            )
            or 0
        )
        if not num_experts or not num_moe_layers:
            # Not a MoE model (or config missing the fields): no MoE cache to plan.
            return (None, None)

        from freetoken.engine.cache_budget import plan_cache_budget

        dtype_bytes = getattr(dtype, "itemsize", 2) or 2
        num_layers = int(getattr(model_config, "num_layers", 0) or 0)
        num_kv_heads = int(getattr(model_config, "num_key_value_heads", 0) or 0)
        # Mirror the KV pool's derivation: an explicit head_dim, else hidden_size
        # // num_attention_heads (the TINY / Qwen3-MoE shape never sets it).
        head_dim = int(
            getattr(model_config, "head_dim", None)
            or (
                int(getattr(model_config, "hidden_size", 0) or 0)
                // max(1, int(getattr(model_config, "num_attention_heads", 0) or 0))
            )
            or 0
        )
        moe_intermediate = int(getattr(model_config, "moe_intermediate_size", 0) or 0)
        hidden_size = int(getattr(model_config, "hidden_size", 0) or 0)
        kv_reserve = int(getattr(self.config, "kv_reserve_tokens", 8192) or 8192)
        memory_ratio = float(getattr(self.config, "memory_ratio", 0.9) or 0.9)

        # An operator ``moe_cache_rate`` (when set) caps the MoE share to that
        # fraction of the budget (the KV pool keeps its reserve floor and
        # absorbs the rest); otherwise the MoE cache is the pure priority.
        rate = getattr(self.config, "moe_cache_rate", None)
        cache = plan_cache_budget(
            total_vram_bytes=total_vram,
            memory_ratio=memory_ratio,
            kv_reserve_tokens=kv_reserve,
            num_experts=num_experts,
            moe_intermediate_size=moe_intermediate,
            hidden_size=hidden_size,
            num_moe_layers=num_moe_layers,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype_bytes=dtype_bytes,
            moe_fraction=rate if rate else None,
        )
        return (cache.moe_cache_size, cache.kv_num_pages)

    def _build_sampler(self, config, model_config, device) -> "object":
        from freetoken.engine.sample import Sampler

        # The reference path previously never stopped a request early: the
        # sampler was hardcoded to eos_token_id=-1 (outside any real vocab),
        # so every request ran to the full max_tokens budget regardless of the
        # model's own stop signal. That silently forces decoding well past a
        # response's natural end -- exactly where small / merged models tend
        # to collapse into repetition loops, which looked like a decode-quality
        # problem but was actually the engine ignoring EOS entirely. Read the
        # checkpoint's real eos_token_id (top-level, or nested under
        # ``text_config`` for the multimodal Qwen3.5/3.6-MoE config shape) and
        # fall back to -1 (unreachable, i.e. the old always-run-to-budget
        # behavior) only when the checkpoint truly doesn't declare one.
        hf_config = config.hf_config
        eos_token_id = getattr(hf_config, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(getattr(hf_config, "text_config", None), "eos_token_id", None)
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0] if eos_token_id else None
        return Sampler(eos_token_id=eos_token_id if eos_token_id is not None else -1, device=device)

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

        When prefix caching is on (issue `kvcache`, #12), the prompt is
        matched against the cache BEFORE admission: a matched prefix is
        locked (protected from eviction while this request depends on it)
        and its length flows into the scheduler so ``PrefillAdder`` only
        extends -- and only budgets -- the un-cached tail. Also allocates
        this row's own disjoint KV-pool slot run (issue `engine-kv-
        addressing`, #173) -- a chunked prompt's later continuations reuse
        this SAME table_idx/allocation for the rest of the request's
        lifetime, so one allocation here covers the whole request;
        :meth:`_free_slot` releases it when the request finishes (or
        :meth:`step` detaches it first, if this request's own final
        sequence gets committed into the tree).
        """
        cached_len = 0
        cache_handle = req.cache_handle
        matched_indices = None
        if self.cache_manager is not None:
            ids_tensor = torch.tensor(req.input_ids, dtype=torch.int64, device=self.page_table.device)
            cached_len, cache_handle = self.cache_manager.match(ids_tensor)
            # Never match the WHOLE prompt: the model still needs to run at
            # least one real forward step to produce the logits the first
            # generated token is sampled from (a fully-matched prompt has
            # nothing left to extend, so the prefill batch would carry zero
            # tokens for this request -- confirmed directly: an unclamped
            # full match produced an empty generation). Leave the last
            # prompt token un-cached, same convention real serving engines
            # (vLLM/sglang) use for this exact edge case. Re-align down to
            # the page boundary afterward: clamping can land mid-page when
            # page_size > 1 (page_size == 1 in every config tried so far,
            # where this is a no-op), and the pool/tree's own addressing
            # assumes cached_len is always a whole number of pages.
            from freetoken.utils import align_down

            cached_len = align_down(min(cached_len, len(req.input_ids) - 1), self.page_size)
            self.cache_manager.lock(cache_handle)
            matched_indices = cache_handle.get_matched_indices()
            req.cache_handle = cache_handle
            req.cached_len = cached_len
        pending = make_pending_req(req.uid, req.input_ids, req.sampling_params, cache_handle, cached_len)
        uid = self.scheduler.add(pending)
        req.uid = uid
        req.table_idx = pending._table_idx  # noqa: SLF001
        if self.cache_manager is not None:
            self._admitted_cached_len[req.table_idx] = cached_len
        self._allocate_slot(req.table_idx, cached_len, matched_indices)
        return req

    def _allocate_slot(
        self, table_idx: int, cached_len: int = 0, matched_indices: torch.Tensor | None = None
    ) -> None:
        """Give page-table row ``table_idx`` its own disjoint KV-pool slot
        run for the un-cached tail, real per-request isolation instead of
        the old shared identity map (issue #173) -- plus, when ``cached_len``
        > 0 (issue #12), the matched (reused, tree-owned) slot indices for
        the cached prefix, so attention over the full history reads the
        already-computed K/V for those positions instead of a fresh
        (garbage, never-written) allocation.

        With prefix caching on (issue #12), a finished request's slots stay
        tree-owned rather than returning to the pool's free list (see
        :meth:`step`'s commit block), so the pool can run genuinely empty
        even though most of its bytes are just cached, reusable data sitting
        idle. If a fresh allocation doesn't fit, evict enough LRU
        (unlocked) tree nodes to make room and retry once before giving up.
        """
        if cached_len:
            self.page_table[table_idx, :cached_len] = matched_indices[:cached_len]
        remaining = self.max_seq_len - cached_len
        num_pages = -(-remaining // self.page_size)  # ceil
        try:
            slots = self.kv_cache.allocate(table_idx, num_pages=num_pages)
        except RuntimeError:
            if self.cache_manager is None:
                raise
            need = num_pages * self.page_size
            to_evict = min(need, self.cache_manager.evictable_size)
            if to_evict <= 0:
                raise
            evicted = self.cache_manager.evict(to_evict)
            self.kv_cache.free_slots(evicted)
            slots = self.kv_cache.allocate(table_idx, num_pages=num_pages)
        self.page_table[table_idx, cached_len : self.max_seq_len] = slots[:remaining]

    def abort_request(self, uid: int) -> bool:
        """Free a request's page slot and drop it from the scheduler (any phase).

        An aborted request's partial sequence is never committed into the
        prefix cache (issue #12) -- only a request that actually finishes
        does (see :meth:`step`) -- but a matched handle it was holding
        (locked at admission, in :meth:`add_request`) still needs
        unlocking, or those tree nodes stay pinned (never evictable) forever.
        """
        if self.cache_manager is not None:
            handle = self._find_cache_handle(uid)
            if handle is not None:
                self.cache_manager.unlock(handle)
        before = set(self.scheduler._free_slots)  # noqa: SLF001
        ok = self.scheduler.abort(uid)
        if ok:
            for table_idx in set(self.scheduler._free_slots) - before:  # noqa: SLF001
                self._free_slot(table_idx)
                self._full_ids.pop(table_idx, None)
                self._admitted_cached_len.pop(table_idx, None)
        return ok

    def _find_cache_handle(self, uid: int):
        """The live request's locked prefix-cache handle (issue #12), if
        any -- looked up by uid across the pending queue (including a
        chunked continuation) and the decode set, since ``Scheduler.abort``
        itself does not hand the request back to the caller."""
        for pending in self.scheduler.prefill_manager.pending_list:
            if pending.uid == uid:
                return pending.chunked_req.cache_handle if pending.chunked_req is not None else pending._cache_handle  # noqa: SLF001
        for req in self.scheduler.decode_manager.running_reqs:
            if req.uid == uid:
                return req.cache_handle
        return None

    def _free_slot(self, table_idx: int) -> None:
        # Return this row's KV-pool slot run so a later request admitted to
        # the same table_idx can allocate a fresh one (issue #173).
        self.kv_cache.free(table_idx)

    # -- the loop -------------------------------------------------------------

    @torch.inference_mode()
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
            if self.cache_manager is not None:
                # Track this request's FULL token history for a later commit
                # into the prefix cache (issue #12) -- req.input_ids itself
                # is replaced with just the latest token during decode (see
                # the comment on that a few lines below), so it alone can't
                # supply the full sequence once this request finishes.
                if batch.phase == "decode":
                    self._full_ids[req.table_idx].append(tok)
                else:
                    self._full_ids[req.table_idx] = list(req.input_ids)
            # Tool-call anchor detection (issue #171): the FIRST time this
            # request samples the tool-call-opener token, record the state
            # length just after it (req.device_len, already bumped above --
            # matches Req.toolcall_anchor_len's own field docstring: "its
            # index + 1"). Once only (the `is None` guard): a real tool
            # call is opened once per turn, and re-arming on a later
            # occurrence (e.g. inside the arguments JSON, or a second call
            # in the same turn) would move the anchor to a shallower-reuse
            # -- but still theoretically valid -- point that isn't what a
            # client-side rewrite of the FIRST call actually needs.
            if (
                self.toolcall_anchor_id is not None
                and tok == self.toolcall_anchor_id
                and req.toolcall_anchor_len is None
            ):
                req.toolcall_anchor_len = req.device_len
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

        # Commit each finished request's full sequence into the prefix cache
        # (issue #12) BEFORE the scheduler frees its row below -- a later
        # request can then reuse it instead of recomputing. Ownership of the
        # committed slots transfers to the tree: detach (not free) releases
        # this table_idx's own bookkeeping so a future, unrelated request
        # can reuse the row, without returning the now-tree-owned slots to
        # the pool's free list (only the tree's own eviction does that).
        if self.cache_manager is not None:
            for req in finished:
                full_ids = self._full_ids.pop(req.table_idx, None)
                admitted_cached_len = self._admitted_cached_len.pop(req.table_idx, 0)
                if req.cache_handle is not None:
                    self.cache_manager.unlock(req.cache_handle)
                if full_ids:
                    ids_tensor = torch.tensor(full_ids, dtype=torch.int64, device=self.device)
                    indices = self.page_table[req.table_idx, : len(full_ids)].clone()
                    insert_result = self.cache_manager.commit(ids_tensor, indices)
                    # Ownership of the committed slots transfers to the tree
                    # -- detach (not the normal _free_slot/free path below)
                    # releases only this table_idx's bookkeeping, so a
                    # future unrelated request can reuse the row, without
                    # returning the now-tree-owned slots to the pool's free
                    # list (only the tree's own eviction does that).
                    self.kv_cache.detach(req.table_idx)
                    # Not everything this row's own MHAKVCache allocation
                    # (the [admitted_cached_len, max_seq_len) range --
                    # [0, admitted_cached_len) was never this row's own
                    # allocation at all, it was an aliased reuse of ANOTHER
                    # node's slots) ends up tree-owned:
                    # insert_result.cached_len (InsertResult's own "length
                    # already in cache before insertion" -- base.py) is how
                    # much of what we offered the tree already had
                    # elsewhere -- always >= admitted_cached_len (our own
                    # reused prefix is guaranteed still there, since we
                    # locked it), so [admitted_cached_len,
                    # insert_result.cached_len) is real, valid, but
                    # redundant data (freeable), and [insert_result.
                    # cached_len, len(full_ids)) is what the tree just
                    # newly adopted (do NOT free -- it owns those now).
                    # This row's never-actually-written tail (allocated up
                    # front for the whole max_seq_len, past however many
                    # tokens this request actually generated) was never
                    # part of either span. Both freeable ranges must return
                    # to the pool's free list, or they leak: allocated
                    # forever, owned by neither this (now-gone) request nor
                    # the tree.
                    redundant = self.page_table[req.table_idx, admitted_cached_len : insert_result.cached_len]
                    never_written = self.page_table[req.table_idx, len(full_ids) : self.max_seq_len]
                    self.kv_cache.free_slots(torch.cat([redundant, never_written]))

        # Record the step in the scheduler: promote finished prefills into the
        # decode set, refresh the decode set, and free the rows of requests that
        # completed (hit max_tokens / eos). Diff the scheduler's own free-list
        # before/after (rather than trusting `finished` above, which only
        # reflects THIS step's stop-condition checks) so the KV-pool release
        # exactly matches whichever table_idx rows the scheduler itself
        # actually freed (issue #173) -- the single source of truth for row
        # lifetime. A row already detached above (committed into the tree)
        # makes this a harmless no-op for that table_idx (MHAKVCache.free on
        # an already-detached req_id is a no-op).
        before = set(self.scheduler._free_slots)  # noqa: SLF001
        self.scheduler.complete(batch)
        for table_idx in set(self.scheduler._free_slots) - before:  # noqa: SLF001
            self._free_slot(table_idx)
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
        """Re-plan the MoE cache / KV split and resize the pools (issue #16).

        Re-runs :meth:`_plan_cache_budget` against the *current* free VRAM (not
        the size captured at construction) and resizes the two pools without
        reloading any weights: the host-resident expert banks are untouched, the
        MoE slot cache is re-allocated at the new size (:meth:`OffloadMoeCache.rebuild`,
        which clears its LRU), and the KV pool / page table are rebuilt at the new
        page count. This is what makes the split "elastic": an operator can
        change the VRAM split at runtime (e.g. after a long-context burst eats
        the KV floor) and the engine rebalances in place.
        """
        from freetoken.kvcache import create_kv_pool
        from freetoken.utils.arch import xpu_total_memory

        # Only meaningful on the host-offload MoE path (a live moe_cache).
        cache = getattr(self.model, "moe_cache", None)
        if cache is None:
            return
        if xpu_total_memory() is None:
            # No live VRAM to re-plan against (CPU box); leave the pools alone.
            return

        new_cache_size, new_num_pages = self._plan_cache_budget(
            self.config.model_config, self.config.dtype
        )
        # Resize the MoE slot pool in place (host banks untouched).
        if new_cache_size and new_cache_size != cache.cache_size:
            cache.rebuild(new_cache_size)
        # Resize the KV pool + page table + scheduler at the new page count.
        if new_num_pages and new_num_pages != self._pool_num_pages:
            self._rebuild_kv_pool(new_num_pages)

    def _rebuild_kv_pool(self, num_pages: int) -> None:
        """Rebuild the paged KV pool + page table + scheduler page budget.

        Called by :meth:`rebuild_cache` after the split is re-planned. This
        replaces the pool with a brand-new, empty one (no in-flight request's
        actual KV bytes survive a rebuild -- a pre-existing limitation of the
        elastic-VRAM feature, not something this fixes); the fresh page table
        starts all-zero (no rows allocated), matching a freshly-constructed
        engine's own state. The scheduler's page budget + max-pages are
        updated so it schedules against the new pool.
        """
        # +1 page of slack for MHAKVCache's reserved slot 0 -- see the same
        # comment at construction time (issue #173).
        num_pages += 1
        self.kv_cache = create_kv_pool(
            self.config.model_config,
            page_size=self.page_size,
            num_pages=num_pages,
            device=self.device,
            dtype=self.config.dtype if isinstance(self.config.dtype, torch.dtype) else torch.bfloat16,
        )
        self.page_table = torch.zeros(
            (self.max_running_req + 1, self.max_seq_len), dtype=torch.int64, device=self.device
        )
        self.kv_cache.attach_page_table(self.page_table)
        self.ctx.kv_cache = self.kv_cache
        self.ctx.page_table = self.page_table
        self._pool_num_pages = num_pages
        self._pool_budget = num_pages * self.page_size
        # Keep the scheduler's bookkeeping consistent with the resized pool.
        # max_pages / cache_budget are plain attrs the admission path reads; the
        # prefill manager holds its own copy of the budget (the chunked-prefill
        # fit test), so resync that too. (In-flight requests already own their
        # page slots; only the budget that bounds *new* admission changes.)
        self.scheduler.max_pages = num_pages
        self.scheduler.cache_budget = self._pool_budget
        prefill = getattr(self.scheduler, "prefill_manager", None)
        if prefill is not None and hasattr(prefill, "cache_budget"):
            prefill.cache_budget = self._pool_budget
