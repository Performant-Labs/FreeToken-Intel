"""Main engine loop: prefill, decode, sampling.

Upstream NVIDIA path: python/freetoken/engine/engine.py
Fill in: GitHub issue `engine-loop` (see docs/architecture.md).

This is the minimal *functional* engine the B70 port runs. It wires the pieces
that are already real -- the model's forward, the paged KV pool, the reference
attention backend, and the sampler -- into a prefill/decode loop:

    Engine(config)
        .add_request(Req)      -> assign a slot / table index
        .generate()             -> prefill the prompt, then decode until every
                                   request hits its stop condition (eos or
                                   max_tokens) and return the generated ids

Each :meth:`step` builds the batch's device tensors (``input_ids`` /
``positions`` / ``out_loc``) from the request state, sets them on the global
context, runs ``model(...)`` once, samples a next token per request, and
advances each request's history length. Prefill and decode differ only in how
the tensors are assembled (a request's whole extend during prefill; one new
token during decode), which the model's forward already branches on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from freetoken.attention import create_attention_backend
from freetoken.core import Batch, Context, Req, get_global_ctx, set_global_ctx
from freetoken.kvcache import create_kv_pool
from freetoken.models import create_model
from freetoken.models.loader import load_model
from freetoken.utils.arch import is_xpu_available


@dataclass
class ForwardOutput:
    """Per-request results of one engine step."""

    # Next-token id per request, in batch order (int64 [bs]).
    next_token_ids: torch.Tensor
    # Which requests finished this step (hit eos / max_tokens).
    finished: List[Req]


class Engine:
    """A minimal functional inference engine (issue `engine-loop`, #14).

    ``config`` is an :class:`~freetoken.engine.config.EngineConfig`. On
    construction the engine builds the model (loading weights from the
    checkpoint, or fabricating dummy expert banks when ``use_dummy_weight`` is
    set), creates the paged KV pool + page table + attention backend + sampler,
    and installs them on the global context so the model's ``forward`` can read
    them.
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
        # on host offload banks). use_dummy_weight fabricates the expert banks
        # offline so the engine is testable without a checkpoint on disk.
        dtype = config.dtype if isinstance(config.dtype, torch.dtype) else None
        self.model, _expert_sources = load_model(
            config.model_path,
            device,
            dtype=dtype,
            dummy=bool(getattr(config, "use_dummy_weight", False)),
        )

        # Size the paged KV pool. The pool is indexed by token slot
        # (page_size==1 in the reference path), one row per (request, position).
        max_seq_len = config.max_seq_len
        num_pages = config.num_page_override or (config.max_running_req * max_seq_len)
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

        # Global context the model's forward reads.
        self.ctx = Context(page_size=self.page_size)
        self.ctx.kv_cache = self.kv_cache
        self.ctx.attn_backend = self.attn_backend
        self.ctx.page_table = self.page_table
        set_global_ctx(self.ctx)

        # Request bookkeeping.
        self._reqs: List[Req] = []
        self._next_table_idx = 0

    def _build_sampler(self, model_config, device) -> "object":
        from freetoken.engine.sample import Sampler

        # No eos id in the reference path; -1 is outside any real vocab.
        return Sampler(eos_token_id=-1, device=device)

    # -- request admission ---------------------------------------------------

    def add_request(self, req: Req) -> Req:
        """Admit a request: assign it a KV slot (table index) and queue it."""
        if self._next_table_idx >= self.max_running_req:
            raise RuntimeError(
                f"cannot admit request: max_running_req={self.max_running_req} reached"
            )
        req.table_idx = self._next_table_idx
        self._next_table_idx += 1
        req.cached_len = 0
        self._reqs.append(req)
        return req

    # -- the loop -------------------------------------------------------------

    def step(self) -> ForwardOutput:
        """Run one engine step (one model forward + one sample) over all reqs."""
        reqs = self._reqs
        if not reqs:
            return ForwardOutput(next_token_ids=torch.empty((0,), device=self.device), finished=[])
        phase = "decode"
        batch = Batch(reqs=reqs, phase=phase)

        input_ids: List[int] = []
        positions: List[int] = []
        out_locs: List[int] = []
        for req in reqs:
            if phase == "prefill":
                ext = req.extend_len
                start = req.cached_len
                ids = req.input_ids
                for t in range(ext):
                    pos = start + t
                    input_ids.append(int(ids[pos]))
                    positions.append(pos)
                    out_locs.append(pos)
            else:  # decode: one new token per request
                pos = req.device_len - 1
                last = req.input_ids
                input_ids.append(int(last[-1]))
                positions.append(pos)
                out_locs.append(pos)

        device = self.device
        batch.input_ids = torch.tensor(input_ids, dtype=torch.int64, device=device)
        batch.positions = torch.tensor(positions, dtype=torch.int64, device=device)
        batch.out_loc = torch.tensor(out_locs, dtype=torch.int64, device=device)

        with self.ctx.forward_batch(batch):
            self.attn_backend.prepare_metadata(batch)
            logits = self.model(batch.input_ids, batch.positions, batch.out_loc)

        from freetoken.engine.sample import BatchSamplingArgs

        sampling_args = BatchSamplingArgs([req.sampling_params for req in reqs])
        next_ids = self.sampler.sample(logits, sampling_args)

        finished: List[Req] = []
        for i, req in enumerate(reqs):
            tok = int(next_ids[i])
            # This step's token is the one that may trip the stop condition, so
            # append it *before* marking the request finished (see generate()).
            req.input_ids = (
                [tok] if phase == "decode" else list(req.input_ids) + [tok]
            )
            req.device_len += 1
            sp = req.sampling_params
            done = tok == self.sampler.eos_token_id
            if not sp.ignore_eos and done:
                req.aborted = True
            if req.device_len >= req.max_device_len:
                req.aborted = True
            if req.aborted:
                finished.append(req)
        return ForwardOutput(next_token_ids=next_ids, finished=finished)

    def generate(self, max_steps: int | None = None) -> List[List[int]]:
        """Run admitted requests to completion; return generated ids per request.

        The first step prefills every request's prompt (which yields each one's
        first generated token); subsequent steps decode one token per request
        until each request hits its stop condition (eos or max_tokens) or
        ``max_steps`` is reached. Returns a list, aligned to the admission
        order, of the *generated* token ids (prompt tokens excluded).

        With no admitted requests this is a no-op returning an empty list (the
        spine's ``Engine.generate(None)`` wiring probe relies on that).
        """
        reqs = getattr(self, "_reqs", None)
        if not reqs:
            return []
        budget = max_steps if max_steps is not None else max(req.max_device_len for req in reqs)
        # ``tokens[i]`` accumulates request i's generated tokens. A request may
        # emit exactly one *final* token that trips its stop condition; that
        # token is appended in the same step that aborts it, so nothing is lost
        # (the old "skip aborted" check dropped the last token).
        tokens: List[List[int]] = [[] for _ in reqs]
        done: List[bool] = [False] * len(reqs)
        steps = 0
        while any(not d for d in done) and steps < budget:
            step = self.step()
            steps += 1
            for i, req in enumerate(reqs):
                if done[i]:
                    continue
                tokens[i].append(int(step.next_token_ids[i]))
                if req.aborted:
                    done[i] = True
            # Retire finished requests (frees their KV slot) so the loop does
            # not keep stepping an already-complete request.
            self._reqs = [r for i, r in enumerate(self._reqs) if not done[i]]
        return [tokens[i] for i in range(len(reqs))]

    def rebuild_cache(self) -> None:
        # No radix / hybrid cache in the reference engine; nothing to rebuild.
        pass
