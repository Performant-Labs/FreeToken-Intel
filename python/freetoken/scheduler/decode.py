"""Decode batching.

Upstream NVIDIA path: python/freetoken/scheduler/decode.py
Filled in: GitHub issue ``scheduler`` (see docs/architecture.md).

The decode manager owns the set of requests that have already finished their
prompt (``cached_len == device_len``) and are one token per step. Its job is
pure bookkeeping: admit requests that just finished prefilling, drop the ones
that hit their stop condition, and (in :meth:`schedule_next_batch`) emit the
batch of live decode requests. There is no per-token state to manage beyond the
request's own history length, so this stays plain-Python and torch-free.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable

from freetoken.core import Batch, Req

# A set that preserves insertion order. A plain ``set`` has no ordering, so the
# decode batch (and the slot-freeing that follows a step) would iterate in an
# arbitrary, hash-dependent order. Upstream schedules the decode batch in uid
# order; mirroring that with an insertion-ordered set keeps the ported policy
# deterministic and the same order as upstream.
OrderedSet = OrderedDict


@dataclass
class DecodeManager:
    """Holds the in-flight decode requests and emits the decode batch."""

    page_size: int
    running_reqs: "OrderedSet[Req]" = field(default_factory=OrderedSet)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        """Admit / refresh the running set from a batch that just executed.

        ``reqs`` is the batch that just ran. After a *prefill* step a request
        has its first token, so ``can_decode`` (``remain_len > 0``) is what decides whether it
        moves into the decode set. After a *decode* step the request is either
        still decoding (stays) or finished (``remain_len`` drops to 0 ->
        dropped, mirroring the upstream filter). Requests that are neither
        decodable nor runnable are dropped, so the running set only ever
        contains live decode requests. Insertion order is preserved (new admits
        append, survivors keep their place), so the emitted batch stays in
        admission/uid order.
        """
        merged = OrderedDict(self.running_reqs)
        for req in reqs:
            merged[req] = None  # move-to-end if present (re-admit / refresh)
        self.running_reqs = OrderedDict((k, v) for k, v in merged.items() if k.can_decode)

    def remove_req(self, req: Req) -> None:
        self.running_reqs.pop(req, None)

    def abort_req(self, uid: int) -> Req | None:
        """Remove (and return) the running decode request with ``uid``.

        Returns ``None`` when the uid is not in the decode set. The freed page
        slot is released by the caller (the engine frees the table index); this
        only updates the bookkeeping.
        """
        for req in self.running_reqs:
            if req.uid == uid:
                del self.running_reqs[req]
                return req
        return None

    @property
    def inflight_tokens(self) -> int:
        # Upstream reserves (page_size - 1) tokens per request so a decode step
        # that crosses a page boundary never overruns the pool; with
        # page_size == 1 (the Intel token-granular pool) this is 0.
        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)
        return sum(req.remain_len for req in self.running_reqs) + tokens_reserved

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        return Batch(reqs=sorted(self.running_reqs, key=lambda req: req.uid), phase="decode")

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
