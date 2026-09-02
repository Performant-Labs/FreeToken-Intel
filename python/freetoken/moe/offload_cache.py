"""Global LRU expert slot cache: host banks -> small device slot pool (issue #7).

Upstream NVIDIA path: python/freetoken/moe/offload_cache.py (1042 lines, CUDA
kernels + flashlib slot_cache + cudaMemcpyBatchAsync). This port is the
**pure-torch CPU mirror** of that timestamp LRU (ADR 0002): the same slot
bookkeeping tensors and the same per-step algorithm, but executed in Python
with ``torch`` so it runs and is unit-testable on a CPU-only box. The one
genuinely XPU-specific piece -- streaming the missed experts from pinned host
memory into the XPU slot pool -- is a plain ``.to(device)`` copy in
:meth:`copy_missing`; on the B70 it becomes oneAPI ``queue.memcpy`` / USM (the
``moe-offload`` issue) with the *same* ``evict_slots`` / ``src_indices`` /
``num_indices`` plan.

Design (from the upstream ``lru_ensure`` / ``_materialize_layer`` kernels):

* The cache owns ``num_layers * num_experts`` flat ids (``id = layer * E +
  expert``) and ``cache_size`` slots. ``slot_for_id[id]`` is the slot an id
  currently occupies (-1 = not resident); ``id_of_slot[slot]`` is the inverse;
  ``usage[slot]`` is the last step the slot was active (the LRU key).
* **Prefill** (:meth:`materialize_layer`) places a *whole layer* into slots
  ``[0, E)`` (slot == expert id, no LRU) and evicts any slot another layer
  owns.
* **Decode** (:meth:`ensure_experts`) touches only the slots at or above
  ``2E``: an already-resident routed expert is a **hit** (its slot's usage is
  bumped); a **miss** evicts the victim ``min(range(2E, S), key=(usage, slot))``
  -- the double buffer owns slots below ``2E``, so it is never evicted -- and
  schedules the evicted slots + their host rows for :meth:`copy_missing`.
* :meth:`copy_missing` copies each scheduled (slot, host-row) pair from the
  host bank into the device slot cache, leaving the slot cache correct for the
  forward's pure-torch gather over the resident slots.

The ``bf16`` schema is the only layout the #17 loader produces today
(``gate_up [L, E, 2I, H]`` + ``down [L, E, H, I]``); later quantized formats
must produce the same logical row shapes or the pool must learn a new schema
(ADR 0002).
"""
from __future__ import annotations

from typing import Optional

import torch

from freetoken.utils import init_logger, is_xpu_available

logger = init_logger(__name__)

# The single bank layout the loader produces today (ADR 0002). A format's schema
# names its banks in registration order; the cache machinery is layout-agnostic
# and iterates the schema. "bf16" is the only layout wired up in this port.
_BANK_SCHEMAS: dict[str, tuple[str, ...]] = {
    "bf16": ("gate_up", "down"),
}


class OffloadMoeCache:
    """A timestamp-LRU pool of device slots fed by per-layer host expert banks.

    Args:
        num_layers: number of MoE layers (layers that carry experts).
        num_experts: experts per MoE layer (``E``).
        cache_size: number of device slots (``S``); must be ``>= 2E`` when
            prefill overlap is enabled (the double buffer borrows the first
            ``2E`` slots), and ``>= E`` otherwise.
        device: the slot-cache device (XPU on the B70, CPU in tests).
        quant_format: bank schema (default ``"bf16"``).
    """

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        cache_size: int,
        device: torch.device,
        quant_format: str = "bf16",
        prefill_overlap: bool = False,
    ) -> None:
        if quant_format not in _BANK_SCHEMAS:
            raise ValueError(f"unknown quant_format {quant_format!r}")
        if cache_size < num_experts:
            raise ValueError(
                f"cache_size {cache_size} must be >= num_experts {num_experts} "
                "(the prefill double buffer borrows the first 2*num_experts slots)"
            )
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.cache_size = cache_size
        self.device = torch.device(device)
        self.quant_format = quant_format
        self.bank_schema = _BANK_SCHEMAS[quant_format]
        self.prefill_overlap = prefill_overlap

        # Forward map: flat id (layer*E + expert) -> slot, -1 if not resident.
        self.slot_for_id = torch.full(
            (num_layers, num_experts), -1, dtype=torch.int32, device=self.device
        )
        # Inverse map: slot -> flat id, -1 if the slot is empty.
        self.id_of_slot = torch.full(
            (cache_size,), -1, dtype=torch.int32, device=self.device
        )
        # LRU key: the step a slot was last active (0 = oldest / never used).
        self.usage = torch.zeros((cache_size,), dtype=torch.int64, device=self.device)
        # Monotonic step counter; bumped by materialize_layer / ensure_experts.
        self.step = torch.zeros((), dtype=torch.int64, device=self.device)
        # Per-layer active mask (diagnostics; mirrors upstream).
        self.active_mask = torch.zeros((num_experts,), dtype=torch.int32, device=self.device)

        # Fixed-shape staging for the copy plan. Both phases stage one row per
        # (evicted) expert: materialize_layer stages up to num_experts (a whole
        # layer) and ensure_experts stages one per miss, so size for the larger.
        plan_slots = max(num_experts, cache_size)
        self.evict_slots = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.src_indices = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.num_indices = torch.zeros((1,), dtype=torch.int64, device=self.device)

        # Host source banks + device slot caches, per bank (attached via
        # set_bank_sources). One [E, ...] tensor per layer per bank on host; the
        # slot cache mirrors the row shape as one unified device pool.
        self.bank_sources: dict[str, list] = {}
        self.bank_caches: dict[str, torch.Tensor] = {}
        self.banks: list[tuple[list, torch.Tensor]] = []

        # The layer whose misses ensure_experts / materialize_layer staged last.
        self._pending_src_layer: Optional[int] = None
        self._pending_whole_layer = False

        # Miss-rate counters (host-side; the XPU port keeps these device-side).
        self.stat_missing = torch.zeros((), dtype=torch.int64, device=self.device)
        self.stat_calls = torch.zeros((), dtype=torch.int64, device=self.device)

    # -- bank wiring -----------------------------------------------------------

    def set_bank_sources(self, sources: dict) -> None:
        """Attach the host expert banks and allocate a device slot cache per bank.

        ``sources`` maps each bank name in the format's schema to a list of
        ``num_layers`` host tensors, one ``[num_experts, ...]`` per layer (the
        layout :func:`freetoken.models.weight.load_moe_expert_sources` returns).
        Each bank's slot cache mirrors the row shape / dtype of its per-layer
        host tensor as a single ``[cache_size, ...]`` pool on ``self.device``.
        """
        if set(sources) != set(self.bank_schema):
            raise ValueError(f"banks {sorted(sources)} do not match the {self.quant_format!r} schema {self.bank_schema}")
        self.banks = []
        for bank in self.bank_schema:
            per_layer = sources[bank]
            if len(per_layer) != self.num_layers:
                raise ValueError(f"bank {bank!r} has {len(per_layer)} layers, expected {self.num_layers}")
            row_shape = per_layer[0].shape[1:]  # [E, ...] -> row is [...,]
            cache = torch.zeros((self.cache_size, *row_shape), dtype=per_layer[0].dtype, device=self.device)
            self.bank_sources[bank] = list(per_layer)
            self.bank_caches[bank] = cache
            self.banks.append((self.bank_sources[bank], cache))

    def rebuild(self, cache_size: int) -> None:
        """Re-allocate the device slot pool at a new ``cache_size`` (issue #16).

        The host source banks are untouched (they live in host RAM), so resizing
        the *device* pool is cheap: reallocate the size-dependent tensors and the
        per-bank slot caches at the new size, then reset the (now stale) LRU
        bookkeeping. This is what makes the elastic-memory split re-plannable at
        runtime -- the engine re-runs the budget planner and calls this to resize
        the cache without reloading any weights.
        """
        if cache_size < self.num_experts:
            raise ValueError(
                f"rebuild cache_size {cache_size} must be >= num_experts {self.num_experts}"
            )
        old_size = self.cache_size
        self.cache_size = cache_size
        # Re-allocate the size-dependent slot-pool tensors. (slot_for_id and
        # active_mask are E-shaped, not S-shaped, so they are left as-is.)
        self.id_of_slot = torch.full((cache_size,), -1, dtype=torch.int32, device=self.device)
        self.usage = torch.zeros((cache_size,), dtype=torch.int64, device=self.device)
        plan_slots = max(self.num_experts, cache_size)
        self.evict_slots = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.src_indices = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        # Re-allocate each bank's device slot cache at the new size (keep the
        # host sources; the row shape / dtype come from the registered source).
        # A live engine always has banks registered (the loader attaches them at
        # load, and rebuild only runs on a live cache); the guard merely makes a
        # bare rebuild() a clean no-op on the bank dicts instead of a KeyError.
        if self.bank_sources:
            for bank in self.bank_schema:
                per_layer = self.bank_sources[bank]
                row_shape = per_layer[0].shape[1:]
                self.bank_caches[bank] = torch.zeros(
                    (cache_size, *row_shape), dtype=per_layer[0].dtype, device=self.device
                )
            # Rebuild the (sources, cache) pairs with the new caches.
            self.banks = [
                (self.bank_sources[bank], self.bank_caches[bank]) for bank in self.bank_schema
            ]
        # The old pool is gone: clear the LRU mapping + step counter.
        self.slot_for_id.fill_(-1)
        self.id_of_slot.fill_(-1)
        self.usage.zero_()
        self.step.zero_()
        self.active_mask.zero_()
        self._pending_src_layer = None
        self._pending_whole_layer = False

    def bank_views(self, n: int | None = None) -> tuple[torch.Tensor, ...]:
        """Per-bank slot-cache views in registration order: the full ``[S]`` pool
        (decode gather) or its first ``n`` slots (a materialized prefill layer)."""
        if not self.banks:
            raise AssertionError("set_bank_sources must register the banks first")
        if n is None:
            return tuple(cache for _, cache in self.banks)
        return tuple(cache[:n] for _, cache in self.banks)

    # -- reset -----------------------------------------------------------------

    def _resync_slot_for_id(self) -> None:
        """Rebuild the per-layer ``slot_for_id`` rows from ``id_of_slot``.

        ``id_of_slot`` (slot -> flat id) is the single source of truth: every
        mutation (eviction or (re)acquisition) updates it, and it is the view
        the eviction scan (``id_of_slot[victim]``) and :meth:`resident_slots`
        already trust. ``slot_for_id`` (the per-layer expert -> slot rows) is the
        forward's view, and the two must never disagree: when a layer evicts a
        slot that another layer owns, the ``flat_slots[old_id] = -1`` clear is
        computed from the evicting layer's id and lands on the *wrong* row (a
        no-op), leaving the evicted layer's ``slot_for_id`` row holding a stale
        positive slot id that points at a slot now owned by the other layer. The
        forward would then index the slot pool by that stale id and read the
        wrong expert's bytes -- a silent divergence from the in-VRAM path.
        Deriving ``slot_for_id`` from ``id_of_slot`` after each mutation makes
        the desync impossible: a slot is counted as layer L's expert e only if
        ``id_of_slot[slot] == L*E + e``.

        Vectorized (issue moe-offload-gil / #112): the original per-slot
        Python loop called ``.item()`` once per slot -- each one a
        device->host sync on XPU -- so this scaled with ``cache_size`` badly
        enough that growing the cache to fix thrashing (the actual point of
        #112) just moved the bottleneck here instead. This is pure small
        index bookkeeping (not weight-sized data), so batching it into a
        handful of tensor ops has none of ``copy_missing``'s "extra
        intermediate buffer" pitfall -- there's no large data to double-move.
        """
        E = self.num_experts
        ids_cpu = self.id_of_slot.to("cpu", dtype=torch.int64)
        valid = ids_cpu >= 0
        slots = valid.nonzero(as_tuple=True)[0]
        ids_valid = ids_cpu[slots]
        resynced = torch.full((self.num_layers, self.num_experts), -1, dtype=torch.int32)
        resynced[ids_valid // E, ids_valid % E] = slots.to(torch.int32)
        self.slot_for_id.copy_(resynced.to(self.device, non_blocking=True))

    def reset(self) -> None:
        """Clear all slot mappings, usage, the step counter, and the stats."""
        self.slot_for_id.fill_(-1)
        self.id_of_slot.fill_(-1)
        self.usage.zero_()
        self.step.zero_()
        self.active_mask.zero_()
        self.num_indices.zero_()
        self._pending_src_layer = None
        self._pending_whole_layer = False
        if self.bank_caches:
            for cache in self.bank_caches.values():
                cache.zero_()
        self.reset_stats()

    def reset_stats(self) -> None:
        self.stat_missing.zero_()
        self.stat_calls.zero_()

    def decode_miss_stats(self) -> dict:
        return {"missing": int(self.stat_missing.item()), "calls": int(self.stat_calls.item())}

    # -- prefill: whole-layer materialize --------------------------------------

    def materialize_layer(self, layer_id: int, expert_ids: torch.Tensor | None = None) -> None:
        """Place a whole layer into the slot pool for a prefill forward.

        ``expert_ids`` is accepted for signature compatibility only; this method
        does NOT rewrite it. The offload forward maps routed *expert* ids to slot
        ids on the host (from :attr:`slot_for_id`) and never reads a device-side
        slot-id view of the routing tensor. Rewriting ``expert_ids`` in place to
        slot ids (the old contract) is what leaked stale slot ids: ``torch.topk``
        reuses that same storage each step, so an unchanged routing skipped the
        rewrite and the next step read the *prior* step's slot ids as if they were
        expert ids -- an out-of-bounds slots[] read (XPU IndexError) / a silent
        wrong-expert gather (CPU logit divergence, issue #7).

        This is a *batched timestamp-LRU* pass over the whole layer's experts,
        not a fixed-slot double buffer: the pool is a single global LRU shared by
        all layers (the only place 61 GB of experts physically fit), so a layer
        that is being prefilled evicts the LRU-resident experts of *other* layers
        and re-uses their slots. (Upstream, ``materialize_layer`` is a
        double-buffer *prefetch* -- it writes into a separate prefill double
        buffer and never evicts a resident decode slot the way a fixed [0, E)
        remap would; this mirror folds that into the unified global LRU, which is
        the correctness-preserving equivalent.)

        Per expert: an already-resident expert is a **hit** (keeps its slot,
        usage bumped, no re-copy); a **miss** takes a free slot or evicts the
        global LRU victim. After the loop every expert of the layer is resident
        (the forward may then gather the whole layer), and ``num_indices`` counts
        the missed experts so a following :meth:`copy_missing` streams exactly
        those host rows. Requires ``cache_size >= num_experts`` (the constructor
        enforces this) -- below that the pool cannot hold a whole layer.
        """
        if not (0 <= layer_id < self.num_layers):
            raise ValueError(f"Invalid materialize layer id {layer_id}")
        E = self.num_experts
        S = self.cache_size
        base = layer_id * E
        if S < E:
            raise ValueError(
                f"cache_size {S} < num_experts {E}: cannot materialize a whole layer"
            )
        step = int(self.step.item()) + 1
        self.step.fill_(step)
        self.active_mask.zero_()
        for expert in range(E):
            self.active_mask[expert] = 1

        layer_slots = self.slot_for_id[layer_id]
        flat_slots = self.slot_for_id.view(-1)

        # Phase 1: hits -- a resident expert keeps its slot and bumps usage (no
        # re-copy), so re-prefilling an already-warm layer is cheap.
        for expert in range(E):
            slot = int(layer_slots[expert].item())
            if slot != -1:
                self.usage[slot] = step

        # Phase 2: misses -- a non-resident expert takes a free slot (an expert
        # whose id_of_slot was cleared when another layer evicted it) or, if the
        # pool is full, the global LRU victim. Every evicted (slot, host-row)
        # pair is scheduled for :meth:`copy_missing`.
        missing = [e for e in range(E) if int(layer_slots[e].item()) == -1]
        self.stat_missing += len(missing)
        self.stat_calls += 1
        num = 0
        usage = self.usage.tolist()
        for expert in missing:
            if num >= self.evict_slots.shape[0]:
                raise RuntimeError(
                    f"layer {layer_id}: {len(missing)} misses exceed the staging buffer"
                )
            # A free slot: id_of_slot[slot] == -1 (the owner was evicted).
            free = [s for s in range(S) if int(self.id_of_slot[s].item()) == -1]
            if free:
                victim = free[0]
            else:
                victim = min(range(S), key=lambda s: (usage[s], s))
            old_id = int(self.id_of_slot[victim].item())
            if old_id >= 0:
                flat_slots[old_id] = -1
            self.id_of_slot[victim] = base + expert
            layer_slots[expert] = victim
            self.usage[victim] = step
            usage[victim] = step
            self.evict_slots[num] = victim
            self.src_indices[num] = expert  # layer-local host row
            num += 1
        self.num_indices.fill_(num)
        self._pending_src_layer = layer_id
        self._pending_whole_layer = True

        # Resync the per-layer rows from id_of_slot (the authoritative view):
        # Phase 2's flat clear only clears the evicting layer's own row, so an
        # evicted layer's stale entries would otherwise survive (see
        # _resync_slot_for_id). id_of_slot was already set for every touched slot
        # above, so this rebuilds slot_for_id exactly.
        self._resync_slot_for_id()

    # -- decode: timestamp LRU -------------------------------------------------

    def ensure_experts(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        """Make this layer's routed experts resident in the global LRU pool.

        The pool is a single **global** timestamp LRU shared by every layer.
        A routed expert that is already resident (in *any* slot, any layer) is a
        **hit**: it keeps its slot and its usage is bumped. A **miss** evicts the
        global LRU victim ``min(range(S), key=(usage, slot))`` -- including a
        slot another layer holds, which is the whole point of offload (experts
        are re-fetched on demand, never all resident at once) -- and schedules the
        (slot, host-row) pair for :meth:`copy_missing`.

        This mirrors upstream's decode kernel (``min(range(cache_size),
        key=usage)``), which evicts across the *entire* slot space; a double
        buffer is only a prefill optimization, not a protected region.
        """
        E = self.num_experts
        S = self.cache_size
        base = layer_id * E
        if self.banks and S < 2:
            raise RuntimeError(
                f"cache_size {S} < 2: the pool cannot hold a routed expert plus "
                "a victim to evict; raise moe_cache_size"
            )

        flat = expert_ids.reshape(-1)
        seen: list[int] = []
        for expert in flat.tolist():
            if expert not in seen:
                seen.append(expert)
        self.active_mask.zero_()
        for expert in seen:
            self.active_mask[expert] = 1

        step = int(self.step.item()) + 1
        self.step.fill_(step)
        # Row view for this layer (expert -> slot); the flat view is used for the
        # cross-layer inverse update on eviction (upstream's slot_for_id.view(-1)).
        layer_slots = self.slot_for_id[layer_id]
        flat_slots = self.slot_for_id.view(-1)

        # Phase 1: hits -- every resident routed expert (any slot, any layer) keeps
        # its slot and is NOT re-copied; its slot's usage is stamped to this step.
        # Mirrors upstream's ``tl.store(usage+slot, step, mask=is_hit)``.
        for expert in seen:
            slot = int(layer_slots[expert].item())
            if slot != -1:
                self.usage[slot] = step

        # Phase 2: misses -- a routed expert that is NOT resident evicts the
        # global LRU slot (min (usage, slot) over ALL slots) and schedules the copy.
        # A resident expert is a hit and never lands here. (With a warm pool the
        # misses are the experts another layer evicted; with a cold pool they are
        # everything routed.)
        missing = [e for e in seen if int(layer_slots[e].item()) == -1]
        missing.sort()
        self.stat_missing += len(missing)
        self.stat_calls += 1
        num = 0
        usage = self.usage.tolist()
        for idx, expert in enumerate(missing):
            if num >= self.evict_slots.shape[0]:
                raise RuntimeError(
                    f"layer {layer_id}: {len(missing)} misses exceed the staging buffer"
                )
            victim = min(range(S), key=lambda s: (usage[s], s))
            old_id = int(self.id_of_slot[victim].item())
            if old_id >= 0:
                flat_slots[old_id] = -1
            self.id_of_slot[victim] = base + expert
            layer_slots[expert] = victim
            self.usage[victim] = step
            usage[victim] = step
            self.evict_slots[num] = victim
            self.src_indices[num] = expert  # layer-local host row
            num += 1
        self.num_indices.fill_(num)
        self._pending_src_layer = layer_id
        self._pending_whole_layer = False

        # Resync the per-layer rows from id_of_slot: a miss evicts the global
        # LRU slot, which another layer may own; the flat clear above only
        # touches the evicting layer's row, so the evicted layer's stale
        # entries would otherwise survive (see _resync_slot_for_id).
        # (The old Phase-3 in-place rewrite of ``expert_ids`` -> slot ids was
        # removed: see the class/`materialize_layer` docstrings. The forward
        # maps expert -> slot on the host; the routing tensor must keep holding
        # *expert* ids so the next step's topk snapshot is always in-bounds.)
        self._resync_slot_for_id()

    # -- copy engine -----------------------------------------------------------

    def copy_missing(self) -> None:
        """Copy the staged (slot, host-row) pairs from the host banks into the
        device slot caches.

        On the B70 this is the XPU-specific piece (oneAPI ``queue.memcpy`` / USM
        between pinned host and XPU); here it is a plain per-row ``.to(device)``
        copy, which is exactly what a pure-torch gather needs to see the fresh
        bytes. The plan (``evict_slots`` / ``src_indices`` / ``num_indices``) is
        identical, so the XPU engine swaps only this method's body.
        """
        if not self.banks:
            raise AssertionError("set_bank_sources must register the banks first")
        layer_id = self._pending_src_layer
        if layer_id is None:
            return  # nothing staged
        n = int(self.num_indices.item())
        evict = self.evict_slots[:n].tolist()
        src = self.src_indices[:n].tolist()
        for per_layer, cache in self.banks:
            host = per_layer[layer_id]
            for slot, row in zip(evict, src):
                cache[slot].copy_(host[row])
        self._pending_src_layer = None
        self._pending_whole_layer = False

    # -- device helpers --------------------------------------------------------

    def resident_slots(self, layer_id: int) -> list[int]:
        """The slots currently holding this layer's experts (for the forward's
        gather); empty if none are resident. (Under the global-LRU scheme this is
        just the set of slots whose ``id_of_slot`` falls in this layer's id range.)"""
        E = self.num_experts
        S = self.cache_size
        base = layer_id * E
        out: list[int] = []
        for slot in range(S):
            id_ = int(self.id_of_slot[slot].item())
            if base <= id_ < base + E:
                out.append(slot)
        return out

    @property
    def is_xpu(self) -> bool:
        return self.device.type == "xpu" and is_xpu_available()


__all__ = ["OffloadMoeCache", "_BANK_SCHEMAS"]
