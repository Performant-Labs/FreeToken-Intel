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

        # Fixed-shape staging for the copy plan. A layer's routed set has at most
        # batch*top_k positions, so size these for the larger of (E, cache_size).
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

    def bank_views(self, n: int | None = None) -> tuple[torch.Tensor, ...]:
        """Per-bank slot-cache views in registration order: the full ``[S]`` pool
        (decode gather) or its first ``n`` slots (a materialized prefill layer)."""
        if not self.banks:
            raise AssertionError("set_bank_sources must register the banks first")
        if n is None:
            return tuple(cache for _, cache in self.banks)
        return tuple(cache[:n] for _, cache in self.banks)

    # -- reset -----------------------------------------------------------------

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

    def materialize_layer(self, layer_id: int) -> None:
        """Place a whole layer into slots ``[0, E)`` (slot == expert, no LRU).

        Mirrors the upstream ``_materialize_layer`` kernel: a slot already
        owned by *this* layer is just refreshed (usage = step); a slot owned by
        *another* layer is released (its id's ``slot_for_id`` cleared) and then
        reused. ``num_indices`` is set to ``num_experts`` so a following
        :meth:`copy_missing` copies the entire layer's host rows into the slots.
        """
        if not (0 <= layer_id < self.num_layers):
            raise ValueError(f"Invalid materialize layer id {layer_id}")
        E = self.num_experts
        S = self.cache_size
        base = layer_id * E
        step = int(self.step.item()) + 1
        self.step.fill_(step)

        # Slot -> old id (a python list so the loop below stays allocation-free
        # relative to the tensor writes). Two cases release a slot:
        #  * owned by a DIFFERENT layer -> released (that layer's expert is
        #    evicted; it re-fetches on its next decode step), and
        #  * owned by THIS layer in a DECODE slot (>= 2E) -> released too, since
        #    prefill re-homes this layer's experts into the double buffer.
        #     Without this, a decode slot a previous decode step placed an expert
        #     in would leave the expert double-resident (slot_for_id says the
        #     decode slot, id_of_slot still claims the double-buffer slot), so the
        #     next decode step would mis-classify a hit as a miss (or vice versa).
        slot_for_id = self.slot_for_id.view(-1)
        old_ids = self.id_of_slot.tolist()
        for slot in range(S):
            old_id = old_ids[slot]
            if old_id < 0:
                continue
            if not (base <= old_id < base + E) or slot >= E:
                slot_for_id[old_id] = -1
                self.id_of_slot[slot] = -1
                self.usage[slot] = 0
        for expert in range(E):
            self.id_of_slot[expert] = base + expert
            slot_for_id[base + expert] = expert
            self.usage[expert] = step
            self.evict_slots[expert] = expert
            self.src_indices[expert] = expert  # layer-local row == expert
        self.num_indices.fill_(E)
        self._pending_src_layer = layer_id
        self._pending_whole_layer = True

    # -- decode: timestamp LRU -------------------------------------------------

    def ensure_experts(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        """Make this layer's routed experts resident; rewrite ``expert_ids`` to
        slot ids in place (the downstream gather indexes the slot cache by it).

        Hits (already resident, slot >= 2E) bump the slot's usage. Misses evict
        the victim ``min(range(2E, S), key=(usage, slot))`` and schedule the
        (slot, host-row) pair for :meth:`copy_missing`. The double buffer (slots
        < 2E) is never evicted. Raises if a miss cannot be placed (pool exhausted
        for the double buffer) -- the operator must size ``cache_size`` larger.
        """
        E = self.num_experts
        S = self.cache_size
        base = layer_id * E
        if self.banks and S < E + 1:
            raise RuntimeError(
                f"cache_size {S} cannot hold the double buffer (2E={2 * E}) plus a "
                "decode slot; raise moe_cache_size"
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

        # Phase 1: hits -- every resident routed expert (slot != -1, whether it
        # lives in the double buffer [0, E) or a decode slot >= 2E) keeps its
        # slot and is NOT re-copied; its slot's usage is stamped to this step.
        # This mirrors upstream's ``tl.store(usage+slot, step, mask=is_hit)``
        # (is_hit = active & slot >= 0) and, crucially, refreshes a double-buffer
        # slot that another expert's decode-slot copy is about to reuse, so the
        # eviction phase below never picks it as the LRU victim.
        for expert in seen:
            slot = int(layer_slots[expert].item())
            if slot != -1:
                self.usage[slot] = step

        # Phase 2: misses -- a routed expert that is NOT resident evicts the LRU
        # decode slot (min (usage, slot) over slots >= 2E) and schedules the copy.
        # A resident expert (even in the double buffer) is a hit and never lands
        # here, so a decode step right after a prefill materialize is all hits.
        missing = [e for e in seen if int(layer_slots[e].item()) == -1]
        missing.sort()
        self.stat_missing += len(missing)
        self.stat_calls += 1
        num = 0
        usage = self.usage.tolist()
        for idx, expert in enumerate(missing):
            if num >= self.num_indices.shape[0]:
                raise RuntimeError(
                    f"layer {layer_id}: {len(missing)} misses exceed the staging buffer"
                )
            victim = min(range(2 * E, S), key=lambda s: (usage[s], s))
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

        # Phase 3: rewrite every position to its (new) slot; a position whose
        # expert was left non-resident (pool exhausted) stays -1.
        for i in range(flat.numel()):
            flat[i] = int(layer_slots[int(flat[i].item())].item())

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
        gather); empty if none resident."""
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
