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
#
# "gptq_int4" (issue moe-quant-banks-schema, #136): a GPTQ-Int4-quantized
# checkpoint (e.g. the official Qwen/Qwen3.5-35B-A3B-GPTQ-Int4) packs each
# projection into three per-expert tensors -- qweight/qzeros/scales (see
# freetoken.kernel.triton.gptq_linear for the bit layout) -- instead of bf16's
# one. Six banks total: {qweight,qzeros,scales} x {gate_up,down}. All six are
# genuinely one-row-per-expert ([E, ...], same as bf16's two banks) so they
# fit set_bank_sources/copy_missing/rebuild completely unchanged -- these are
# plain index bookkeeping + generic tensor copies with no bf16/float
# assumption anywhere (verified by reading the whole file, not assumed).
#
# GPTQ's fourth component, g_idx, deliberately has NO bank here: it is a
# single [K] tensor per (layer, projection type) -- identical for every
# expert of that projection (g_idx[k] = k // group_size depends only on K and
# group_size, both architecture constants) -- so it does not fit the
# per-expert [E, ...] row-per-slot contract every bank above shares, and
# manufacturing E identical copies of it would be real, pointless memory and
# LRU-slot churn for data that never varies per expert or per step. It lives
# in OffloadMoeCache.extra_metadata instead (see set_extra_metadata /
# get_extra_metadata below): a plain per-layer side table the compute step
# (issue #137) reads directly, never routed through the LRU slot machinery.
#
# "fp8_block" (issue moe-quant-banks-fp8, #152): a block-FP8-quantized
# checkpoint (DeepSeek-V3 / sglang / vLLM's public w8a8_block_fp8 convention,
# see freetoken.kernel.triton.fp8_block_linear) packs each projection into
# two per-expert tensors -- weight (fp8) + weight_scale_inv (the per-block
# scale table) -- instead of GPTQ's three. Four banks total: {weight,
# weight_scale_inv} x {gate_up,down}. All four are one-row-per-expert
# ([E, ...]), same as bf16/gptq_int4, so they fit set_bank_sources/
# copy_missing/rebuild unchanged too. Unlike GPTQ, block-FP8 has no
# shared-across-experts side tensor (no g_idx analogue): every expert's
# weight_scale_inv is genuinely its own, so nothing needs extra_metadata here.
#
# "mxfp4" (issue moe-quant-banks-mxfp4, #153): OCP Microscaling MXFP4 --
# each projection packs into two per-expert tensors (uint8 fp4-nibble
# blocks + uint8 E8M0 shared-exponent scale), half of GPTQ's four, and
# with no g_idx-equivalent side table (MXFP4's scale is fully local to
# its own 32-element block, never shared across a whole projection) --
# see freetoken.kernel.triton.mxfp4_linear.dequantize_mxfp4_blocks and
# freetoken.models.weight.MxfpExpertBank. Four banks total, all
# genuinely one-row-per-expert, so (like gptq_int4) they fit
# set_bank_sources/copy_missing/rebuild unchanged.
#
# "int8_channel" (issue moe-quant-banks-int8, #154): compressed-tensors'
# "pack-quantized" INT8 scheme (verified against a real checkpoint,
# rj1013/gemma-4-26B-A4B-it_q8 -- an EARLIER, unverified draft of this
# format assumed plain unpacked int8 tensors; that was wrong, see issue
# #154's own comment trail and freetoken.kernel.triton.int8_packed_linear's
# module docstring for the full correction). Each projection packs into two
# per-expert tensors -- weight_packed (int32, 4 int8 values densely packed
# per word along K) + weight_scale (one value per (output channel, group)
# pair; num_groups==1 degenerates to pure per-channel, the same mechanism
# serves both) -- unlike gptq_int4 there is no shared per-projection side
# tensor for either of these (packing/grouping is entirely along K, so an
# N-axis expert-row concat never crosses a word or group boundary), but K
# itself (the real logical in-features, from the checkpoint's own
# weight_shape tensor) IS shared across every expert of one projection type
# -- and, unlike GPTQ's group_size (checkpoint-wide but not derivable from
# architecture dims alone), gate_up's K is always hidden_size and down's K
# is always moe_intermediate_size, both architecture constants identical
# across every MoE layer -- so SlotWeightAccessor reads them as two plain
# scalar cache attributes (cache.int8_k_gate_up / cache.int8_k_down), the
# same pattern as cache.gptq_group_size, not a per-layer side table.
_BANK_SCHEMAS: dict[str, tuple[str, ...]] = {
    "bf16": ("gate_up", "down"),
    "gptq_int4": (
        "qweight_gate_up",
        "qzeros_gate_up",
        "scales_gate_up",
        "qweight_down",
        "qzeros_down",
        "scales_down",
    ),
    "fp8_block": (
        "weight_gate_up",
        "scale_gate_up",
        "weight_down",
        "scale_down",
    ),
    "mxfp4": (
        "blocks_gate_up",
        "scales_gate_up",
        "blocks_down",
        "scales_down",
    ),
    "int8_channel": (
        "weight_packed_gate_up",
        "weight_scale_gate_up",
        "weight_packed_down",
        "weight_scale_down",
    ),
}


def gptq_int4_bytes_per_expert_slot(
    hidden_size: int,
    moe_intermediate_size: int,
    group_size: int,
    *,
    qweight_bytes: int = 4,  # int32
    scale_bytes: int = 2,  # fp16/bf16
) -> int:
    """Real packed VRAM bytes for ONE expert slot under the ``gptq_int4``
    schema -- for :func:`freetoken.engine.cache_budget.plan_cache_budget`'s
    ``bytes_per_slot_override`` (issue #16's planner hardcodes the bf16
    2-bank ``(gate_up + down) * dtype_bytes`` formula; a packed int4 slot's
    real footprint is a different, smaller shape entirely, summed per bank
    below rather than approximated by one scalar ``dtype_bytes``).

    Mirrors :func:`freetoken.kernel.triton.gptq_linear`'s packing: for a
    ``[K, N]`` logical projection, ``qweight`` packs 8 int4 values per int32
    word along ``K`` (``K // 8`` words x ``N``); ``qzeros`` packs 8 per word
    along ``N``, one row per group (``ceil(K/group_size) x N // 8``);
    ``scales`` is one fp16/bf16 value per (group, output channel)
    (``ceil(K/group_size) x N``). ``gate_up`` fuses gate_proj + up_proj (K =
    hidden_size, N = 2 * moe_intermediate_size); ``down`` is the reverse
    (K = moe_intermediate_size, N = hidden_size).
    """
    if hidden_size <= 0 or moe_intermediate_size <= 0 or group_size <= 0:
        raise ValueError(
            "gptq_int4_bytes_per_expert_slot needs positive hidden_size / "
            "moe_intermediate_size / group_size"
        )

    def _proj_bytes(k: int, n: int) -> int:
        groups = -(-k // group_size)  # ceil
        qweight = (k // 8) * n * qweight_bytes
        qzeros = groups * (n // 8) * qweight_bytes
        scales = groups * n * scale_bytes
        return qweight + qzeros + scales

    gate_up = _proj_bytes(hidden_size, 2 * moe_intermediate_size)
    down = _proj_bytes(moe_intermediate_size, hidden_size)
    return gate_up + down


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

        # Per-layer side data that does NOT vary per expert/slot (issue #136),
        # e.g. gptq_int4's g_idx: [K], identical for every expert of a
        # projection type, so it never needs the LRU slot-cache/copy_missing
        # machinery above -- see the _BANK_SCHEMAS["gptq_int4"] comment.
        self.extra_metadata: dict[str, list] = {}

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

    def set_extra_metadata(self, name: str, per_layer: list) -> None:
        """Attach a per-layer side tensor that does NOT vary per expert/slot
        (issue #136) -- e.g. gptq_int4's ``g_idx``. Stored as-is (host or
        device, whatever the caller passes), with none of ``set_bank_sources``'s
        per-expert-row / LRU-slot-cache machinery: there is nothing to evict
        or copy-on-miss for data that is identical for every expert of a
        projection type. ``per_layer`` must have exactly ``num_layers``
        entries, one per layer (matching every other bank's per-layer list
        length contract), even though each entry itself has no expert axis.
        """
        if len(per_layer) != self.num_layers:
            raise ValueError(f"extra metadata {name!r} has {len(per_layer)} layers, expected {self.num_layers}")
        self.extra_metadata[name] = list(per_layer)

    def get_extra_metadata(self, name: str, layer_id: int):
        """The per-layer side tensor set by :meth:`set_extra_metadata`, for
        this layer. Raises ``KeyError`` if ``name`` was never set."""
        return self.extra_metadata[name][layer_id]

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


class SlotWeightAccessor:
    """Per-step accessor for one MoE layer's resident-slot expert weights,
    abstracting the offload forward's ``gu[s_i, ...]`` / ``dn[s_i]`` bf16
    indexing over a quantized bank format too (issue moe-quant-banks-
    compute, #137).

    For ``"bf16"`` this is a thin, zero-cost wrapper around the existing
    plain-tensor indexing (behavior is unchanged from before this class
    existed). For ``"gptq_int4"``, ``"fp8_block"`` (issue moe-quant-banks-fp8,
    #152), ``"mxfp4"`` (issue moe-quant-banks-mxfp4, #153), and
    ``"int8_channel"`` (issue moe-quant-banks-int8, #154), each distinct
    slot's packed weights are dequantized **at most once per instance** (a
    `_forward_offload_core` call handles one MoE layer for one step) -- a
    decode step's working set is typically small (<= num_experts active
    slots), so this is a bounded, cheap cost per step, never the whole
    checkpoint at once (the RAM-saving point of #134's whole epic).

    ``dtype`` is the dequant output dtype -- must match the activation
    tensor's dtype (``flat``, whatever ``moe_backend="offload"`` loaded the
    model with -- bf16 by convention, but not guaranteed), NOT the
    checkpoint's own stored scales dtype (a real bug this class had until
    issue #138's real-checkpoint validation caught it: the official
    Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 checkpoint stores ``scales`` as fp16,
    so dequantizing to "whatever scales.dtype is" silently produced fp16
    expert weights that then crashed matmul-ing against the bf16
    activations everywhere else in the model -- caught as a real forward
    pass failure against the real checkpoint, not by any synthetic test).
    ``int8_channel`` dequantizes to ``self._dtype`` for exactly the same
    reason, not to the checkpoint's own scale dtype.
    """

    def __init__(self, cache: "OffloadMoeCache", intermediate: int, dtype: torch.dtype) -> None:
        self.quant_format = getattr(cache, "quant_format", "bf16")
        self._intermediate = intermediate
        self._dtype = dtype
        self._is_xpu = bool(getattr(cache, "is_xpu", False))
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        if self.quant_format == "gptq_int4":
            self._banks = dict(zip(cache.bank_schema, cache.bank_views()))
            # g_idx is deliberately not a bank (see _BANK_SCHEMAS["gptq_int4"]'s
            # own comment) -- get() below reconstructs it implicitly from
            # group_size, correct for this port's only supported case
            # (desc_act=False, the real checkpoint's own setting).
            #
            # No default here on purpose: a wrong-but-plausible group_size
            # (e.g. silently falling back to some other checkpoint's value)
            # would dequantize with the wrong group boundaries and produce
            # silently-wrong-but-finite numbers, not a crash -- caught during
            # this class's own test-writing, when a fixture forgot to set it
            # and got a real, non-obvious 50%-of-elements mismatch instead of
            # an error. The loader (issue #135 / #138) must set
            # ``cache.gptq_group_size`` from the checkpoint's own
            # ``quantization_config.group_size`` before any gptq_int4 forward
            # runs.
            group_size = getattr(cache, "gptq_group_size", None)
            if group_size is None:
                raise ValueError(
                    "OffloadMoeCache.gptq_group_size is not set -- the loader must set it "
                    "from the checkpoint's quantization_config.group_size before any "
                    "gptq_int4 forward pass runs (SlotWeightAccessor refuses to guess)"
                )
            self._group_size = int(group_size)
        elif self.quant_format == "fp8_block":
            self._banks = dict(zip(cache.bank_schema, cache.bank_views()))
            # No group_size-style checkpoint parameter to thread through here:
            # block-FP8's block size is a fixed convention (128), not a
            # per-checkpoint choice like GPTQ's group_size -- every real
            # block-FP8 checkpoint found so far (DeepSeek-V3 included) uses
            # weight_block_size == [128, 128]. An optional cache.fp8_block_size
            # override is still honored (falls back to the module default)
            # in case a future checkpoint's quantization_config.weight_block_size
            # ever differs.
            from freetoken.kernel.triton.fp8_block_linear import _BLOCK as _FP8_DEFAULT_BLOCK

            self._fp8_block = int(getattr(cache, "fp8_block_size", None) or _FP8_DEFAULT_BLOCK)
        elif self.quant_format == "mxfp4":
            self._banks = dict(zip(cache.bank_schema, cache.bank_views()))
        elif self.quant_format == "int8_channel":
            self._banks = dict(zip(cache.bank_schema, cache.bank_views()))
            # K (real logical in-features) is an architecture constant per
            # projection type -- gate_up's is always hidden_size, down's is
            # always moe_intermediate_size -- so, like gptq_group_size, the
            # loader sets these once as plain scalar cache attributes rather
            # than a per-layer side table. No default here on purpose (see
            # the gptq_int4 branch above for the same "refuse to guess"
            # rationale): the loader must set both before any int8_channel
            # forward runs.
            k_gate_up = getattr(cache, "int8_k_gate_up", None)
            k_down = getattr(cache, "int8_k_down", None)
            if k_gate_up is None or k_down is None:
                raise ValueError(
                    "OffloadMoeCache.int8_k_gate_up / int8_k_down are not set -- the loader "
                    "must set them from the checkpoint's real weight_shape before any "
                    "int8_channel forward pass runs (SlotWeightAccessor refuses to guess)"
                )
            self._int8_k_gate_up = int(k_gate_up)
            self._int8_k_down = int(k_down)
        else:
            self._gu, self._dn = cache.bank_views()

    def get_gptq_packed(
        self, s_i: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Raw packed ``(qweight_gate_up, qzeros_gate_up, scales_gate_up,
        qweight_down, qzeros_down, scales_down, group_size)`` views for slot
        ``s_i`` -- ``gptq_int4`` only, and never dequantized (unlike
        :meth:`get`). For the native fused-GEMM forward path (issue
        `moe-quant-banks-native`, #139): :func:`freetoken.kernel.triton.
        gptq_fused_linear.fused_gptq_expert_forward` consumes these directly,
        skipping the dense-weight materialization :meth:`get` performs.
        """
        if self.quant_format != "gptq_int4":
            raise ValueError(f"get_gptq_packed is gptq_int4-only, got quant_format={self.quant_format!r}")
        b = self._banks
        return (
            b["qweight_gate_up"][s_i], b["qzeros_gate_up"][s_i], b["scales_gate_up"][s_i],
            b["qweight_down"][s_i], b["qzeros_down"][s_i], b["scales_down"][s_i],
            self._group_size,
        )

    def get_fp8_packed(
        self, s_i: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Raw packed ``(weight_fp8_gate_up, weight_scale_inv_gate_up,
        weight_fp8_down, weight_scale_inv_down, block)`` views for slot
        ``s_i`` -- ``fp8_block`` only, and never dequantized (unlike
        :meth:`get`). For the native fused-GEMM forward path (issue
        `moe-quant-banks-native-multi`, #163): :func:`freetoken.kernel.
        triton.fused_fp8_linear.fused_fp8_expert_forward` consumes these
        directly, skipping the dense-weight materialization :meth:`get`
        performs.
        """
        if self.quant_format != "fp8_block":
            raise ValueError(f"get_fp8_packed is fp8_block-only, got quant_format={self.quant_format!r}")
        b = self._banks
        return (
            b["weight_gate_up"][s_i], b["scale_gate_up"][s_i],
            b["weight_down"][s_i], b["scale_down"][s_i],
            self._fp8_block,
        )

    def get_mxfp4_packed(
        self, s_i: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Raw packed ``(blocks_gate_up, scales_gate_up, blocks_down,
        scales_down)`` views for slot ``s_i`` -- ``mxfp4`` only, and never
        dequantized (unlike :meth:`get`). For the native fused-GEMM forward
        path (issue `moe-quant-banks-native-multi`, #163):
        :func:`freetoken.kernel.triton.fused_mxfp4_linear.
        fused_mxfp4_expert_forward` consumes these directly, skipping the
        dense-weight materialization :meth:`get` performs. No shared
        per-format cache attribute needed (unlike gptq_int4's group_size or
        fp8_block's block size): the quantization block size (32) is a
        fixed MXFP4 convention, not a per-checkpoint choice.
        """
        if self.quant_format != "mxfp4":
            raise ValueError(f"get_mxfp4_packed is mxfp4-only, got quant_format={self.quant_format!r}")
        b = self._banks
        return (
            b["blocks_gate_up"][s_i], b["scales_gate_up"][s_i],
            b["blocks_down"][s_i], b["scales_down"][s_i],
        )

    def get_int8_packed(
        self, s_i: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """Raw packed ``(weight_packed_gate_up, weight_scale_gate_up,
        weight_packed_down, weight_scale_down, k_gate_up, k_down)`` views
        for slot ``s_i`` -- ``int8_channel`` only, and never dequantized
        (unlike :meth:`get`). For the native fused-GEMM forward path (issue
        `moe-quant-banks-native-multi`, #163): :func:`freetoken.kernel.
        triton.fused_int8_linear.fused_int8_expert_forward` consumes these
        directly, skipping the dense-weight materialization :meth:`get`
        performs.
        """
        if self.quant_format != "int8_channel":
            raise ValueError(f"get_int8_packed is int8_channel-only, got quant_format={self.quant_format!r}")
        b = self._banks
        return (
            b["weight_packed_gate_up"][s_i], b["weight_scale_gate_up"][s_i],
            b["weight_packed_down"][s_i], b["weight_scale_down"][s_i],
            self._int8_k_gate_up, self._int8_k_down,
        )

    def expert_forward(self, s_i: int, x: torch.Tensor) -> torch.Tensor:
        """One MoE expert's SwiGLU forward (``down(silu(gate(x)) * up(x))``)
        for slot ``s_i`` against input ``x``. Previously each offload
        forward call site (``qwen3_moe``/``qwen3_5_moe``) ran this same
        formula itself, against :meth:`get`'s dense output, via a small
        ``_expert_compute`` helper duplicated in each model file; centralized
        here instead so it can also dispatch to a native fused-GEMM path
        (issue `moe-quant-banks-native`, #139; extended to fp8_block/mxfp4/
        int8_channel by #163) when one is expected to win: real XPU
        hardware, a format with a fused kernel, and ``x``'s row count at or
        below that format's measured crossover (see each format's own
        ``prefer_fused_over_dequant``: :func:`freetoken.kernel.triton.
        gptq_fused_linear.prefer_fused_over_dequant`,
        :func:`freetoken.kernel.triton.fused_fp8_linear.
        prefer_fused_over_dequant`, :func:`freetoken.kernel.triton.
        fused_mxfp4_linear.prefer_fused_over_dequant`,
        :func:`freetoken.kernel.triton.fused_int8_linear.
        prefer_fused_over_dequant`). Every other case
        falls back to dequantizing via :meth:`get` first, the same math the
        old per-model helpers ran.
        """
        if self.quant_format == "gptq_int4" and self._is_xpu:
            from freetoken.kernel.triton.gptq_fused_linear import (
                fused_gptq_expert_forward,
                prefer_fused_over_dequant,
            )

            if prefer_fused_over_dequant(x.shape[0]):
                qw_gu, qz_gu, s_gu, qw_dn, qz_dn, s_dn, group_size = self.get_gptq_packed(s_i)
                return fused_gptq_expert_forward(
                    x, qw_gu, qz_gu, s_gu, qw_dn, qz_dn, s_dn,
                    group_size=group_size, intermediate=self._intermediate, out_dtype=self._dtype,
                )
        elif self.quant_format == "fp8_block" and self._is_xpu:
            from freetoken.kernel.triton.fused_fp8_linear import (
                fused_fp8_expert_forward,
                prefer_fused_over_dequant,
            )

            if prefer_fused_over_dequant(x.shape[0]):
                w_gu, s_gu, w_dn, s_dn, block = self.get_fp8_packed(s_i)
                return fused_fp8_expert_forward(
                    x, w_gu, s_gu, w_dn, s_dn,
                    intermediate=self._intermediate, block=block, out_dtype=self._dtype,
                )
        elif self.quant_format == "mxfp4" and self._is_xpu:
            from freetoken.kernel.triton.fused_mxfp4_linear import (
                fused_mxfp4_expert_forward,
                prefer_fused_over_dequant,
            )

            if prefer_fused_over_dequant(x.shape[0]):
                blk_gu, sc_gu, blk_dn, sc_dn = self.get_mxfp4_packed(s_i)
                return fused_mxfp4_expert_forward(
                    x, blk_gu, sc_gu, blk_dn, sc_dn,
                    intermediate=self._intermediate, out_dtype=self._dtype,
                )
        elif self.quant_format == "int8_channel" and self._is_xpu:
            from freetoken.kernel.triton.fused_int8_linear import (
                fused_int8_expert_forward,
                prefer_fused_over_dequant,
            )

            if prefer_fused_over_dequant(x.shape[0]):
                wp_gu, ws_gu, wp_dn, ws_dn, k_gu, k_dn = self.get_int8_packed(s_i)
                return fused_int8_expert_forward(
                    x, wp_gu, ws_gu, wp_dn, ws_dn,
                    intermediate=self._intermediate, k_gate_up=k_gu, k_down=k_dn, out_dtype=self._dtype,
                )
        gate_w, up_w, down_w = self.get(s_i)
        return (torch.nn.functional.silu(x @ gate_w.t()) * (x @ up_w.t())) @ down_w.t()

    def get(self, s_i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(gate_w, up_w, down_w)`` for slot ``s_i``, in ``[out, in]``
        weight orientation (matching ``nn.Linear.weight`` / :meth:`expert_forward`'s
        expected shape) -- gate/up are ``[I, H]``, down is ``[H, I]``."""
        if self.quant_format == "bf16":
            i = self._intermediate
            return self._gu[s_i, 0:i], self._gu[s_i, i : 2 * i], self._dn[s_i]
        cached = self._cache.get(s_i)
        if cached is not None:
            return cached
        if self.quant_format == "fp8_block":
            from freetoken.kernel.triton.fp8_block_linear import dequantize_block_fp8

            b = self._banks
            # dequantize_block_fp8 reconstructs [N, K] -- already the [out, in]
            # orientation this port's bank rows use (unlike GPTQ's dequant,
            # which returns [in, out] and needs a .T). out_dtype is
            # self._dtype (the model's activation dtype), NOT the checkpoint's
            # own stored weight_scale_inv dtype -- the same dtype bug class
            # issue #138 found for gptq_int4 (see the class docstring).
            gu_dense = dequantize_block_fp8(
                b["weight_gate_up"][s_i], b["scale_gate_up"][s_i],
                block=self._fp8_block, out_dtype=self._dtype,
            )
            dn_dense = dequantize_block_fp8(
                b["weight_down"][s_i], b["scale_down"][s_i],
                block=self._fp8_block, out_dtype=self._dtype,
            )
            i = self._intermediate
            result = (gu_dense[0:i].contiguous(), gu_dense[i : 2 * i].contiguous(), dn_dense.contiguous())
            self._cache[s_i] = result
            return result
        if self.quant_format == "int8_channel":
            from freetoken.kernel.triton.int8_packed_linear import dequantize_int8_packed as _dequant

            b = self._banks
            # weight_packed/weight_scale are already in [out, in-packed] /
            # [out, groups] orientation (per output channel = per row) -- no
            # transpose needed, unlike GPTQ's dequant which returns [in, out]
            # and must be .T'd. out_dtype is self._dtype (the model's
            # activation dtype), NOT the checkpoint's own stored scale dtype
            # -- the same dtype bug class issue #138 found for gptq_int4.
            gu_dense = _dequant(
                b["weight_packed_gate_up"][s_i], b["weight_scale_gate_up"][s_i],
                k=self._int8_k_gate_up, out_dtype=self._dtype,
            )
            dn_dense = _dequant(
                b["weight_packed_down"][s_i], b["weight_scale_down"][s_i],
                k=self._int8_k_down, out_dtype=self._dtype,
            )
            i = self._intermediate
            result = (gu_dense[0:i], gu_dense[i : 2 * i], dn_dense)
            self._cache[s_i] = result
            return result
        if self.quant_format == "mxfp4":
            from freetoken.kernel.triton.mxfp4_linear import dequantize_mxfp4_blocks

            b = self._banks
            # dequantize_mxfp4_blocks already returns [out_features, K] --
            # gate_up's out_features axis is the bank row's leading axis
            # (unlike GPTQ's dequantize_gptq_int4_sequential_groups, which
            # returns [in_features, out_features] and needs a .T), so the
            # bf16 bank rows' [out, in] orientation falls out directly.
            # out_dtype is self._dtype (the model's activation dtype), NOT
            # any checkpoint-stored scales dtype -- see the class docstring
            # for the real bug (#138) this exact mistake caused for GPTQ.
            gu_dense = dequantize_mxfp4_blocks(
                b["blocks_gate_up"][s_i], b["scales_gate_up"][s_i], out_dtype=self._dtype,
            )
            dn_dense = dequantize_mxfp4_blocks(
                b["blocks_down"][s_i], b["scales_down"][s_i], out_dtype=self._dtype,
            )
            i = self._intermediate
            result = (gu_dense[0:i], gu_dense[i : 2 * i], dn_dense)
            self._cache[s_i] = result
            return result

        from freetoken.kernel.triton.gptq_linear import dequantize_gptq_int4_sequential_groups as _dequant

        b = self._banks
        # dequantize_gptq_int4_sequential_groups returns [in_features, out_features]
        # (nn.Linear's transpose); .T gives the [out, in] orientation this port's
        # bf16 bank rows already use. out_dtype is self._dtype (the model's
        # activation dtype), NOT the checkpoint's own stored scales dtype -- see the
        # class docstring for the real bug this was.
        gu_dense = _dequant(
            b["qweight_gate_up"][s_i], b["qzeros_gate_up"][s_i], b["scales_gate_up"][s_i],
            group_size=self._group_size, out_dtype=self._dtype,
        ).T.contiguous()
        dn_dense = _dequant(
            b["qweight_down"][s_i], b["qzeros_down"][s_i], b["scales_down"][s_i],
            group_size=self._group_size, out_dtype=self._dtype,
        ).T.contiguous()
        i = self._intermediate
        result = (gu_dense[0:i], gu_dense[i : 2 * i], dn_dense)
        self._cache[s_i] = result
        return result


__all__ = ["OffloadMoeCache", "SlotWeightAccessor", "_BANK_SCHEMAS"]
