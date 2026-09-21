"""Plan the elastic split of the B70's VRAM between the MoE expert slot cache
and the paged KV pool (issue #16, ``elastic-memory``).

Upstream NVIDIA path: python/freetoken/engine/cache_budget.py

Why it exists
-------------
ADR 0002 keeps the MoE experts in *host* RAM and gives the XPU only a small
fixed pool of "slots" (``cache_size``), streaming in the routed experts on
demand. But that pool is XPU memory, and it shares the B70's 32 GB with the
paged KV pool, the dense weights, and the runtime. The loader sizes the slot
pool off the *layer count* (``num_experts + max(2, num_moe)``) and the engine
sizes the KV pool off ``max_running_req * max_seq_len`` -- neither looks at how
much VRAM is actually free, so the two pools can together over-commit the card
(or leave it half-empty) with no signal.

This module is the single place that decides how the *free* VRAM is divided:

* ``memory_ratio`` of total VRAM is the addressable budget (the headroom the
  OS / runtime keeps for itself).
* Within that budget, **the MoE expert cache is prioritized** -- grow it as
  large as the budget allows -- and the **KV pool is floored** at
  ``kv_reserve_tokens`` tokens so that long-context requests can still be
  scheduled. When the budget cannot cover "a full KV floor *plus* a minimum
  useful expert cache", the fit assert below raises rather than silently
  over-allocating and OOMing at first prefill.

The planner is deliberately **device-agnostic**: it reasons about byte counts
and slot/token *counts*, never about a live device, so it is unit-testable on a
CPU-only box (the CI non-xpu suite) and only *reads* total VRAM when a real
XPU is present. The engine applies the returned counts (re)building the pools
-- which is what makes the split "elastic": it can be re-planned and the pools
rebuilt without reloading the (host-resident) weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "CacheBudget",
    "MoeCachePlan",
    "plan_cache_budget",
    "resolve_served_context_len",
    "check_pinned_kv_fit",
]


@dataclass(frozen=True)
class CacheBudget:
    """The resolved split of the addressable VRAM between the two pools.

    Attributes are *counts* (the unit each pool is allocated in), not byte
    sizes: ``moe_cache_size`` is the number of expert slots and ``kv_num_pages``
    is the number of KV pool pages. The ``*_bytes`` fields are the budget the
    planner reasoned with, exposed for logging / the fit assert.
    """

    # The addressable VRAM budget (bytes): total_vram * memory_ratio.
    budget_bytes: int
    # The two allocations that consume that budget (bytes).
    moe_cache_bytes: int
    kv_bytes: int
    # What each allocation buys, in the pool's own unit.
    moe_cache_size: int  # expert slots
    kv_num_pages: int  # KV pool pages (== tokens when page_size == 1)
    # True when the KV pool was set to the operator's reserve floor (the budget
    # could not afford more KV after the MoE cache took its share).
    kv_is_floored: bool = field(default=False, compare=False)


@dataclass(frozen=True)
class MoeCachePlan:
    """The MoE expert-cache side of the split (what the loader needs)."""

    cache_size: int
    # Bytes the slot pool will occupy, for the engine's fit assert.
    bytes_per_slot: int
    total_bytes: int


def _bytes_per_expert_slot(
    num_experts: int,
    moe_intermediate_size: int,
    hidden_size: int,
    dtype_bytes: int,
) -> int:
    """Bytes for ONE expert slot in the bf16 bank schema (ADR 0002).

    A slot holds one expert's two fused projections, laid out exactly as the
    loader's banks: ``gate_up [2*moe_intermediate, hidden]`` and
    ``down [hidden, moe_intermediate]``. (``num_experts`` is irrelevant here --
    a slot is one expert, and the pool's ``cache_size`` counts slots.)
    """
    if moe_intermediate_size <= 0 or hidden_size <= 0 or dtype_bytes <= 0:
        raise ValueError(
            "plan_cache_budget needs positive moe_intermediate_size / hidden_size / dtype_bytes"
        )
    gate_up = 2 * moe_intermediate_size * hidden_size
    down = hidden_size * moe_intermediate_size
    return (gate_up + down) * dtype_bytes


def _bytes_per_kv_token(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
) -> int:
    """Bytes for ONE token's KV row across every layer (the pool's row shape).

    The pool stores ``[num_layers, num_pages, num_kv_heads, head_dim]`` *per
    buffer* (K and V), one row per token, so a token costs
    ``2 * num_layers * num_kv_heads * head_dim * dtype_bytes``.
    """
    if num_layers <= 0 or num_kv_heads <= 0 or head_dim <= 0 or dtype_bytes <= 0:
        raise ValueError(
            "plan_cache_budget needs positive num_layers / num_kv_heads / head_dim / dtype_bytes"
        )
    return 2 * num_layers * num_kv_heads * head_dim * dtype_bytes


def plan_cache_budget(
    *,
    total_vram_bytes: int,
    memory_ratio: float,
    kv_reserve_tokens: int,
    num_experts: int,
    moe_intermediate_size: int,
    hidden_size: int,
    num_moe_layers: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
    min_moe_cache_size: Optional[int] = None,
    moe_fraction: Optional[float] = None,
    bytes_per_slot_override: Optional[int] = None,
) -> CacheBudget:
    """Split the addressable VRAM into a MoE expert cache and a KV pool.

    Policy (MoE-priority, KV-floor, pre-alloc fit assert):

    1. ``budget = total_vram_bytes * memory_ratio`` (both floors inclusive of
       this -- the ratio already reserves the OS/runtime headroom).
    2. Reserve the KV **floor** first: ``kv_reserve_tokens`` tokens, so long
       context is always schedulable. This is a *floor*, not a cap.
    3. Give the MoE expert cache everything left over in the budget (it is the
       elastic, prioritized allocation): as many full slots as fit, never below
       ``min_moe_cache_size`` (default: ``num_experts``, the minimum that can
       materialize a whole MoE layer). When ``moe_fraction`` is given the MoE
       share is instead capped to ``moe_fraction * budget`` (the rest of the
       budget goes to KV), so the operator can steer the split without giving up
       the KV floor.
    4. Whatever bytes the MoE cache does not consume returns to the KV pool, so
       the KV pool is at least its floor (often more, when the MoE cache is
       small).
    5. **Fit assert**: if the budget cannot cover the KV floor *and* the minimum
       MoE cache at the same time, raise -- the caller should shrink
       ``kv_reserve_tokens`` / lower ``memory_ratio`` / pick a smaller model
       rather than over-commit and OOM at first prefill.

    Args:
        total_vram_bytes: the device's total VRAM (bytes).
        memory_ratio: fraction of total VRAM to treat as addressable (0,1].
        kv_reserve_tokens: the KV pool floor, in tokens (== pages at
            page_size 1, which is the reference layout).
        num_experts: experts per MoE layer (``E``); bounds the minimum cache.
        moe_intermediate_size: the MoE FFN intermediate size (``I``).
        hidden_size: the model hidden size (``H``).
        num_moe_layers: number of MoE layers (the cache's layer count).
        num_layers: total decoder layers (the KV pool's layer count).
        num_kv_heads: KV (GQA) heads per layer.
        head_dim: per-head dim.
        dtype_bytes: bytes per element (2 for bf16/fp16, 4 for fp32). Ignored
            when ``bytes_per_slot_override`` is given.
        min_moe_cache_size: override the minimum slot count (default ``E``).
        bytes_per_slot_override: use this exact per-slot byte count instead
            of ``_bytes_per_expert_slot``'s bf16 ``(gate_up + down) *
            dtype_bytes`` formula (issue #16 / #136) -- for a non-bf16 bank
            schema (e.g. a GPTQ-Int4-packed offload cache), whose real
            per-slot footprint is a different shape the single-scalar
            ``dtype_bytes`` formula cannot express. See
            ``freetoken.moe.offload_cache.gptq_int4_bytes_per_expert_slot``
            for the gptq_int4 schema's real byte-size helper -- this module
            stays torch-free, so it does not import that helper itself; the
            caller computes it and passes the result in here.

    Returns:
        A :class:`CacheBudget` with the resolved counts.

    Raises:
        ValueError: on a non-positive budget, a ratio outside (0,1], or when the
            budget cannot cover both the KV floor and the minimum MoE cache.
    """
    if memory_ratio <= 0 or memory_ratio > 1:
        raise ValueError(f"memory_ratio must be in (0, 1], got {memory_ratio}")
    if total_vram_bytes <= 0:
        raise ValueError(f"total_vram_bytes must be positive, got {total_vram_bytes}")
    if kv_reserve_tokens <= 0:
        raise ValueError(f"kv_reserve_tokens must be positive, got {kv_reserve_tokens}")
    if num_experts <= 0 or num_moe_layers <= 0:
        raise ValueError("plan_cache_budget is for MoE models (num_experts/num_moe_layers > 0)")
    if moe_fraction is not None and (moe_fraction <= 0 or moe_fraction > 1):
        raise ValueError(f"moe_fraction must be in (0, 1] when set, got {moe_fraction}")

    budget = int(total_vram_bytes * memory_ratio)

    if bytes_per_slot_override is not None:
        if bytes_per_slot_override <= 0:
            raise ValueError(f"bytes_per_slot_override must be positive, got {bytes_per_slot_override}")
        bytes_per_slot = bytes_per_slot_override
    else:
        bytes_per_slot = _bytes_per_expert_slot(
            num_experts, moe_intermediate_size, hidden_size, dtype_bytes
        )
    bytes_per_kv_token = _bytes_per_kv_token(num_layers, num_kv_heads, head_dim, dtype_bytes)
    if bytes_per_kv_token <= 0:
        raise ValueError("bytes_per_kv_token computed to zero")

    kv_floor_bytes = kv_reserve_tokens * bytes_per_kv_token
    min_slots = min_moe_cache_size if min_moe_cache_size is not None else num_experts
    min_moe_bytes = min_slots * bytes_per_slot

    # Pre-alloc fit assert (policy step 5): the budget must cover BOTH the KV
    # floor and the minimum MoE cache. Below that, the card is over-committed.
    if budget < kv_floor_bytes + min_moe_bytes:
        raise ValueError(
            "VRAM budget cannot cover the MoE cache floor and the KV reserve "
            f"together: budget {budget} bytes < KV floor {kv_floor_bytes} "
            f"+ min MoE cache ({min_slots} slots) {min_moe_bytes}. Lower "
            f"kv_reserve_tokens ({kv_reserve_tokens}), lower memory_ratio "
            f"({memory_ratio}), or use a smaller model."
        )

    # KV floor first (always schedulable), then the MoE cache takes the rest of
    # the budget (the prioritized allocation), then any MoE surplus returns to KV.
    kv_bytes = kv_floor_bytes
    moe_available = budget - kv_floor_bytes
    # An operator ``moe_fraction`` caps the MoE share to a slice of the budget;
    # without it the MoE cache is the pure priority and takes everything the KV
    # floor leaves.
    if moe_fraction is not None:
        moe_available = min(moe_available, int(budget * moe_fraction))
    moe_cache_size = max(min_slots, moe_available // bytes_per_slot)
    moe_bytes = moe_cache_size * bytes_per_slot  # round down to whole slots
    kv_bytes = budget - moe_bytes  # MoE surplus (or the unused fraction) -> KV
    kv_num_pages = max(kv_reserve_tokens, kv_bytes // bytes_per_kv_token)

    return CacheBudget(
        budget_bytes=budget,
        moe_cache_bytes=moe_bytes,
        kv_bytes=kv_bytes,
        moe_cache_size=moe_cache_size,
        kv_num_pages=kv_num_pages,
        kv_is_floored=kv_num_pages == kv_reserve_tokens,
    )


def resolve_served_context_len(
    *,
    max_seq_len: int,
    planned_kv_pages: int | None,
) -> tuple[int, str | None]:
    """Cap the served context to what the auto-planned KV pool can actually admit.

    Admission (``Engine._allocate_slot``) allocates a request's FULL ``max_seq_len``
    up front, but the auto planner (:func:`plan_cache_budget`) can return FEWER KV
    pages than the checkpoint's ``max_position_embeddings`` whenever the MoE cache
    takes priority (issue #246): the engine then plans, say, 8489 pages, the first
    ``add_request`` asks for 40960, and every request dies with "KV pool full".
    This is the single place that reconciles the two -- the engine applies the
    returned cap to its pool / page table / ``self.max_seq_len`` and the server
    reports the same number as ``max_model_len`` on ``/v1/models``.

    ``planned_kv_pages is None`` means "no auto plan" (a pinned ``--moe-cache-size``
    or a dense model): the conventional pool is ``max_running_req * max_seq_len``
    which always covers the demand, so no cap. The planned count is what the pool
    allocates INCLUDING the engine's +1 slack page minus MHAKVCache's reserved
    slot 0, so capping to exactly ``planned_kv_pages`` leaves admission a exact fit.

    Returns:
        ``(effective_max_seq_len, cap_reason)`` -- ``cap_reason`` is ``None`` when
        the checkpoint's full context is servable, else a loud, operator-actionable
        explanation of the cap.
    """
    if planned_kv_pages is None or planned_kv_pages >= max_seq_len:
        return max_seq_len, None
    reason = (
        f"auto-planned KV pool has {planned_kv_pages} pages but the checkpoint "
        f"needs {max_seq_len} per request; capping the served context "
        f"(max_model_len) to {planned_kv_pages} tokens. Raise --kv-reserve-tokens, "
        f"pass a smaller --max-model-len, or pin --moe-cache-size to take the "
        f"conventional full-context pool."
    )
    return planned_kv_pages, reason


def check_pinned_kv_fit(
    *,
    total_vram_bytes: Optional[int],
    memory_ratio: float,
    moe_cache_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
    requested_num_pages: int,
    moe_intermediate_size: int = 0,
    hidden_size: int = 0,
    bytes_per_slot_override: Optional[int] = None,
    reserved_pages: int = 0,
) -> tuple[int, Optional[str]]:
    """Cap the pinned (non-auto) KV pool to what actually fits in VRAM.

    Issue #260: the pinned path (``planned_num_pages is None``, i.e. the
    operator set ``--moe-cache-size`` or left everything default so
    ``moe_cache_auto`` never ran) has always sized the KV pool as
    ``max_running_req * max_seq_len`` on the claim that this is "always >=
    the demand" -- but unlike :func:`plan_cache_budget`'s auto path, that
    formula was never checked against actual VRAM. If the pinned MoE cache
    plus the conventional KV formula together don't fit, the failure mode is
    a raw allocator OOM inside ``create_kv_pool``, not a loud, graceful cap.

    This is the pinned-mode equivalent of the auto path's fit assert: it reads
    the same total-VRAM query the auto planner uses (passed in by the
    caller -- this module stays device-agnostic and torch-free, see the
    module docstring), subtracts the operator's own pinned MoE cache
    footprint, and caps ``requested_num_pages`` down to whatever whole KV
    pages the remaining budget can hold, with a loud, operator-actionable
    reason -- mirroring :func:`resolve_served_context_len`'s cap-and-log
    treatment for the auto path.

    ``total_vram_bytes is None`` means there is no live device to measure
    against (e.g. a CPU test/dev box, or a dense/no-XPU run): nothing to
    check, so the request is returned unchanged and the caller keeps the
    conventional formula exactly as before.

    Args:
        total_vram_bytes: the device's total VRAM (bytes), or ``None`` when
            there is no live device to query.
        memory_ratio: fraction of total VRAM treated as addressable (0, 1] --
            the same knob :func:`plan_cache_budget` uses.
        moe_cache_size: the MoE expert cache's actual slot count in this run
            (the operator's pin, or the loader's own conventional default when
            unpinned) -- 0 for a dense (non-MoE) model.
        num_layers, num_kv_heads, head_dim, dtype_bytes: the KV pool's row
            shape, same meaning as in :func:`plan_cache_budget`.
        requested_num_pages: the conventional ``max_running_req * max_seq_len``
            page count (plus any operator ``--num-page-override``) the engine
            was about to allocate, unchecked.
        moe_intermediate_size, hidden_size: needed to size one expert slot's
            bytes (ignored when ``moe_cache_size`` is 0, or when
            ``bytes_per_slot_override`` is given).
        bytes_per_slot_override: use this exact per-slot byte count instead of
            the bf16 ``(gate_up + down) * dtype_bytes`` formula -- same escape
            hatch as :func:`plan_cache_budget` for a non-bf16 bank schema.
        reserved_pages: additional whole KV pages the caller will allocate on
            top of whatever this function returns, that must ALSO fit in the
            same VRAM budget -- e.g. ``Engine``'s own "+1 page of slack" for
            ``MHAKVCache``'s reserved slot 0 (issue #173), added
            *unconditionally* after this check runs. Without this, a fit check
            against ``requested_num_pages`` alone could return a cap that
            itself fits exactly, only for the caller's own +1 slack page to
            push the real allocation back over budget -- the same
            "conventional formula asserted safe but never actually checked"
            failure this whole function exists to close (issue #260's own
            review comment on the pinned-fit PR). Reflected in the fit
            arithmetic (``requested_num_pages + reserved_pages`` must fit) but
            NOT in the returned page count or the reason string (both still
            describe the caller's own conventional demand) -- the caller adds
            ``reserved_pages`` back on top, exactly as it always did.

    Returns:
        ``(effective_num_pages, cap_reason)`` -- ``cap_reason`` is ``None``
        when the conventional pool actually fits, else a loud explanation of
        the cap.

    Raises:
        ValueError: on a non-positive ``requested_num_pages`` / ``memory_ratio``,
            or when even a single KV page cannot fit alongside the pinned MoE
            cache (the pinned-mode equivalent of :func:`plan_cache_budget`'s
            fit assert -- there is nothing sane to cap to at that point).
    """
    if total_vram_bytes is None:
        return requested_num_pages, None
    if memory_ratio <= 0 or memory_ratio > 1:
        raise ValueError(f"memory_ratio must be in (0, 1], got {memory_ratio}")
    if requested_num_pages <= 0:
        raise ValueError(f"requested_num_pages must be positive, got {requested_num_pages}")
    if reserved_pages < 0:
        raise ValueError(f"reserved_pages must be non-negative, got {reserved_pages}")

    budget = int(total_vram_bytes * memory_ratio)

    moe_bytes = 0
    if moe_cache_size > 0:
        if bytes_per_slot_override is not None:
            if bytes_per_slot_override <= 0:
                raise ValueError(
                    f"bytes_per_slot_override must be positive, got {bytes_per_slot_override}"
                )
            bytes_per_slot = bytes_per_slot_override
        else:
            bytes_per_slot = _bytes_per_expert_slot(
                moe_cache_size, moe_intermediate_size, hidden_size, dtype_bytes
            )
        moe_bytes = moe_cache_size * bytes_per_slot

    bytes_per_kv_token = _bytes_per_kv_token(num_layers, num_kv_heads, head_dim, dtype_bytes)
    kv_budget = budget - moe_bytes
    # fit_pages is the TOTAL whole KV pages the remaining budget can hold --
    # including the reserved_pages the caller will add on top of whatever we
    # return, so the two together never exceed budget (see reserved_pages'
    # own docstring above for why this matters).
    fit_pages = max(0, kv_budget) // bytes_per_kv_token
    fit_pages_for_caller = fit_pages - reserved_pages

    if fit_pages_for_caller <= 0:
        raise ValueError(
            "VRAM budget cannot fit even one KV page alongside the pinned MoE "
            f"cache: budget {budget} bytes, pinned MoE cache {moe_cache_size} "
            f"slots ({moe_bytes} bytes), reserved_pages {reserved_pages}. Lower "
            f"--moe-cache-size, lower --max-running-req / --max-model-len, or "
            f"raise --memory-ratio."
        )

    if fit_pages_for_caller >= requested_num_pages:
        return requested_num_pages, None

    reason = (
        f"pinned KV pool would need {requested_num_pages} pages "
        f"(max_running_req * max_seq_len) but only {fit_pages_for_caller} pages "
        f"fit in the addressable VRAM budget ({budget} bytes) once the pinned "
        f"MoE cache ({moe_cache_size} slots, {moe_bytes} bytes){' and ' + str(reserved_pages) + ' reserved slack page(s)' if reserved_pages else ''} "
        f"are set aside; capping the KV pool to {fit_pages_for_caller} pages "
        f"instead of OOMing at first allocation. Lower --max-running-req, pass "
        f"a smaller --max-model-len, lower --moe-cache-size, or use "
        f"--moe-cache-auto to let the planner balance the split."
    )
    return fit_pages_for_caller, reason
