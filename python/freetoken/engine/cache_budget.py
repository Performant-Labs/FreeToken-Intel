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

__all__ = ["CacheBudget", "plan_cache_budget", "MoeCachePlan"]


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
        dtype_bytes: bytes per element (2 for bf16/fp16, 4 for fp32).
        min_moe_cache_size: override the minimum slot count (default ``E``).

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
