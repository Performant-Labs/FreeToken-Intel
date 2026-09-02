"""Tests for the elastic-memory budget planner (issue #16, ``cache_budget.py``).

The planner splits the device's addressable VRAM (``total_vram * memory_ratio``)
between the MoE expert slot cache and the paged KV pool. The policy under test:

* **MoE-priority** -- the expert cache is the elastic allocation and takes all
  the budget the KV floor leaves;
* **KV-floor** -- the KV pool is always at least ``kv_reserve_tokens`` tokens so
  long context stays schedulable;
* **pre-alloc fit assert** -- if the budget cannot cover the KV floor *and* the
  minimum expert cache (``num_experts`` slots) at once, raise rather than
  over-commit and OOM at first prefill.

The planner is deliberately device-agnostic (it reasons about byte counts and
slot/token counts, never a live device), so these tests run in the torch-free
per-PR CPU venv -- they import only the planner, which has no torch dependency.
The torch / XPU integration (the engine actually building the pools at the
planned sizes) is exercised separately by the xpu-marked suite on the B70.
"""
from __future__ import annotations

import pytest

from freetoken.engine.cache_budget import (
    CacheBudget,
    MoeCachePlan,  # noqa: F401  (exported; the loader-facing plan is thin)
    _bytes_per_expert_slot,
    _bytes_per_kv_token,
    plan_cache_budget,
)

# A Qwen3-30B-A3B-shaped MoE (the B70 hero) at bf16, on a 32 GB card. The
# numbers are large enough that the slot/token arithmetic is unambiguous and
# small enough to stay in exact integer range.
GB = 1024**3
BASE = dict(
    total_vram_bytes=32 * GB,
    memory_ratio=0.9,
    kv_reserve_tokens=8192,
    num_experts=128,
    moe_intermediate_size=768,
    hidden_size=2048,
    num_moe_layers=48,
    num_layers=48,
    num_kv_heads=8,
    head_dim=128,
    dtype_bytes=2,
)
BUDGET = int(32 * GB * 0.9)
BYTES_PER_SLOT = _bytes_per_expert_slot(128, 768, 2048, 2)  # (2*I*H + H*I) * 2
BYTES_PER_KV_TOKEN = _bytes_per_kv_token(48, 8, 128, 2)  # 2 * L * kv * head * 2


def test_slot_and_token_byte_math():
    # One expert slot = gate_up [2I, H] + down [H, I], in dtype bytes.
    assert BYTES_PER_SLOT == (2 * 768 * 2048 + 2048 * 768) * 2
    # One KV token row = K and V buffers, [L, kv, head] each, in dtype bytes.
    assert BYTES_PER_KV_TOKEN == 2 * 48 * 8 * 128 * 2


def test_moe_priority_fills_budget_kv_keeps_floor():
    c = plan_cache_budget(**BASE)
    # The MoE cache is the priority: it soaks the budget minus the KV floor.
    assert c.moe_cache_size >= BASE["num_experts"]
    assert c.moe_cache_size >= 128 + 48  # at least the old layer-count heuristic
    # KV is at least its floor (the MoE surplus may have pushed it above it).
    assert c.kv_num_pages >= BASE["kv_reserve_tokens"]
    # The split never over-commits the budget.
    assert c.moe_cache_bytes + c.kv_bytes <= c.budget_bytes
    # The MoE cache is the big slice (the whole point of the policy).
    assert c.moe_cache_bytes > c.kv_bytes


def test_kv_is_floored_when_moe_cache_is_small():
    # A small budget where the MoE cache can't eat everything: KV absorbs the
    # surplus and the flag reports the pool is larger than its floor.
    c = plan_cache_budget(**{**BASE, "total_vram_bytes": 12 * GB, "num_experts": 16})
    assert c.kv_num_pages >= BASE["kv_reserve_tokens"]
    # With only 16 experts the MoE cache is small, so the KV pool grows past
    # the floor (the flag is False) rather than sitting exactly on it.
    assert c.kv_is_floored is False
    assert c.moe_cache_size >= 16


def test_fit_assert_raises_when_budget_too_small():
    # Budget can't cover the 8192-token KV floor plus 128 expert slots at once.
    with pytest.raises(ValueError, match="cannot cover"):
        plan_cache_budget(**{**BASE, "total_vram_bytes": 2 * GB, "memory_ratio": 0.9})


def test_fit_assert_honors_min_moe_cache_size_override():
    # Same budget, but force a *larger* minimum MoE cache so the assert trips.
    with pytest.raises(ValueError, match="cannot cover"):
        plan_cache_budget(**{**BASE, "total_vram_bytes": 4 * GB, "min_moe_cache_size": 5000})


def test_moe_fraction_caps_moe_share():
    full = plan_cache_budget(**BASE)
    capped = plan_cache_budget(**{**BASE, "moe_fraction": 0.3})
    # Capping the MoE share frees budget for KV: MoE shrinks, KV grows (but
    # never below its floor).
    assert capped.moe_cache_size <= full.moe_cache_size
    assert capped.kv_num_pages >= full.kv_num_pages
    # The MoE cache stays at or above its minimum even when the fraction is low.
    assert capped.moe_cache_size >= BASE["num_experts"]
    assert capped.kv_num_pages >= BASE["kv_reserve_tokens"]


def test_moe_fraction_out_of_range_raises():
    with pytest.raises(ValueError, match="moe_fraction"):
        plan_cache_budget(**{**BASE, "moe_fraction": 0.0})
    with pytest.raises(ValueError, match="moe_fraction"):
        plan_cache_budget(**{**BASE, "moe_fraction": 1.5})


@pytest.mark.parametrize("bad", [(-1.0, "memory_ratio"), (0.0, "memory_ratio"), (1.5, "memory_ratio")])
def test_bad_memory_ratio_raises(bad):
    with pytest.raises(ValueError, match=bad[1]):
        plan_cache_budget(**{**BASE, "memory_ratio": bad[0]})


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        plan_cache_budget(**{**BASE, "total_vram_bytes": 0})
    with pytest.raises(ValueError):
        plan_cache_budget(**{**BASE, "kv_reserve_tokens": 0})
    with pytest.raises(ValueError):
        plan_cache_budget(**{**BASE, "num_experts": 0})
    with pytest.raises(ValueError):
        plan_cache_budget(**{**BASE, "num_moe_layers": 0})


def test_cache_budget_is_frozen():
    c = plan_cache_budget(**BASE)
    with pytest.raises(Exception):
        c.moe_cache_size = 123  # frozen dataclass -> assignment is rejected
