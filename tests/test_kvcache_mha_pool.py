"""Paged MHA/GQA pool: per-request page allocate/free + buffer-shape contract.

CPU-safe (the dual-venv contract allows torch in ``.venv-xpu``; the CPU venv
deselects via ``importorskip``). These tests lock in the two halves of issue
``kvcache`` acceptance that belong to the pool:

  * ``allocate`` / ``free`` hand out and reclaim ``page_size``-grained slot
    runs off a free-list, with the reserved slot 0 never handed out.
  * the ``k_buffer`` / ``v_buffer`` shape is the exact ``[L, S, H, D]`` the
    SYCL attention kernel indexes directly (a regression here silently breaks
    the B70 hero), and ``write_kv`` / ``read_kv`` round-trip through the page
    table.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from freetoken.kvcache import MHAKVCache, create_kv_pool  # noqa: E402


class _ModelConfig:
    """Minimal parsed-ModelConfig stand-in the pool's constructor reads."""

    num_key_value_heads = 2
    num_layers = 3
    num_attention_heads = 4
    hidden_size = 64
    head_dim = 16

    def __init__(self) -> None:
        self.num_kv_heads = self.num_key_value_heads


def _pool(num_pages: int = 8, page_size: int = 2):
    return MHAKVCache(
        _ModelConfig(),
        page_size=page_size,
        num_pages=num_pages,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_buffer_shape_contract():
    # [num_layers, num_slots, num_kv_heads, head_dim]; num_slots = pages * page_size
    pool = _pool(num_pages=8, page_size=2)
    assert tuple(pool.k_buffer.shape) == (3, 16, 2, 16)
    assert tuple(pool.v_buffer.shape) == (3, 16, 2, 16)


def test_reserve_slot_zero_and_allocatable_count():
    pool = _pool(num_pages=8, page_size=2)  # 16 slots, slot 0 reserved -> 15 allocatable
    assert pool.num_slots == 16
    assert pool.num_allocatable_slots == 15


def test_allocate_returns_contiguous_run_from_low():
    pool = _pool(num_pages=8, page_size=2)
    slots = pool.allocate(7, num_pages=2)
    assert slots.tolist() == [1, 2, 3, 4]
    assert slots.dtype == torch.int64
    assert pool.is_allocated(7)
    assert pool.num_allocatable_slots == 11


def test_free_returns_slots_to_pool():
    pool = _pool(num_pages=8, page_size=2)
    pool.allocate(7, num_pages=2)
    pool.free(7)
    assert not pool.is_allocated(7)
    assert pool.num_allocatable_slots == 15
    # free is idempotent
    pool.free(7)
    assert pool.num_allocatable_slots == 15


def test_realloc_after_free_reuses_slots():
    pool = _pool(num_pages=8, page_size=2)
    first = pool.allocate(1, num_pages=2).tolist()
    pool.free(1)
    second = pool.allocate(2, num_pages=2).tolist()
    assert first == second  # same contiguous run handed back


def test_allocate_rejects_double_alloc():
    pool = _pool(num_pages=8, page_size=2)
    pool.allocate(1, num_pages=2)
    with pytest.raises(ValueError):
        pool.allocate(1, num_pages=2)


def test_allocate_rejects_when_full():
    pool = _pool(num_pages=2, page_size=2)  # 4 slots, slot0 reserved -> 3 free
    with pytest.raises(RuntimeError):
        pool.allocate(1, num_pages=4)  # needs 8 slots, only 3 free


def test_page_size_granularity_boundary():
    # page_size=1 -> every token is its own page (the engine's token-granular mode)
    pool = _pool(num_pages=4, page_size=1)  # 4 slots, slot0 reserved -> 3 free
    slots = pool.allocate(0, num_pages=3)
    assert slots.tolist() == [1, 2, 3]
    assert pool.num_allocatable_slots == 0


def test_write_read_kv_roundtrip_through_page_table():
    pool = _pool(num_pages=4, page_size=2)  # 8 slots
    L, S, H, D = pool.num_layers, pool.num_slots, pool.num_kv_heads, pool.head_dim
    # Identity page table over 2 requests x max_seq_len rows (like the engine).
    page_table = torch.arange(S, dtype=torch.int64).reshape(1, S).repeat(2, 1)
    pool.attach_page_table(page_table)
    # The model hands k/v in head-major [H, num_tokens, D].
    num_tokens = 4
    k = torch.randn(H, num_tokens, D)
    v = torch.randn(H, num_tokens, D)
    out_loc = torch.arange(1, 1 + num_tokens, dtype=torch.int64)  # slots 1..4
    pool.write_kv(k, v, out_loc, layer_id=1)
    # Read back the same positions from layer 1 (row 0 of the page table).
    got_k, got_v = pool.read_kv(0, torch.arange(1, 1 + num_tokens), layer_id=1)
    exp_k = k.transpose(0, 1).reshape(num_tokens, H, D)
    exp_v = v.transpose(0, 1).reshape(num_tokens, H, D)
    assert torch.allclose(got_k, exp_k)
    assert torch.allclose(got_v, exp_v)
    # A different layer's slice is untouched (per-layer isolation).
    got_k2, _ = pool.read_kv(0, torch.arange(1, 1 + num_tokens), layer_id=0)
    assert torch.all(got_k2 == 0)


def test_create_kv_pool_factory_returns_mha_pool():
    # Issue `engine-kv-addressing` (#173): the flat identity pool's shared
    # page_table (every row -> the SAME slot range) let two requests
    # decoding concurrently at overlapping positions corrupt each other's
    # KV. create_kv_pool -- the engine's own factory -- now builds
    # MHAKVCache, which gives each request a real, disjoint slot run.
    pool = create_kv_pool(
        _ModelConfig(), page_size=2, num_pages=8, device=torch.device("cpu"), dtype=torch.float32
    )
    assert isinstance(pool, MHAKVCache)
