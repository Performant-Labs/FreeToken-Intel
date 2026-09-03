"""LinearStatePool: stacked-tensor GDN state pool with ping-pong COW slots
(issue `semantic-cache-linear-pool`, #170, part of the `semantic-cache`
epic, #32).

CPU-safe (dual-venv contract). Small synthetic dims, no real model.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from freetoken.kvcache.linear_state_pool import LinearStatePool, linear_state_bytes_per_req, ssm_state_dtype

NUM_LAYERS, NUM_K_HEADS, NUM_V_HEADS, K_DIM, V_DIM, CONV_KERNEL = 2, 4, 4, 8, 8, 4


def _pool(num_slots: int = 8, tp_size: int = 1, layer_ids=None, dtype=torch.float32) -> LinearStatePool:
    return LinearStatePool(
        num_layers=NUM_LAYERS,
        num_key_heads=NUM_K_HEADS,
        num_value_heads=NUM_V_HEADS,
        key_head_dim=K_DIM,
        value_head_dim=V_DIM,
        conv_kernel_dim=CONV_KERNEL,
        num_slots=num_slots,
        dtype=dtype,
        device=torch.device("cpu"),
        layer_ids=layer_ids,
        tp_size=tp_size,
    )


def test_slot_0_is_reserved_padding_not_allocatable():
    pool = _pool(num_slots=4)
    assert pool.num_free_slots == 3
    slots = pool.alloc(3)
    assert 0 not in slots
    assert pool.num_free_slots == 0


def test_alloc_raises_when_exhausted():
    pool = _pool(num_slots=2)  # 1 allocatable slot (slot 0 reserved)
    pool.alloc(1)
    with pytest.raises(RuntimeError, match="exhausted"):
        pool.alloc(1)


def test_copy_from_is_a_real_independent_copy_not_a_view():
    """The whole point of ping-pong COW: freezing a live slot into a track
    slot must survive the live slot later being overwritten."""
    pool = _pool()
    src, dst = pool.alloc(2)
    pool.conv_state(0, src)[:] = 5.0
    pool.recurrent_state(0, src)[:] = 7.0

    pool.copy_from(src, dst)
    torch.testing.assert_close(pool.conv_state(0, dst), pool.conv_state(0, src))
    torch.testing.assert_close(pool.recurrent_state(0, dst), pool.recurrent_state(0, src))

    # Overwrite the source AFTER the copy -- the destination must be unaffected.
    pool.conv_state(0, src)[:] = 99.0
    assert pool.conv_state(0, dst).max().item() == 5.0
    assert pool.conv_state(0, src).max().item() == 99.0


def test_copy_from_covers_every_layer():
    pool = _pool()
    src, dst = pool.alloc(2)
    for layer_id in (0, 1):
        pool.conv_state(layer_id, src)[:] = 3.0
        pool.recurrent_state(layer_id, src)[:] = 4.0
    pool.copy_from(src, dst)
    for layer_id in (0, 1):
        assert pool.conv_state(layer_id, dst).max().item() == 3.0
        assert pool.recurrent_state(layer_id, dst).max().item() == 4.0


def test_free_returns_slots_to_the_free_list():
    pool = _pool(num_slots=4)
    slots = pool.alloc(3)
    assert pool.num_free_slots == 0
    pool.free(slots)
    assert pool.num_free_slots == 3
    # And they're allocatable again.
    pool.alloc(3)


def test_free_accepts_int_list_and_tensor():
    pool = _pool(num_slots=8)
    s = pool.alloc(1)[0]
    pool.free(s)  # bare int
    a, b = pool.alloc(2)
    pool.free([a, b])  # list
    c = pool.alloc(1)[0]
    pool.free(torch.tensor([c]))  # tensor
    assert pool.num_free_slots == 7


def test_clear_slots_zeros_state_across_all_layers():
    pool = _pool()
    slots = pool.alloc(2)
    for layer_id in (0, 1):
        pool.conv_state(layer_id, slots[0])[:] = 1.0
        pool.recurrent_state(layer_id, slots[0])[:] = 1.0
    pool.clear_slots(slots)
    for layer_id in (0, 1):
        assert pool.conv_state(layer_id, slots[0]).abs().max().item() == 0.0
        assert pool.recurrent_state(layer_id, slots[0]).abs().max().item() == 0.0


def test_reset_zeros_a_single_table_idx_slot():
    pool = _pool()
    slot = pool.alloc(1)[0]
    pool.conv_state(0, slot)[:] = 1.0
    pool.reset(slot)
    assert pool.conv_state(0, slot).abs().max().item() == 0.0


def test_reclaim_all_slots_restores_the_full_free_list():
    pool = _pool(num_slots=4)
    pool.alloc(3)
    assert pool.num_free_slots == 0
    pool.reclaim_all_slots()
    assert pool.num_free_slots == 3


def test_rebuild_preserves_object_identity_and_resizes():
    pool = _pool(num_slots=4)
    pool.rebuild(16)
    assert pool.num_slots == 16
    assert pool.num_free_slots == 15
    assert pool.conv_states.shape[1] == 16
    assert pool.recurrent_states.shape[1] == 16


def test_layer_ids_map_global_ids_to_local_tensor_index():
    pool = _pool(layer_ids=[3, 7])  # non-contiguous, interleaved with full-attention layers
    assert pool.is_linear_layer(3) and pool.is_linear_layer(7)
    assert not pool.is_linear_layer(0)
    assert pool.local_index(3) == 0
    assert pool.local_index(7) == 1
    assert pool.num_linear_layers == 2


def test_layer_ids_length_mismatch_raises():
    with pytest.raises(ValueError, match="layer_ids"):
        _pool(layer_ids=[0])  # NUM_LAYERS=2, only 1 id given


def test_tp_sharding_evenly_divides_head_counts():
    """tp_size=2 must halve the local head counts (and thus the conv/
    recurrent tensor sizes) relative to tp_size=1."""
    pool_tp1 = _pool(tp_size=1)
    pool_tp2 = _pool(tp_size=2)
    assert pool_tp1.conv_states.shape[2] == pool_tp2.conv_states.shape[2] * 2
    assert pool_tp1.recurrent_states.shape[2] == pool_tp2.recurrent_states.shape[2] * 2


def test_bytes_per_slot_matches_the_standalone_estimator():
    pool = _pool()
    estimated = linear_state_bytes_per_req(
        num_layers=NUM_LAYERS,
        num_key_heads=NUM_K_HEADS,
        num_value_heads=NUM_V_HEADS,
        key_head_dim=K_DIM,
        value_head_dim=V_DIM,
        conv_kernel_dim=CONV_KERNEL,
        tp_size=1,
        dtype=torch.float32,
    )
    assert pool.bytes_per_slot() == estimated


def test_recurrent_state_dtype_defaults_to_float32_regardless_of_conv_dtype():
    """ssm_state_dtype() (FREETOKEN_MAMBA_SSM_DTYPE, default fp32) governs
    the recurrent state independently of the conv state's own (model)
    dtype -- constructing with a bf16 conv dtype must not silently make the
    recurrent state bf16 too."""
    pool = _pool(dtype=torch.bfloat16)
    assert pool.conv_states.dtype == torch.bfloat16
    assert pool.recurrent_states.dtype == ssm_state_dtype() == torch.float32
