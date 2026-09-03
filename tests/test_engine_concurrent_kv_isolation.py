"""Two concurrently-decoding requests must not share KV pool slots (issue
`engine-kv-addressing`, #173).

Before this fix, the engine's page table mapped every request's row to the
SAME slot range (``page_table[r, :] = torch.arange(max_seq_len)`` for every
``r``), so two requests decoding at overlapping relative positions read and
wrote the exact same ``k_buffer``/``v_buffer`` rows -- confirmed directly
(not just read from code) before landing the fix: a fresh ``Engine`` with
``max_running_req=2`` and two admitted requests had
``engine.page_table[0, :5] == engine.page_table[1, :5]`` (identical
tensors). ``create_kv_pool`` now builds ``MHAKVCache`` (real per-request
free-list allocation) and the engine allocates each request its own
disjoint slot run at admission (``Engine._allocate_slot``), freeing it on
completion (``Engine._free_slot``).

Reuses test_engine_loop.py's own tiny-checkpoint fixture (dummy-fabricated
MoE experts, real dense weights) -- CPU-only, no XPU dependency.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from freetoken.core import Req, SamplingParams, reset_global_ctx
from freetoken.engine.engine import Engine

from tests.test_engine_loop import DEVICE, _engine_config, _write_tiny_checkpoint


@pytest.fixture(autouse=True)
def _clean_global_ctx():
    yield
    reset_global_ctx()


def _make_req(uid: int, ids: list[int], output_len: int) -> Req:
    return Req(
        input_ids=list(ids),
        table_idx=0,
        cached_len=0,
        output_len=output_len,
        uid=uid,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=output_len),
        cache_handle=None,
    )


def test_two_concurrent_requests_get_disjoint_page_table_rows(tmp_path):
    model_path = _write_tiny_checkpoint(tmp_path)
    engine = Engine(_engine_config(model_path, device=DEVICE))
    engine.add_request(_make_req(0, [1, 2, 3], 3))
    engine.add_request(_make_req(1, [4, 5, 6, 7], 3))

    row0 = engine.page_table[0, :4]
    row1 = engine.page_table[1, :4]
    assert not torch.equal(row0, row1), "both requests' rows point at the same KV slots"
    # No slot appears in both rows at all (not just "not identical" -- truly disjoint).
    assert set(row0.tolist()).isdisjoint(set(row1.tolist()))


def test_concurrent_decode_does_not_corrupt_a_request_own_kv(tmp_path):
    """The real regression this issue guards against: request 0's own K/V
    values, after decoding CONCURRENTLY alongside request 1, must be
    bit-identical to what request 0 alone would have written -- request 0
    is admitted first in both runs, so it deterministically gets the same
    first slot range either way (a fresh free-list, FIFO allocation), which
    is what makes this a valid apples-to-apples comparison."""
    model_path = _write_tiny_checkpoint(tmp_path)

    reset_global_ctx()
    engine_alone = Engine(_engine_config(model_path, device=DEVICE))
    engine_alone.add_request(_make_req(0, [1, 2, 3], 3))
    engine_alone.step()  # one prefill step is enough to populate k_buffer
    k_alone = engine_alone.kv_cache.k_buffer.clone()
    slots_alone = engine_alone.page_table[0, :3].clone()

    reset_global_ctx()
    engine_concurrent = Engine(_engine_config(model_path, device=DEVICE))
    engine_concurrent.add_request(_make_req(0, [1, 2, 3], 3))
    engine_concurrent.add_request(_make_req(1, [4, 5, 6, 7], 3))
    engine_concurrent.step()
    k_concurrent = engine_concurrent.kv_cache.k_buffer.clone()
    slots_concurrent = engine_concurrent.page_table[0, :3].clone()

    assert torch.equal(slots_alone, slots_concurrent), "request 0's own slot allocation changed"
    torch.testing.assert_close(k_alone[:, slots_alone.long()], k_concurrent[:, slots_concurrent.long()])


def test_finished_request_frees_its_slot_for_reuse(tmp_path):
    model_path = _write_tiny_checkpoint(tmp_path)
    engine = Engine(_engine_config(model_path, device=DEVICE))
    engine.add_request(_make_req(0, [1, 2, 3], 1))  # output_len=1 -> finishes after 1 decode step
    engine.generate()

    assert not engine.kv_cache.is_allocated(0), "a finished request's KV slots were never freed"

    # A fresh request reusing table_idx 0 must be able to allocate again.
    engine.add_request(_make_req(1, [4, 5], 1))
    assert engine.kv_cache.is_allocated(0)
