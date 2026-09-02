"""XPU tests: TritonAttentionBackend's fixed-shape (graph-capturable) KV read.

Issue attn-triton-fixed-kv (#118), a follow-up to engine-graph (#15):
``_attend_one`` normally reads exactly ``[0, written)`` keys, a shape that
grows every decode step -- incompatible with ``torch.xpu.graph()``, which
replays the identical kernel launches (identical tensor shapes) every time.
While ``self._capturing`` is armed (``prepare_for_capture``), it instead
reads a FIXED ``[0, max_seq_len)`` range with an extra ``keypos < written``
mask term, so the shape never changes.

Scope actually proven here: capture is possible (the shape-varying error
XpuGraphRunner would otherwise hit is gone) and the captured/replayed output
matches eager for the fixed state it was captured at. Genuine reuse across
DIFFERENT (growing) decode steps without recapturing needs `written` itself
to flow as a persistent tensor buffer (like the SYCL table's kv_len row),
not a Python int baked into the graph at capture time -- the same
persistent-buffer plumbing through the engine's step loop that #117
(XpuGraphRunner) and #118's own issue body already flag as separate,
larger follow-up work. Not attempted here.

``xpu``-marked: deselected on a torch-free / no-XPU box (see ``conftest.py``).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from freetoken.attention.triton import TritonAttentionBackend
from freetoken.engine.graph import XpuGraphRunner

XPU = pytest.mark.skipif(not torch.xpu.is_available(), reason="no XPU available")


class _FakeReq:
    def __init__(self, table_idx: int) -> None:
        self.table_idx = table_idx


class _FakePool:
    """A minimal read_kv-only KV pool: k/v laid out [max_seq_len, kv, D]."""

    def __init__(self, max_seq_len: int, kv: int, d: int, device: torch.device) -> None:
        self.k = torch.randn(max_seq_len, kv, d, device=device, dtype=torch.float32)
        self.v = torch.randn(max_seq_len, kv, d, device=device, dtype=torch.float32)

    def read_kv(self, table_idx, pos, layer_id):
        return self.k[pos], self.v[pos]


@XPU
@pytest.mark.xpu
def test_capture_time_fixed_shape_matches_eager_growing_slice():
    """A capture-armed decode step matches the ordinary (eager) growing-slice
    result for the exact state it was captured at, and XpuGraphRunner can
    actually capture it (the point of this fix: no shape-varying error)."""
    dev = torch.device("xpu")
    torch.manual_seed(0)

    qh_n, kv_n, d, max_seq_len = 4, 2, 8, 16
    written = 5  # this decode step's real history length (< max_seq_len)

    import freetoken.attention.triton as triton_mod

    monkeypatched_pool = _FakePool(max_seq_len, kv_n, d, dev)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(triton_mod, "_get_ctx", lambda: type("Ctx", (), {"kv_cache": monkeypatched_pool})())

    try:
        backend = TritonAttentionBackend(config=object())
        req = _FakeReq(table_idx=0)
        qh = torch.randn(qh_n, 1, d, device=dev, dtype=torch.float32)  # one new token
        q_pos = torch.tensor([written - 1], device=dev, dtype=torch.int64)
        repeat = qh_n // kv_n
        scale = 1.0 / (d ** 0.5)

        # Eager (growing-slice) reference: _capturing is False by default.
        assert backend._capturing is False
        eager_out = backend._attend_one(req, qh, q_pos, written, repeat, scale).clone()

        # Arm capture mode and confirm the fixed-shape path matches eager
        # exactly for this same state (same q/k/v, same written).
        backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=[1])
        backend.prepare_for_capture(batch=None)
        assert backend._capturing is True

        static_out = torch.empty_like(eager_out)

        def _fn():
            static_out.copy_(backend._attend_one(req, qh, q_pos, written, repeat, scale))

        runner = XpuGraphRunner(warmup_iters=3)
        runner.capture(_fn)  # must not raise (this is the actual fix under test)
        runner.replay()
        torch.xpu.synchronize()

        diff = (static_out - eager_out).abs().max().item()
        assert diff < 1e-4, f"captured/replayed output diverged from eager: {diff}"
    finally:
        monkeypatch.undo()


def test_reset_capture_clears_capturing_state():
    backend = TritonAttentionBackend(config=object())
    backend._capturing = True
    backend._graph_max_seq_len = 16
    backend.reset_capture()
    assert backend._capturing is False
    assert backend._graph_max_seq_len == 0
