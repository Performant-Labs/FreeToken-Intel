"""XPU tests: SyclAttentionBackend is graph-capturable (issue attn-sycl-graph-capture, #119).

A follow-up to engine-graph (#15): the SYCL kernel used to open its own
``sycl::queue`` internally and call ``torch.xpu.synchronize()`` before every
launch, so ``torch.xpu.graph()`` could neither see the kernel's work nor
tolerate the sync (a hard capture error: "wait cannot be called for a queue
which is recording to a command graph"). Fixed by threading the caller's
active SYCL queue (``torch.xpu.current_stream().sycl_queue``) through to the
kernel -- the SYCL/XPU analog of upstream's own CUDA kernels taking the
caller's ``cudaStream_t`` -- and skipping the host syncs while a capture is
armed (``prepare_for_capture``).

``xpu``-marked: deselected on a torch-free / no-XPU box (see ``conftest.py``).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from freetoken.engine.graph import XpuGraphRunner

from tests.test_attention_sycl import _write_tiny_checkpoint, _engine_config, _add_prompt

XPU = pytest.mark.skipif(not torch.xpu.is_available(), reason="no XPU available")


@XPU
@pytest.mark.xpu
def test_sycl_attention_forward_is_graph_capturable(tmp_path):
    """A real SyclAttentionBackend.forward() call, captured and replayed,
    reproduces eager exactly -- the queue this used to create itself made
    torch.xpu.graph() unable to see (or capture) the kernel's work at all;
    this is the fix, verified against the real kernel end to end."""
    from freetoken.core import reset_global_ctx
    from freetoken.engine.engine import Engine

    reset_global_ctx()
    model_path = _write_tiny_checkpoint(tmp_path, random=True)
    engine = Engine(_engine_config(model_path, device="xpu", attention_backend="sycl", dummy_weight=False))
    _add_prompt(engine, output_len=5, prompt_ids=[1, 2, 3])

    # Drive to a stable decode state (prefill + a couple of decode steps).
    for _ in range(3):
        engine.step()

    batch = engine.scheduler.schedule()
    assert batch is not None and batch.phase == "decode"
    req = batch.reqs[0]
    pos = req.device_len - 1
    dev = engine.device

    batch.input_ids = torch.tensor([int(req.input_ids[-1])], dtype=torch.int64, device=dev)
    batch.positions = torch.tensor([pos], dtype=torch.int64, device=dev)
    batch.out_loc = torch.tensor([pos], dtype=torch.int64, device=dev)
    batch.extend_lens = torch.tensor([1], dtype=torch.int64, device=dev)
    table_idx = req.table_idx

    try:
        with engine.ctx.forward_batch(batch):
            engine.attn_backend.prepare_metadata(batch)
            engine.attn_backend.prepare_for_capture(batch)
            assert engine.attn_backend._capturing is True

            layer0 = engine.model.layers[0]
            attn = layer0.self_attn
            static_hidden = torch.randn(1, engine.model.config.hidden_size, device=dev, dtype=torch.float32)
            static_normed = layer0.input_layernorm(static_hidden)

            eager_out = attn(static_normed, batch.positions, table_idx, engine.ctx, batch).clone()
            torch.xpu.synchronize()

            static_out_buf = torch.empty_like(eager_out)

            def _fn():
                o = attn(static_normed, batch.positions, table_idx, engine.ctx, batch)
                static_out_buf.copy_(o)

            runner = XpuGraphRunner(warmup_iters=3)
            runner.capture(_fn)  # must not raise -- this is the fix under test

            runner.replay()
            torch.xpu.synchronize()
            diff = (static_out_buf - eager_out).abs().max().item()
            assert diff < 1e-4, f"captured/replayed SYCL attention diverged from eager: {diff}"

            # A second replay (idempotent, same fixed inputs) must still match.
            runner.replay()
            torch.xpu.synchronize()
            diff2 = (static_out_buf - eager_out).abs().max().item()
            assert diff2 < 1e-4
    finally:
        engine.attn_backend.reset_capture()
        reset_global_ctx()
