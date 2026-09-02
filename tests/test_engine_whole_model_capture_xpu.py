"""XPU test: a whole decode-step model.forward() is graph-capturable.

The final piece of issue engine-graph (#15)'s own literal accept criterion
("captured decode replay matches eager decode numerically on a tiny model"):
#117 built XpuGraphRunner, #118/#119 made both attention backends
individually capturable, #122 removed the decoder-layer loop's per-request
extend_lens sync, and #123 gave the fused MoE path a dense (mask-based, no
nonzero(), no host sync) routing while capturing. This test is the
end-to-end proof all four combine correctly: a real decode-step
model.forward() (SYCL attention + fused/in-VRAM MoE, the one MoE backend
that's graph-capturable at all -- offload/cpu/hybrid stay fundamentally
dynamic, see graph.py) is captured once and replayed, matching eager
bit-exact.

``xpu``-marked: deselected on a torch-free / no-XPU box (see ``conftest.py``).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from freetoken.engine.graph import XpuGraphRunner

from tests.test_attention_sycl import _write_tiny_checkpoint, _add_prompt

XPU = pytest.mark.skipif(not torch.xpu.is_available(), reason="no XPU available")


@XPU
@pytest.mark.xpu
def test_whole_decode_step_model_forward_is_graph_capturable(tmp_path):
    from freetoken.core import reset_global_ctx
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig
    from freetoken.engine.engine import Engine

    dev = torch.device("xpu")
    reset_global_ctx()
    model_path = _write_tiny_checkpoint(tmp_path, random=True)
    config = EngineConfig(
        model_path=model_path,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.float32,
        device=dev,
        attention_backend="sycl",
        max_running_req=2,
        page_size=1,
        max_seq_len_override=32,
        use_dummy_weight=False,
        moe_backend="fused",  # the only graph-capturable MoE path
    )
    engine = Engine(config)
    _add_prompt(engine, output_len=5, prompt_ids=[1, 2, 3])

    # Drive to a stable decode state (prefill + a couple of decode steps).
    for _ in range(3):
        engine.step()

    batch = engine.scheduler.schedule()
    assert batch is not None and batch.phase == "decode"
    req = batch.reqs[0]
    pos = req.device_len - 1

    static_input_ids = torch.tensor([int(req.input_ids[-1])], dtype=torch.int64, device=dev)
    static_positions = torch.tensor([pos], dtype=torch.int64, device=dev)
    static_out_loc = torch.tensor([pos], dtype=torch.int64, device=dev)
    batch.input_ids = static_input_ids
    batch.positions = static_positions
    batch.out_loc = static_out_loc
    batch.extend_lens = torch.tensor([1], dtype=torch.int64, device=dev)

    try:
        with engine.ctx.forward_batch(batch):
            engine.attn_backend.prepare_metadata(batch)
            engine.attn_backend.prepare_for_capture(batch)
            engine.model._capturing = True

            eager_logits = engine.model(static_input_ids, static_positions, static_out_loc).clone()
            torch.xpu.synchronize()

            static_logits_buf = torch.empty_like(eager_logits)

            def _fn():
                logits = engine.model(static_input_ids, static_positions, static_out_loc)
                static_logits_buf.copy_(logits)

            runner = XpuGraphRunner(warmup_iters=3)
            runner.capture(_fn)  # must not raise -- this is the point of #15/#123

            runner.replay()
            torch.xpu.synchronize()
            diff = (static_logits_buf - eager_logits).abs().max().item()
            assert diff < 1e-4, f"whole-model captured/replayed logits diverged from eager: {diff}"

            # A second replay (idempotent, same fixed inputs) must still match.
            runner.replay()
            torch.xpu.synchronize()
            diff2 = (static_logits_buf - eager_logits).abs().max().item()
            assert diff2 < 1e-4
    finally:
        engine.attn_backend.reset_capture()
        engine.model._capturing = False
        reset_global_ctx()
