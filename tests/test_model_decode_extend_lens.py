"""Decode-phase forward does not need batch.extend_lens (issue #15's third blocker).

Reading int(extend_lens[i]) per request inside the decoder-layer loop is a
device->host sync (a plain Python int() call on a device tensor element) --
one of the things blocking a whole decode-step model.forward() from being
graph-capturable (found building engine-graph, #15 / XpuGraphRunner, #117).
A decode batch is uniform (the scheduler never mixes phases within one
batch -- see attention/sycl.py's #116 fix and its rationale), so every
request's new-token count is always exactly 1; forward() now special-cases
that and skips reading the tensor for decode.

These tests run on CPU (no XPU needed): the property under test -- decode
does not need batch.extend_lens at all -- is device-independent. Proven by
setting it to None (the "not needed" case) and confirming the forward still
succeeds and matches the tensor-populated case exactly.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from freetoken.core import Req, SamplingParams, reset_global_ctx
from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.engine.engine import Engine

from tests.test_moe_offload_forward import TINY_CONFIG, offload_ckpt  # noqa: F401

DEVICE = torch.device("cpu")


@pytest.fixture(autouse=True)
def _clean_global_ctx():
    yield
    reset_global_ctx()


def _engine_config(model_path: str) -> EngineConfig:
    return EngineConfig(
        model_path=model_path,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.float32,
        device=DEVICE,
        attention_backend="auto",
        moe_backend="fused",
        max_running_req=2,
        page_size=1,
        max_seq_len_override=32,
        num_page_override=64,
    )


def _run_one_decode_step(model_path: str, *, extend_lens_none: bool):
    engine = Engine(_engine_config(model_path))
    engine.add_request(
        Req(
            input_ids=[1, 2, 3],
            table_idx=0,
            cached_len=0,
            output_len=2,
            uid=0,
            sampling_params=SamplingParams(temperature=0.0, max_tokens=2),
            cache_handle=None,
        )
    )
    # Prefill (samples token 1 of 2), leaving one decode step still pending.
    engine.step()
    batch = engine.scheduler.schedule()
    assert batch is not None and batch.phase == "decode"

    engine.attn_backend.prepare_metadata(batch)
    req = batch.reqs[0]
    pos = req.device_len - 1
    batch.input_ids = torch.tensor([int(req.input_ids[-1])], dtype=torch.int64)
    batch.positions = torch.tensor([pos], dtype=torch.int64)
    batch.out_loc = torch.tensor([pos], dtype=torch.int64)
    batch.extend_lens = None if extend_lens_none else torch.tensor([1], dtype=torch.int64)

    with engine.ctx.forward_batch(batch):
        return engine.model(batch.input_ids, batch.positions, batch.out_loc).clone()


def test_decode_forward_does_not_require_extend_lens(offload_ckpt):
    """batch.extend_lens=None must not raise on a decode-phase forward, and
    must produce the exact same logits as the populated-tensor case -- the
    decode shortcut (ext=1 for every request, from batch.phase alone) never
    touches the tensor at all, so its presence/absence cannot matter."""
    with_tensor = _run_one_decode_step(offload_ckpt, extend_lens_none=False)
    reset_global_ctx()
    without_tensor = _run_one_decode_step(offload_ckpt, extend_lens_none=True)

    assert torch.equal(with_tensor, without_tensor), (
        "decode forward must be identical whether or not batch.extend_lens is populated"
    )
