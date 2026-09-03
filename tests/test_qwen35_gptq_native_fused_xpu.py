"""End-to-end: the offload forward's GPTQ-Int4 path actually dispatches to
the native fused-GEMM kernel on real XPU hardware, and produces the same
generated tokens as the dequant-then-matmul path (issue `moe-quant-banks-
native`, #139).

Reuses ``test_qwen35_gptq_e2e_loader.py``'s tiny synthetic GPTQ checkpoint
fixture (issue #138's own small-fabricated-checkpoint approach) -- this
proves the wiring end-to-end without the real 22.73GB Qwen3.5-35B-A3B-
GPTQ-Int4 checkpoint, which isn't loaded locally right now (RAM-constrained
box). Drives a full ``Engine`` (mirrors test_moe_fused_xpu.py's own
CPU-vs-XPU comparison pattern) rather than the lower-level ``_drive_prefill``
helper, which hardcodes a CPU-only KV cache. ``xpu``-marked: deselected on a
torch-free / no-XPU box (see ``conftest.py``); the CPU forward this fixture
already exercises in test_qwen35_gptq_e2e_loader.py is untouched by this
wiring (SlotWeightAccessor.expert_forward only takes the native path when
``cache.is_xpu``).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

XPU = pytest.mark.skipif(not torch.xpu.is_available(), reason="no XPU available")

from freetoken.core import Req, SamplingParams, reset_global_ctx
from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.engine.engine import Engine
from freetoken.kernel.triton import gptq_fused_linear

from tests.test_qwen35_gptq_e2e_loader import V, qwen35_gptq_ckpt  # noqa: F401

_PROMPT_IDS = [1, 2, 3, 4, 5]


@pytest.fixture(autouse=True)
def _clean_global_ctx():
    yield
    reset_global_ctx()


def _engine_config(model_path: str, device: torch.device) -> EngineConfig:
    return EngineConfig(
        model_path=model_path,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.float32,
        device=device,
        attention_backend="auto",
        moe_backend="offload",
        max_running_req=2,
        page_size=1,
        max_seq_len_override=32,
        num_page_override=64,
    )


def _add_prompt(engine: Engine, output_len: int) -> None:
    engine.add_request(
        Req(
            input_ids=list(_PROMPT_IDS),
            table_idx=0,
            cached_len=0,
            output_len=output_len,
            uid=0,
            sampling_params=SamplingParams(temperature=0.0, max_tokens=output_len),
            cache_handle=None,
        )
    )


@XPU
@pytest.mark.xpu
def test_native_fused_path_is_actually_dispatched(qwen35_gptq_ckpt):
    with patch(
        "freetoken.kernel.triton.gptq_fused_linear.fused_gptq_expert_forward",
        wraps=gptq_fused_linear.fused_gptq_expert_forward,
    ) as spy:
        engine = Engine(_engine_config(qwen35_gptq_ckpt, torch.device("xpu")))
        assert engine.model.moe_cache.is_xpu
        _add_prompt(engine, output_len=3)
        tokens = engine.generate()

    assert spy.call_count > 0, "native fused path was never dispatched -- SlotWeightAccessor.expert_forward fell through to the dequant fallback"
    assert len(tokens[0]) == 3
    assert all(0 <= t < V for t in tokens[0])


@XPU
@pytest.mark.xpu
def test_native_fused_path_matches_dequant_fallback_tokens(qwen35_gptq_ckpt):
    """Same checkpoint, same prompt: the native path and the dequant-then-
    matmul fallback must produce identical greedy tokens."""
    native_engine = Engine(_engine_config(qwen35_gptq_ckpt, torch.device("xpu")))
    _add_prompt(native_engine, output_len=4)
    native_tokens = native_engine.generate()
    reset_global_ctx()

    # SlotWeightAccessor.expert_forward imports prefer_fused_over_dequant
    # lazily from gptq_fused_linear on each call, so patching it there is
    # what actually forces the dequant fallback path here.
    with patch("freetoken.kernel.triton.gptq_fused_linear.prefer_fused_over_dequant", lambda m: False):
        fallback_engine = Engine(_engine_config(qwen35_gptq_ckpt, torch.device("xpu")))
        _add_prompt(fallback_engine, output_len=4)
        fallback_tokens = fallback_engine.generate()

    assert native_tokens == fallback_tokens, (
        f"native fused path diverged from the dequant fallback: {native_tokens} != {fallback_tokens}"
    )
