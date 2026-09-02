"""XPU regression test: the in-VRAM ("fused") MoE forward on real XPU hardware.

``test_moe_offload_forward.py``'s in-VRAM *reference* runs on CPU only
(``DEVICE = torch.device("cpu")``) -- it exists to validate the *offload*
path, not the in-VRAM path itself, so it never exercised ``_Qwen3MoE.forward``'s
resident-experts branch on an XPU tensor. That branch used device-side
boolean-mask indexing (``flat[sel]`` / ``out[sel] +=``, backed by
``torch.nonzero``) which silently returns an EMPTY result on this torch/XPU
build for a multi-token step (multiple experts routing different tokens) --
``sel.sum()`` / ``sel.tolist()`` show the mask is correct, but the indexed
gather/scatter through ``nonzero()`` is not, and the forward raises inside the
expert matmul (``numel: integer multiplication overflow``) once the *bogus*
empty selection reaches a downstream shape check. A prefill (multiple new
tokens in one step) exercises this; a bs=1 decode-only run does not, which is
why the existing (CPU-only) coverage never caught it. Fixed by resolving the
routing indices on the host, the same way the offload/hybrid/cpu backends
already do.

``xpu``-marked: deselected on a torch-free / no-XPU box (see ``conftest.py``).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from freetoken.core import Req, SamplingParams, reset_global_ctx
from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.engine.engine import Engine
from freetoken.models.loader import load_model

from tests.test_moe_offload_forward import TINY_CONFIG, offload_ckpt  # noqa: F401

XPU = pytest.mark.skipif(not torch.xpu.is_available(), reason="no XPU available")


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
        moe_backend="fused",
        max_running_req=2,
        page_size=1,
        max_seq_len_override=32,
        num_page_override=64,
    )


# 9 tokens (> num_experts * top_k), prefilled in one step: reliably reproduced
# the device-side nonzero() bug in manual repro (empty result for a multi-True
# boolean mask on XPU), unlike a 3-4 token prompt, where the bug is
# intermittent -- it depends on *which* positions are True, not just how many.
_PROMPT_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9]


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
def test_fused_inram_forward_matches_cpu_reference_on_xpu(offload_ckpt):
    """A multi-token prefill through the in-VRAM path on real XPU hardware.

    The 9-token prompt is prefilled in one step, so multiple experts route
    different tokens in the same forward call -- the shape that reliably
    exposed the device-side ``nonzero()`` bug (see module docstring). This
    must not crash, and must reproduce the CPU in-VRAM reference's greedy
    tokens exactly (same weights, same math, only the device differs).
    """
    dev = torch.device("xpu")

    cpu_model, _ = load_model(offload_ckpt, torch.device("cpu"), dtype=torch.float32, moe_backend="fused")
    assert cpu_model.layers[0].mlp.experts is not None, "fused path must build resident experts"
    cpu_engine = Engine(_engine_config(offload_ckpt, torch.device("cpu")))
    _add_prompt(cpu_engine, output_len=6)
    cpu_tokens = cpu_engine.generate()

    xpu_model, _ = load_model(offload_ckpt, dev, dtype=torch.float32, moe_backend="fused")
    assert xpu_model.layers[0].mlp.experts is not None
    xpu_engine = Engine(_engine_config(offload_ckpt, dev))
    _add_prompt(xpu_engine, output_len=6)
    xpu_tokens = xpu_engine.generate()  # must not raise (the bug crashed here)

    assert xpu_tokens == cpu_tokens, (
        f"fused in-VRAM forward diverged between CPU and XPU: {xpu_tokens} != {cpu_tokens}"
    )
    assert len(xpu_tokens[0]) == 6
    assert all(0 <= t < TINY_CONFIG["vocab_size"] for t in xpu_tokens[0])
