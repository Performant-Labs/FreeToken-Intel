"""XPU tests: Qwen3.5/3.6 MoE forward through the engine with host-offload (issue #18, PR3).

PR2 (#90) proved the qwen3_5_moe forward is *mathematically correct against an independent
analytical reference on CPU. This module closes the remaining Accept criteria on
the B70: the offload path must run a real prefill + decode step *through the
engine* on the XPU and agree with the device-invariant CPU reference -- because
the LRU slot pool is a byte-identical transport for the expert weights, not a
change to the math (ADR 0002).

The reference is the CPU path (not the XPU in-VRAM "fused" path) because the
Qwen3.5 Gated-Delta-Net linear-attention recurrence is chaotic in float32: the
in-VRAM and host-offload expert paths use different float32 reduction orders,
and ``s = s * g_t`` amplifies a ~1e-7 rounding difference across the decode steps
until the greedy argmax flips. The offload transport is a byte-identical weight
copy, so it matches the CPU reference exactly; the fused path's drift is a
kernel-order artifact, not a transport bug.

The fabricated multimodal checkpoint is shared with ``test_models_qwen35_loader``
(2 layers -- 1 Gated-Delta-Net linear-attention + 1 gated GQA full-attention --
4 routed experts + a shared expert, small hidden/vocab). The in-VRAM reference and
the offload-under-test load the *same* file, so they consume byte-identical dense
+ expert bytes: any divergence is a transport bug, not a weight difference.

``xpu``-marked: deselected on a torch-free / no-XPU box (see ``conftest.py``);
runs under the B70 nightly.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from freetoken.core import Req, SamplingParams, reset_global_ctx
from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.engine.engine import Engine
from freetoken.models.loader import load_model

# Reuse the loader suite's proven-correct fabricated weights (same shapes/names),
# so the offload-under-test and the reference consume identical bytes.
from tests.test_models_qwen35_loader import (
    V as _V,
    _qwen35_text_config,
    _qwen35_weights,
)

# XPU is the point of this module: skip cleanly (not fail) where there is none.
XPU = pytest.mark.skipif(not torch.xpu.is_available(), reason="no XPU available")


@pytest.fixture(scope="module")
def qwen35_xpu_ckpt(tmp_path_factory):
    from safetensors.torch import save_file

    path = tmp_path_factory.mktemp("qwen35-xpu")
    config = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "tie_word_embeddings": True,
        "vision_config": {"hidden_size": 8, "num_chunks": 2},
        "text_config": _qwen35_text_config(),
    }
    torch.manual_seed(2024)  # order-independent fabricated weights
    weights = _qwen35_weights()
    (path / "config.json").write_text(json.dumps(config))
    save_file({k: v.contiguous() for k, v in weights.items()}, str(path / "model.safetensors"))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in weights}})
    )
    return str(path)


@pytest.fixture(autouse=True)
def _clean_global_ctx():
    # Each Engine installs a global context; clear it between tests so two
    # engines in one process never collide.
    yield
    reset_global_ctx()
    if torch.xpu.is_available():
        torch.xpu.empty_cache()


def _engine_config(model_path: str, device, *, moe_backend) -> EngineConfig:
    return EngineConfig(
        model_path=model_path,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.float32,
        device=device,
        attention_backend="auto",
        moe_backend=moe_backend,
        max_running_req=2,
        page_size=1,
        max_seq_len_override=32,
        num_page_override=64,
    )


def _add_prompt(engine: Engine, output_len: int) -> None:
    engine.add_request(
        Req(
            input_ids=[1, 2, 3, 5, 8],
            table_idx=0,
            cached_len=0,
            output_len=output_len,
            uid=0,
            sampling_params=SamplingParams(temperature=0.0, max_tokens=output_len),
            cache_handle=None,
        )
    )


@XPU
def test_qwen35_offload_forward_matches_reference_on_xpu(qwen35_xpu_ckpt):
    """Accept (2): a real prefill + decode step on the B70, offload == reference.

    The same tiny Qwen3.5/3.6 is loaded host-offload through the real engine on
    the XPU and must emit the same greedy tokens as the independent CPU reference
    (the analytical correctness was proven on CPU in PR2). A slot-pool bug (wrong
    expert read, off-by-one layer map, dtype/shape mismatch in the gather) changes
    the logits and fails; a shape-only check would pass regardless.

    The offload path is compared against the CPU reference (not the XPU in-VRAM
    "fused" path) because the Qwen3.5 Gated-Delta-Net linear-attention recurrence
    is *chaotic in float32*: the in-VRAM expert path and the host-offload gather
    use different float32 reduction orders for the same math, and the Gated-Delta-Net
    state ``s = s * g_t`` amplifies a ~1e-7 rounding difference across the decode
    steps until the greedy argmax flips (observed at step 7 on the B70: fused
    ``[...4, 20]`` vs offload/reference ``[...53, 52]``). The offload transport is
    a byte-identical weight copy (ADR 0002), so it matches the device-invariant CPU
    reference exactly; the fused path's float32 drift is a kernel-order artifact,
    not a transport bug.
    """
    dev = torch.device("xpu")

    # Reference: the independent CPU path (device-invariant float32 reference).
    # The CPU reference is the analytical ground truth from PR2; comparing the XPU
    # offload against it catches any transport bug (wrong expert, bad slot map,
    # dtype/shape mismatch) while being immune to the XPU fused path's float32
    # reduction-order sensitivity in the Gated-Delta-Net recurrence.
    cpu_dev = torch.device("cpu")
    ref_engine = Engine(_engine_config(qwen35_xpu_ckpt, cpu_dev, moe_backend="fused"))
    _add_prompt(ref_engine, output_len=8)
    ref_tokens = ref_engine.generate()

    # Under test: host-offload experts (ADR 0002) through the engine on the XPU.
    off_model, off_sources = load_model(qwen35_xpu_ckpt, dev, dtype=torch.float32, moe_backend="offload")
    assert off_model.moe_offload, "offload path must flag the model"
    assert off_model.moe_cache is not None, "the loader must attach the LRU slot pool"
    assert off_model.layers[0].mlp.experts is None, "offload must not build XPU-resident experts"
    assert len(off_sources[0]) == 2  # one bank per MoE layer (both layers here)

    off_engine = Engine(_engine_config(qwen35_xpu_ckpt, dev, moe_backend="offload"))
    _add_prompt(off_engine, output_len=8)
    off_tokens = off_engine.generate()

    # The offload path ran a full prefill + 8 decode steps on the XPU without error.
    assert len(off_tokens[0]) == 8
    assert all(0 <= t < _V for t in off_tokens[0])
    # The offload transport must not change the math: identical greedy output to
    # the device-invariant CPU reference (any slot-pool bug changes the logits).
    assert off_tokens == ref_tokens, (
        f"offload diverged from the CPU reference on XPU: {off_tokens} != {ref_tokens}"
    )
    # The LRU slot pool must have actually been exercised by the decode steps.
    stats = off_engine.model.moe_cache.decode_miss_stats()
    assert stats["calls"] > 0, "decode must call ensure_experts on the slot pool"


@XPU
def test_qwen35_auto_resolves_to_offload_on_xpu(qwen35_xpu_ckpt):
    """Accept (2) plumbing: the engine's default ``moe_backend='auto'`` resolves
    to the host-offload path for a MoE on an XPU (T1), so a bare ``Engine`` on a
    35B hero offloads rather than OOM-ing the 32 GB.
    """
    dev = torch.device("xpu")
    model, _ = load_model(qwen35_xpu_ckpt, dev, dtype=torch.float32, moe_backend="auto")
    assert model.moe_offload, "auto must resolve to the offload path for a MoE on an XPU"
    assert model.moe_cache is not None

    engine = Engine(_engine_config(qwen35_xpu_ckpt, dev, moe_backend="auto"))
    _add_prompt(engine, output_len=6)
    tokens = engine.generate()
    assert len(tokens[0]) == 6 and all(0 <= t < _V for t in tokens[0])
