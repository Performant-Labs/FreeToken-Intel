"""End-to-end: load_model() wires a compressed-tensors pack-quantized
INT8 checkpoint's routed experts into the fully XPU-resident (fused) path
and runs a real forward step (issue `moe-fused-int8`, #182, part of the
`quant-xpu` epic, #10).

This is the "fused" sibling of ``test_qwen35_int8_e2e_loader.py`` (the
already-merged OFFLOAD e2e test, #154/#163): same checkpoint fixture (reused
directly, not duplicated), different backend (``moe_backend="fused"``
instead of ``"offload"``), proving the previously-missing gap -- loader.py's
in-VRAM placement path did not know how to place a quantized bank at all.

``xpu``-marked: :func:`freetoken.kernel.triton.fused_int8_linear.
fused_int8_expert_forward` needs a real Triton-XPU compile, the same
reasoning ``test_fused_int8_linear_xpu.py`` documents -- no meaningful
CPU-only synthetic-fixture version exists.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

XPU = pytest.mark.skipif(not torch.xpu.is_available(), reason="no XPU available")

from freetoken.models.loader import load_model

from tests.test_qwen35_int8_e2e_loader import E, GROUP_SIZE, H, I, V, _int8_expert_weights


@pytest.fixture(scope="module")
def qwen35_int8_ckpt_fused(tmp_path_factory):
    """Same fixture shape as test_qwen35_int8_e2e_loader.qwen35_int8_ckpt,
    built directly here (a module-scoped fixture from another test module
    can't be reused as a fixture across files) via the same weight/config
    builders that module already exports."""
    import json

    from safetensors.torch import save_file

    path = tmp_path_factory.mktemp("qwen35_int8_fused")
    text_config = {
        "hidden_size": H,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_experts": E,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": I,
        "shared_expert_intermediate_size": I,
        "vocab_size": V,
        "max_position_embeddings": 128,
        "head_dim": 16,
        "attn_output_gate": False,
        "partial_rotary_factor": 0.5,
        "full_attention_interval": 1,
        "layer_types": ["full_attention"],
        "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.5},
    }
    config = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "tie_word_embeddings": True,
        "text_config": text_config,
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {
                    "format": "pack-quantized",
                    "weights": {
                        "num_bits": 8,
                        "type": "int",
                        "strategy": "group",
                        "group_size": GROUP_SIZE,
                        "symmetric": True,
                        "actorder": "static",
                    },
                }
            },
        },
    }
    weights = _int8_expert_weights()
    (path / "config.json").write_text(json.dumps(config))
    save_file({k: v.contiguous() for k, v in weights.items()}, str(path / "model.safetensors"))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in weights}})
    )
    return str(path)


@XPU
@pytest.mark.xpu
def test_load_model_places_int8_experts_fully_resident(qwen35_int8_ckpt_fused):
    """The real point of #182: moe_backend="fused" no longer crashes on a
    quantized bank -- each expert becomes a real _Qwen35Int8Expert holding
    packed weight_packed/weight_scale on the device, not a bf16 nn.Linear."""
    from freetoken.models.qwen3_5_moe import _Qwen35Int8Expert

    model, _ = load_model(
        qwen35_int8_ckpt_fused, torch.device("xpu"), dtype=torch.bfloat16, moe_backend="fused"
    )
    experts = model.layers[0].mlp.experts
    assert len(experts) == E
    for expert in experts:
        assert isinstance(expert, _Qwen35Int8Expert)
        assert expert.weight_packed_gate_up.device.type == "xpu"
        assert expert.weight_packed_gate_up.dtype == torch.int32
        assert expert.k_gate_up == H
        assert expert.k_down == I


@XPU
@pytest.mark.xpu
def test_int8_fused_forward_step_produces_finite_logits(qwen35_int8_ckpt_fused):
    """A real forward pass through the fully-resident INT8 experts -- the
    native fused_int8_expert_forward kernel, no host round-trip at all."""
    from freetoken.attention.triton import TritonAttentionBackend
    from freetoken.core import Batch, Context, Req, SamplingParams, reset_global_ctx, set_global_ctx
    from freetoken.kvcache.base import BaseKVCachePool

    model, _ = load_model(
        qwen35_int8_ckpt_fused, torch.device("xpu"), dtype=torch.bfloat16, moe_backend="fused"
    )
    seq = torch.randint(0, V, (5,), device="xpu")
    T = seq.shape[0]

    req = Req(
        input_ids=seq,
        table_idx=0,
        cached_len=0,
        output_len=1,
        uid=0,
        sampling_params=SamplingParams(),
        cache_handle=None,
    )
    batch = Batch(reqs=[req], phase="prefill")
    batch.input_ids = torch.arange(T, dtype=torch.long, device="xpu")
    batch.positions = torch.arange(T, dtype=torch.long, device="xpu")
    batch.out_loc = torch.arange(T, dtype=torch.long, device="xpu")
    batch.extend_lens = torch.tensor([T])
    ctx = Context(page_size=1)
    ctx.kv_cache = BaseKVCachePool(model.config, 1, 1024, torch.device("xpu"), torch.bfloat16)
    pt = torch.zeros((1, 1024), dtype=torch.int32, device="xpu")
    pt[0, :T] = torch.arange(T, device="xpu")
    ctx.page_table = pt
    ctx.kv_cache.attach_page_table(pt)
    backend = TritonAttentionBackend(model.config)
    backend.prepare_metadata(batch)
    ctx.attn_backend = backend
    ctx.model = model
    set_global_ctx(ctx)
    ctx._batch = batch
    try:
        logits = model.forward(batch.input_ids, batch.positions, batch.out_loc)
    finally:
        reset_global_ctx()

    assert logits.shape == (1, V)
    assert torch.isfinite(logits.float()).all()
