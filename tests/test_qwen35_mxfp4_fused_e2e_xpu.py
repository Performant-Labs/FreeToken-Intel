"""End-to-end: load_model() wires an MXFP4-quantized checkpoint's routed
experts into the fully XPU-resident (fused) path and runs a real forward
step (issue `moe-fused-mxfp4`, #180, part of the `quant-xpu` epic, #10).

This is the "fused" sibling of ``test_qwen35_mxfp4_e2e_loader.py`` (the
already-merged OFFLOAD e2e test, #153): same packed-bank checkpoint shape,
different backend (``moe_backend="fused"`` instead of ``"offload"``),
proving the previously-missing gap -- `loader.py`'s in-VRAM placement path
did not know how to place a quantized bank at all (it assumed a plain
stacked bf16 tensor and would crash on a real `MxfpExpertBank`).

``xpu``-marked: :func:`freetoken.kernel.triton.fused_mxfp4_linear.
fused_mxfp4_expert_forward` needs a real Triton-XPU compile, the same
reasoning ``test_fused_mxfp4_linear_xpu.py`` documents -- no meaningful
CPU-only synthetic-fixture version exists.

A small (few-KB) fabricated MXFP4 checkpoint, not a real multi-GB one --
mirrors ``test_qwen35_gptq_e2e_loader.py``'s own established style.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

XPU = pytest.mark.skipif(not torch.xpu.is_available(), reason="no XPU available")

from freetoken.kernel.triton.mxfp4_linear import quantize_mxfp4_blocks
from freetoken.models.loader import load_model

H, I, E, V, L = 32, 32, 4, 64, 1  # H, I both multiples of 32 (MXFP4's own block size)
NH, NKV, HD = 4, 2, 16
Q_PROJ_DIM = NH * HD * 2  # q_proj always fuses query + output gate (_Qwen35Attention)
O_PROJ_DIM = NH * HD
KV_PROJ_DIM = NKV * HD


def _pack_experts(dense: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[E, N, K] -> (blocks [E, N, K//32, 16], scales [E, N, K//32])``,
    matching MxfpExpertBank's own per-expert layout exactly."""
    blocks, scales = quantize_mxfp4_blocks(dense)
    return blocks, scales


def _qwen35_mxfp4_weights() -> dict:
    """One full-attention layer's dense weights (unquantized) plus MXFP4-
    packed routed experts (the real gpt-oss/Qwen3.6 checkpoint's own
    already-packed-per-projection layout, verified against
    openai/gpt-oss-20b's index.json -- see MxfpExpertBank's own docstring)
    and a plain (unquantized) shared expert."""
    w = {
        "model.language_model.embed_tokens.weight": torch.randn(V, H),
        "model.language_model.norm.weight": torch.randn(H),
        "lm_head.weight": torch.randn(V, H),
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(Q_PROJ_DIM, H),
        "model.language_model.layers.0.self_attn.k_proj.weight": torch.randn(KV_PROJ_DIM, H),
        "model.language_model.layers.0.self_attn.v_proj.weight": torch.randn(KV_PROJ_DIM, H),
        "model.language_model.layers.0.self_attn.o_proj.weight": torch.randn(H, O_PROJ_DIM),
        "model.language_model.layers.0.self_attn.q_norm.weight": torch.randn(HD),
        "model.language_model.layers.0.self_attn.k_norm.weight": torch.randn(HD),
        "model.language_model.layers.0.mlp.gate.weight": torch.randn(E, H),
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": torch.randn(I, H),
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight": torch.randn(I, H),
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight": torch.randn(H, I),
        "model.language_model.layers.0.mlp.shared_expert_gate.weight": torch.randn(1, H),
    }
    gate_up_dense = torch.randn(E, 2 * I, H)  # K = H (must be a multiple of 32)
    down_dense = torch.randn(E, H, I)  # K = I (must be a multiple of 32)
    gu_blocks, gu_scales = _pack_experts(gate_up_dense)
    dn_blocks, dn_scales = _pack_experts(down_dense)
    base = "model.language_model.layers.0.mlp.experts"
    w[f"{base}.gate_up_proj_blocks"] = gu_blocks
    w[f"{base}.gate_up_proj_scales"] = gu_scales
    w[f"{base}.down_proj_blocks"] = dn_blocks
    w[f"{base}.down_proj_scales"] = dn_scales
    return w


@pytest.fixture(scope="module")
def qwen35_mxfp4_ckpt(tmp_path_factory):
    from safetensors.torch import save_file

    path = tmp_path_factory.mktemp("qwen35_mxfp4_fused")
    text_config = {
        "hidden_size": H,
        "num_hidden_layers": L,
        "num_attention_heads": NH,
        "num_key_value_heads": NKV,
        "num_experts": E,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": I,
        "shared_expert_intermediate_size": I,
        "vocab_size": V,
        "max_position_embeddings": 128,
        "head_dim": HD,
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
        "quantization_config": {"quant_method": "mxfp4"},
    }
    weights = _qwen35_mxfp4_weights()
    (path / "config.json").write_text(json.dumps(config))
    save_file({k: v.contiguous() for k, v in weights.items()}, str(path / "model.safetensors"))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in weights}})
    )
    return str(path)


@XPU
@pytest.mark.xpu
def test_load_model_places_mxfp4_experts_fully_resident(qwen35_mxfp4_ckpt):
    """The real point of #180: moe_backend="fused" no longer crashes on a
    quantized bank -- each expert becomes a real _Qwen35MxfpExpert holding
    packed blocks/scales on the device, not a bf16 nn.Linear."""
    from freetoken.models.qwen3_5_moe import _Qwen35MxfpExpert

    model, _ = load_model(
        qwen35_mxfp4_ckpt, torch.device("xpu"), dtype=torch.bfloat16, moe_backend="fused"
    )
    experts = model.layers[0].mlp.experts
    assert len(experts) == E
    for expert in experts:
        assert isinstance(expert, _Qwen35MxfpExpert)
        assert expert.blocks_gate_up.device.type == "xpu"
        assert expert.blocks_gate_up.dtype == torch.uint8


@XPU
@pytest.mark.xpu
def test_mxfp4_fused_forward_step_produces_finite_logits(qwen35_mxfp4_ckpt):
    """A real forward pass through the fully-resident MXFP4 experts -- the
    native fused_mxfp4_expert_forward kernel, no host round-trip at all."""
    from freetoken.attention.triton import TritonAttentionBackend
    from freetoken.core import Batch, Context, Req, SamplingParams, reset_global_ctx, set_global_ctx
    from freetoken.kvcache.base import BaseKVCachePool

    model, _ = load_model(
        qwen35_mxfp4_ckpt, torch.device("xpu"), dtype=torch.bfloat16, moe_backend="fused"
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
