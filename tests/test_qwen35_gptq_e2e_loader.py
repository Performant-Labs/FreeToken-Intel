"""End-to-end: load_model() wires a GPTQ-quantized checkpoint's experts into
a real "gptq_int4" OffloadMoeCache and runs a forward step (issue
`moe-quant-banks-e2e`, #138).

This is the glue #135/#136/#137 (all merged) did not themselves cover:
- weight.py's load_moe_expert_sources must detect a GPTQ checkpoint
  (checkpoint_quant_method) and dispatch to stream_moe_expert_sources_gptq
  instead of the bf16 path
- loader.py's _attach_offload_cache must detect GptqExpertBank-shaped banks
  and build the cache with quant_format="gptq_int4", the six packed banks,
  g_idx as extra_metadata, and cache.gptq_group_size (SlotWeightAccessor,
  #137, refuses to guess this)

A small (few-KB) fabricated GPTQ checkpoint, not the real 22.73GB
Qwen3.5-35B-A3B-GPTQ-Int4 checkpoint -- this proves the wiring is correct
before ever touching real hardware / the real checkpoint at scale.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from freetoken.models.loader import load_model

H, I, E, V, L = 32, 16, 4, 64, 1
GROUP = 16
NH, NKV, HD = 4, 2, 16
Q_PROJ_DIM = NH * HD * 2
O_PROJ_DIM = NH * HD
KV_PROJ_DIM = NKV * HD


def _pack_nibbles(codes: list[int]) -> int:
    word = 0
    for i, c in enumerate(codes):
        word |= c << (4 * i)
    return word


def _gptq_pack(k: int, n: int, *, code: int, zero_code: int, scale: float, group_size: int = GROUP):
    """A trivial (constant-valued, real-shaped) GPTQ-packed projection:
    every element decodes to ``scale * (code - (zero_code + 1))``. Includes
    an explicit ``g_idx`` (sequential groups) -- the real checkpoint ships
    one per expert, even though desc_act=False means it is reconstructible
    (see gptq_linear.dequantize_gptq_int4_sequential_groups, #137)."""
    n_groups = -(-k // group_size)
    qweight = torch.full((k // 8, n), _pack_nibbles([code] * 8), dtype=torch.int32)
    qzeros = torch.full((n_groups, n // 8), _pack_nibbles([zero_code] * 8), dtype=torch.int32)
    scales = torch.full((n_groups, n), scale)
    g_idx = torch.arange(k, dtype=torch.int32) // group_size
    return qweight, qzeros, scales, g_idx


def _qwen35_gptq_weights() -> dict:
    """One full-attention layer's dense weights (unquantized -- attention is
    excluded from quantization_config.dynamic, matching the real checkpoint)
    plus GPTQ-packed routed experts (per-expert raw layout, the real
    checkpoint's own shape) and a plain (unquantized) shared expert."""
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
    for e in range(E):
        base = f"model.language_model.layers.0.mlp.experts.{e}"
        for proj, k, n in (("gate_proj", H, I), ("up_proj", H, I), ("down_proj", I, H)):
            qweight, qzeros, scales, g_idx = _gptq_pack(k, n, code=(e + 1) % 16, zero_code=7, scale=0.1)
            w[f"{base}.{proj}.qweight"] = qweight
            w[f"{base}.{proj}.qzeros"] = qzeros
            w[f"{base}.{proj}.scales"] = scales
            w[f"{base}.{proj}.g_idx"] = g_idx
    return w


@pytest.fixture(scope="module")
def qwen35_gptq_ckpt(tmp_path_factory):
    from safetensors.torch import save_file

    path = tmp_path_factory.mktemp("qwen35_gptq")
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
        "quantization_config": {
            "bits": 4,
            "group_size": GROUP,
            "sym": True,
            "desc_act": False,
            "quant_method": "gptq",
            "dynamic": {
                "-:.*attn.*": {},
                "-:.*shared_expert.*": {},
            },
        },
    }
    weights = _qwen35_gptq_weights()
    (path / "config.json").write_text(json.dumps(config))
    save_file({k: v.contiguous() for k, v in weights.items()}, str(path / "model.safetensors"))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in weights}})
    )
    return str(path)


def test_load_model_builds_a_gptq_int4_offload_cache(qwen35_gptq_ckpt):
    model, _ = load_model(qwen35_gptq_ckpt, torch.device("cpu"), dtype=torch.float32, moe_backend="offload")

    cache = model.moe_cache
    assert cache.quant_format == "gptq_int4"
    assert cache.gptq_group_size == GROUP
    assert set(cache.bank_sources) == {
        "qweight_gate_up",
        "qzeros_gate_up",
        "scales_gate_up",
        "qweight_down",
        "qzeros_down",
        "scales_down",
    }
    assert cache.get_extra_metadata("g_idx_gate_up", 0).shape == (H,)
    assert cache.get_extra_metadata("g_idx_down", 0).shape == (I,)


def _drive_prefill(model, seq, T, table_idx=0):
    """Mirrors test_models_qwen35_loader.py's own helper of the same name --
    a real forward needs a live Context (KV pool, page table, attention
    backend), not just load_model()."""
    from freetoken.attention.triton import TritonAttentionBackend
    from freetoken.core import Batch, Context, Req, SamplingParams, set_global_ctx
    from freetoken.kvcache.base import BaseKVCachePool

    req = Req(
        input_ids=seq,
        table_idx=table_idx,
        cached_len=0,
        output_len=1,
        uid=0,
        sampling_params=SamplingParams(),
        cache_handle=None,
    )
    batch = Batch(reqs=[req], phase="prefill")
    batch.input_ids = torch.arange(T, dtype=torch.long)
    batch.positions = torch.arange(T, dtype=torch.long)
    batch.out_loc = torch.arange(T, dtype=torch.long)
    batch.extend_lens = torch.tensor([T])
    ctx = Context(page_size=1)
    ctx.kv_cache = BaseKVCachePool(model.config, 1, 1024, torch.device("cpu"), torch.float32)
    pt = torch.zeros((1, 1024), dtype=torch.int32)
    pt[0, :T] = torch.arange(T)
    ctx.page_table = pt
    ctx.kv_cache.attach_page_table(pt)
    backend = TritonAttentionBackend(model.config)
    backend.prepare_metadata(batch)
    ctx.attn_backend = backend
    # The real Engine sets this (engine.py: self.ctx.model = self.model) --
    # _forward_offload_core reads ctx.model.moe_cache. Not part of
    # test_models_qwen35_loader.py's own _drive_prefill (its fixture never
    # exercises the offload path), needed here since this test's whole point
    # is the offload/gptq_int4 forward.
    ctx.model = model
    set_global_ctx(ctx)
    ctx._batch = batch
    try:
        logits = model.forward(batch.input_ids, batch.positions, batch.out_loc)
    finally:
        from freetoken.core import reset_global_ctx

        reset_global_ctx()
    return logits, ctx


def test_gptq_checkpoint_forward_step_produces_finite_logits(qwen35_gptq_ckpt):
    """The real point of this issue (#138): a GPTQ-quantized checkpoint's
    forward pass runs end-to-end -- SlotWeightAccessor (#137) dequantizes the
    resident packed experts on the fly, reading from the cache #135/#136/this
    issue's loader wiring actually built."""
    model, _ = load_model(qwen35_gptq_ckpt, torch.device("cpu"), dtype=torch.float32, moe_backend="offload")
    seq = torch.randint(0, V, (5,))
    T = seq.shape[0]
    logits, _ = _drive_prefill(model, seq, T)
    assert logits.shape == (1, V)
    assert torch.isfinite(logits).all()
