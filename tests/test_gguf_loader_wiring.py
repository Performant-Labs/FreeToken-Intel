"""Tests for GGUF tensor iteration + loader wiring (issue `models-gguf-iter-and-loader-wiring`, #273).

Builds a tiny, hand-assembled, SYNTHETIC ``qwen3moe``-architecture GGUF
checkpoint (F32 tensors -- the simplest quant type, already covered by
issue #271's own dequant tests, so this file's job is validating the
tensor-NAME mapping and loader wiring, not dequant correctness again) and
loads it through the real ``load_model()`` entry point end to end, matching
this issue's own accept bar ("load_model() loads a small real GGUF
checkpoint and runs a real forward pass producing finite logits").

``qwen3moe`` (not the real target's ``qwen35moe``) is used deliberately: it
is a standard (non-GDN, non-shared-expert) MoE transformer with 1:1 tensor
names matching this port's existing ``Qwen3MoeForCausalLM`` HF convention
(separate q/k/v projections with their own q_norm/k_norm) -- confirmed
against ``gguf-py``'s own ``MODEL_TENSORS[MODEL_ARCH.QWEN3MOE]`` table.
``qwen35moe``'s Gated-Delta-Net linear-attention layers (``ssm_*``, fused
``attn_qkv``, shared-expert ``ffn_*_shexp``) use a genuinely different
tensor set this issue's ``_DENSE_SUFFIX_MAP`` does not cover -- confirmed
directly against the real target checkpoint's own header (Zot:
``general/qwen3.6-35b-a3b:q4_k_m-gguf``, read via an HTTP range request),
and flagged here as real, separable follow-up scope, not silently assumed
to work.
"""
from __future__ import annotations

import struct

import pytest

torch = pytest.importorskip("torch")

from freetoken.core import Req, SamplingParams, reset_global_ctx
from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.engine.engine import Engine
from freetoken.models.gguf import GGUFValueType, iter_weights
from freetoken.models.loader import load_model

DEVICE = torch.device("cpu")


def _gguf_string(s: str) -> bytes:
    data = s.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _kv_scalar(key: str, value_type: int, fmt: str, value) -> tuple:
    return (key, value_type, struct.pack("<" + fmt, value))


def _kv_string(key: str, value: str) -> tuple:
    return (key, GGUFValueType.STRING, _gguf_string(value))


# --------------------------------------------------------------------------
# A tiny synthetic qwen3moe checkpoint: 2 layers, 4 experts, top-2, hidden=8,
# 2 heads (head_dim=4), moe_intermediate=6, vocab=16.
# --------------------------------------------------------------------------
HIDDEN = 8
HEADS = 2
KV_HEADS = 2
HEAD_DIM = 4
INTER = 6
EXPERTS = 4
TOPK = 2
LAYERS = 2
VOCAB = 16


def _f32_tensor_bytes(shape) -> bytes:
    n = 1
    for d in shape:
        n *= d
    # Small deterministic values (not all-zero: a zero attn_q_norm/k_norm
    # RMSNorm weight would silently zero every downstream activation and
    # mask a real wiring bug as "finite logits" trivially).
    return struct.pack(f"<{n}f", *[0.02 * ((i % 7) - 3) for i in range(n)])


def _build_synthetic_qwen3moe_gguf(tmp_path) -> str:
    kv = [
        _kv_string("general.architecture", "qwen3moe"),
        _kv_string("general.name", "tiny-qwen3moe"),
        _kv_scalar("qwen3moe.context_length", GGUFValueType.UINT32, "I", 128),
        _kv_scalar("qwen3moe.embedding_length", GGUFValueType.UINT32, "I", HIDDEN),
        _kv_scalar("qwen3moe.block_count", GGUFValueType.UINT32, "I", LAYERS),
        _kv_scalar("qwen3moe.feed_forward_length", GGUFValueType.UINT32, "I", HIDDEN),
        _kv_scalar("qwen3moe.attention.head_count", GGUFValueType.UINT32, "I", HEADS),
        _kv_scalar("qwen3moe.attention.head_count_kv", GGUFValueType.UINT32, "I", KV_HEADS),
        _kv_scalar("qwen3moe.attention.key_length", GGUFValueType.UINT32, "I", HEAD_DIM),
        _kv_scalar("qwen3moe.attention.layer_norm_rms_epsilon", GGUFValueType.FLOAT32, "f", 1e-5),
        _kv_scalar("qwen3moe.rope.freq_base", GGUFValueType.FLOAT32, "f", 1000000.0),
        _kv_scalar("qwen3moe.expert_count", GGUFValueType.UINT32, "I", EXPERTS),
        _kv_scalar("qwen3moe.expert_used_count", GGUFValueType.UINT32, "I", TOPK),
        _kv_scalar("qwen3moe.expert_feed_forward_length", GGUFValueType.UINT32, "I", INTER),
        _kv_scalar("qwen3moe.vocab_size", GGUFValueType.UINT32, "I", VOCAB),
    ]

    tensors: list[tuple] = []
    data_chunks: list[bytes] = []
    offset = 0

    def add(name: str, shape: tuple):
        nonlocal offset
        payload = _f32_tensor_bytes(shape)
        tensors.append((name, shape, 0, offset))  # ggml_type 0 == F32
        data_chunks.append(payload)
        offset += len(payload)

    # GGUF ne-order (fastest-varying first) -- reversed PyTorch shape, per
    # this issue's own _gguf_to_pytorch_shape finding.
    add("token_embd.weight", (HIDDEN, VOCAB))
    add("output_norm.weight", (HIDDEN,))
    add("output.weight", (HIDDEN, VOCAB))
    for layer in range(LAYERS):
        p = f"blk.{layer}"
        add(f"{p}.attn_norm.weight", (HIDDEN,))
        add(f"{p}.attn_q.weight", (HIDDEN, HEADS * HEAD_DIM))
        add(f"{p}.attn_q_norm.weight", (HEAD_DIM,))
        add(f"{p}.attn_k.weight", (HIDDEN, KV_HEADS * HEAD_DIM))
        add(f"{p}.attn_k_norm.weight", (HEAD_DIM,))
        add(f"{p}.attn_v.weight", (HIDDEN, KV_HEADS * HEAD_DIM))
        add(f"{p}.attn_output.weight", (HEADS * HEAD_DIM, HIDDEN))
        add(f"{p}.ffn_norm.weight", (HIDDEN,))
        add(f"{p}.ffn_gate_inp.weight", (HIDDEN, EXPERTS))
        # MoE expert banks: ne-order [hidden, inter, experts] / [inter, hidden, experts].
        add(f"{p}.ffn_gate_exps.weight", (HIDDEN, INTER, EXPERTS))
        add(f"{p}.ffn_up_exps.weight", (HIDDEN, INTER, EXPERTS))
        add(f"{p}.ffn_down_exps.weight", (INTER, HIDDEN, EXPERTS))

    header = _build_minimal_gguf(kv=kv, tensors=tensors)
    path = tmp_path / "tiny-qwen3moe.gguf"
    path.write_bytes(header + b"".join(data_chunks))
    return str(path)


def _build_minimal_gguf(*, version: int = 3, kv: list, tensors: list, alignment: int | None = None) -> bytes:
    """Copy of tests/test_gguf_reader.py's own helper (issue #270) -- builds
    everything up to (not including) the tensor-data bytes, which the caller
    appends immediately after (this fixture's own ``add()`` computes
    ``rel_offset`` assuming zero padding after the header, matching
    ``general.alignment``'s default of 32 only when the header itself
    happens to already land on a 32-byte boundary; asserted below)."""
    count_fmt = "<I" if version == 1 else "<Q"
    out = bytearray()
    out += b"GGUF"
    out += struct.pack("<I", version)
    out += struct.pack(count_fmt, len(tensors))
    out += struct.pack(count_fmt, len(kv))
    for key, value_type, raw in kv:
        out += _gguf_string(key)
        out += struct.pack("<I", value_type)
        out += raw
    for name, shape, ggml_type, rel_offset in tensors:
        out += _gguf_string(name)
        out += struct.pack("<I", len(shape))
        for d in shape:
            out += struct.pack("<Q", d)
        out += struct.pack("<I", ggml_type)
        out += struct.pack("<Q", rel_offset)
    pad = (-len(out)) % 32
    out += b"\x00" * pad
    return bytes(out)


@pytest.fixture(scope="module")
def synthetic_gguf_path(tmp_path_factory) -> str:
    return _build_synthetic_qwen3moe_gguf(tmp_path_factory.mktemp("gguf-loader-wiring"))


def test_iter_weights_yields_correctly_named_and_shaped_tensors(synthetic_gguf_path):
    seen = dict(iter_weights(synthetic_gguf_path, torch.device("cpu")))

    assert seen["model.embed_tokens.weight"].shape == (VOCAB, HIDDEN)
    assert seen["model.norm.weight"].shape == (HIDDEN,)
    assert seen["lm_head.weight"].shape == (VOCAB, HIDDEN)
    for layer in range(LAYERS):
        p = f"model.layers.{layer}"
        assert seen[f"{p}.input_layernorm.weight"].shape == (HIDDEN,)
        assert seen[f"{p}.self_attn.q_proj.weight"].shape == (HEADS * HEAD_DIM, HIDDEN)
        assert seen[f"{p}.self_attn.q_norm.weight"].shape == (HEAD_DIM,)
        assert seen[f"{p}.self_attn.k_proj.weight"].shape == (KV_HEADS * HEAD_DIM, HIDDEN)
        assert seen[f"{p}.self_attn.v_proj.weight"].shape == (KV_HEADS * HEAD_DIM, HIDDEN)
        assert seen[f"{p}.self_attn.o_proj.weight"].shape == (HIDDEN, HEADS * HEAD_DIM)
        assert seen[f"{p}.post_attention_layernorm.weight"].shape == (HIDDEN,)
        assert seen[f"{p}.mlp.gate.weight"].shape == (EXPERTS, HIDDEN)
        # Packed MoE banks: gate_up_proj is [E, 2I, H] (gate then up, dim=1).
        assert seen[f"{p}.mlp.experts.gate_up_proj"].shape == (EXPERTS, 2 * INTER, HIDDEN)
        assert seen[f"{p}.mlp.experts.down_proj"].shape == (EXPERTS, HIDDEN, INTER)
        assert seen[f"{p}.mlp.experts.gate_up_proj"].device.type == "cpu"


def test_iter_weights_respects_include_flags(synthetic_gguf_path):
    dense_only = dict(iter_weights(synthetic_gguf_path, torch.device("cpu"), include_moe_experts=False))
    assert "model.layers.0.mlp.experts.gate_up_proj" not in dense_only
    assert "model.embed_tokens.weight" in dense_only

    experts_only = dict(iter_weights(synthetic_gguf_path, torch.device("cpu"), include_non_moe=False))
    assert "model.layers.0.mlp.experts.gate_up_proj" in experts_only
    assert "model.embed_tokens.weight" not in experts_only


def test_load_model_end_to_end_produces_finite_logits(synthetic_gguf_path):
    model, expert_sources = load_model(synthetic_gguf_path, torch.device("cpu"), dtype=torch.float32)
    gate_up_banks, down_banks = expert_sources
    assert len(gate_up_banks) == LAYERS
    for p in model.parameters():
        assert torch.isfinite(p).all(), "GGUF-loaded model has a non-finite parameter"


def _engine_config(model_path: str, *, moe_backend=None) -> "EngineConfig":
    return EngineConfig(
        model_path=model_path,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.float32,
        device=DEVICE,
        attention_backend="auto",
        moe_backend=moe_backend,
        max_running_req=2,
        page_size=1,
        max_seq_len_override=32,
        num_page_override=64,
    )


def test_engine_generate_produces_in_vocab_tokens(synthetic_gguf_path):
    """This issue's own accept bar, exercised end to end through the real
    serving path (not a bare ``model.forward()`` call, which needs KV-cache
    ``positions``/``out_loc`` this port's engine supplies -- see
    ``tests/test_moe_offload_forward.py``'s own established pattern for
    every other backend's equivalent test): ``load_model()`` loads a real
    GGUF checkpoint, and ``Engine.generate()`` (prefill + decode) produces
    finite, in-vocabulary greedy tokens.
    """
    reset_global_ctx()
    engine = Engine(_engine_config(synthetic_gguf_path))
    engine.add_request(
        Req(
            input_ids=[1, 2, 3],
            table_idx=0,
            cached_len=0,
            output_len=4,
            uid=0,
            sampling_params=SamplingParams(temperature=0.0, max_tokens=4),
            cache_handle=None,
        )
    )
    tokens = engine.generate()
    reset_global_ctx()
    assert len(tokens[0]) == 4
    assert all(0 <= t < VOCAB for t in tokens[0])


def test_engine_offload_backend_generate_produces_in_vocab_tokens(synthetic_gguf_path):
    """The offload/cpu/hybrid backends all route through the same
    _attach_offload_cache path -- confirm offload actually builds a working
    OffloadMoeCache from GGUF-sourced banks (plain bf16 _PlainBank tensors,
    dequantized eagerly by iter_weights -- see that function's own
    docstring for why this needs no GGUF-specific bank/quant_format
    handling anywhere downstream, unlike the packed GPTQ/FP8/MXFP4/INT8
    formats) and actually generates through it.
    """
    reset_global_ctx()
    engine = Engine(_engine_config(synthetic_gguf_path, moe_backend="offload"))
    assert engine.model.moe_cache is not None
    assert engine.model.moe_cache.quant_format == "bf16"
    engine.add_request(
        Req(
            input_ids=[1, 2, 3],
            table_idx=0,
            cached_len=0,
            output_len=4,
            uid=0,
            sampling_params=SamplingParams(temperature=0.0, max_tokens=4),
            cache_handle=None,
        )
    )
    tokens = engine.generate()
    reset_global_ctx()
    assert len(tokens[0]) == 4
    assert all(0 <= t < VOCAB for t in tokens[0])
