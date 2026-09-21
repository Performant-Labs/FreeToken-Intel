"""Tests for qwen35moe GGUF tensor-name mapping (issue
`models-gguf-qwen35moe-tensor-mapping`, #279).

Builds a tiny, hand-assembled, SYNTHETIC ``qwen35moe``-architecture GGUF
checkpoint (F32 tensors, mirroring ``test_gguf_loader_wiring.py``'s own
``qwen3moe`` fixture pattern from #273/#278) and loads it through the real
``load_model()`` / ``Engine.generate()`` path end to end.

Every tensor name and shape below was verified directly against the real
target checkpoint's own header (Zot registry:
``general/qwen3.6-35b-a3b:q4_k_m-gguf``, read via an HTTP range request, no
full 22.7GB pull needed) -- see ``python/freetoken/models/gguf/__init__.py``'s
``_QWEN35_MOE_SUFFIX_MAP`` docstring for the full per-tensor evidence. The
hybrid layer split deliberately does NOT follow this checkpoint's own
``full_attention_interval`` formula (layer 5 of 6 is full-attention even
though the interval=4 formula would predict linear-attention there) --
mirroring the real checkpoint's own header, which has exactly this same
"one extra full-attention layer the formula doesn't predict" shape (its
layer 40 of 41). This exercises ``_qwen35_moe_layer_types``'s real
tensor-name-driven layer-type detection, not just the formula fallback.
"""
from __future__ import annotations

import struct

import pytest

torch = pytest.importorskip("torch")

from freetoken.core import Req, SamplingParams, reset_global_ctx
from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.engine.engine import Engine
from freetoken.models.gguf import GGUFValueType, iter_weights, parse_config
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
# A tiny synthetic qwen35moe checkpoint: 6 layers (hybrid: linear-attention
# except layers 3 and 5, which are full-attention -- layer 3 matches the
# full_attention_interval=4 formula, layer 5 deliberately does NOT, matching
# the real checkpoint's own "one extra full-attention layer" shape), 4
# experts top-2 + a shared expert, hidden=8.
# --------------------------------------------------------------------------
HIDDEN = 8
NUM_HEADS = 2
KV_HEADS = 1
HEAD_DIM = 4
ROTARY_DIM = 2  # partial_rotary_factor = ROTARY_DIM / HEAD_DIM = 0.5
EXPERTS = 4
TOPK = 2
INTER = 6
SHARED_INTER = 5
LAYERS = 6
FULL_ATTENTION_INTERVAL = 4
FULL_ATTENTION_LAYERS = {3, 5}  # 5 is NOT predicted by the interval formula
KEY_HEAD_DIM = 4  # ssm.state_size
NUM_KEY_HEADS = 2  # ssm.group_count
NUM_VALUE_HEADS = 4  # ssm.time_step_rank
VALUE_HEAD_DIM = 3  # derived: ssm.inner_size // ssm.time_step_rank
INNER_SIZE = VALUE_HEAD_DIM * NUM_VALUE_HEADS  # ssm.inner_size
CONV_KERNEL = 4
KEY_DIM = KEY_HEAD_DIM * NUM_KEY_HEADS
VALUE_DIM = VALUE_HEAD_DIM * NUM_VALUE_HEADS
CONV_DIM = KEY_DIM * 2 + VALUE_DIM
VOCAB = 16


def _f32_tensor_bytes(shape) -> bytes:
    n = 1
    for d in shape:
        n *= d
    return struct.pack(f"<{n}f", *[0.02 * ((i % 7) - 3) for i in range(n)])


def _build_synthetic_qwen35moe_gguf(tmp_path) -> str:
    kv = [
        _kv_string("general.architecture", "qwen35moe"),
        _kv_string("general.name", "tiny-qwen35moe"),
        _kv_scalar("qwen35moe.context_length", GGUFValueType.UINT32, "I", 128),
        _kv_scalar("qwen35moe.embedding_length", GGUFValueType.UINT32, "I", HIDDEN),
        _kv_scalar("qwen35moe.block_count", GGUFValueType.UINT32, "I", LAYERS),
        _kv_scalar("qwen35moe.feed_forward_length", GGUFValueType.UINT32, "I", INTER),
        _kv_scalar("qwen35moe.attention.head_count", GGUFValueType.UINT32, "I", NUM_HEADS),
        _kv_scalar("qwen35moe.attention.head_count_kv", GGUFValueType.UINT32, "I", KV_HEADS),
        _kv_scalar("qwen35moe.attention.key_length", GGUFValueType.UINT32, "I", HEAD_DIM),
        _kv_scalar("qwen35moe.attention.layer_norm_rms_epsilon", GGUFValueType.FLOAT32, "f", 1e-5),
        _kv_scalar("qwen35moe.rope.freq_base", GGUFValueType.FLOAT32, "f", 1000000.0),
        _kv_scalar("qwen35moe.rope.dimension_count", GGUFValueType.UINT32, "I", ROTARY_DIM),
        _kv_scalar("qwen35moe.expert_count", GGUFValueType.UINT32, "I", EXPERTS),
        _kv_scalar("qwen35moe.expert_used_count", GGUFValueType.UINT32, "I", TOPK),
        _kv_scalar("qwen35moe.expert_feed_forward_length", GGUFValueType.UINT32, "I", INTER),
        _kv_scalar("qwen35moe.expert_shared_feed_forward_length", GGUFValueType.UINT32, "I", SHARED_INTER),
        _kv_scalar("qwen35moe.ssm.conv_kernel", GGUFValueType.UINT32, "I", CONV_KERNEL),
        _kv_scalar("qwen35moe.ssm.state_size", GGUFValueType.UINT32, "I", KEY_HEAD_DIM),
        _kv_scalar("qwen35moe.ssm.group_count", GGUFValueType.UINT32, "I", NUM_KEY_HEADS),
        _kv_scalar("qwen35moe.ssm.time_step_rank", GGUFValueType.UINT32, "I", NUM_VALUE_HEADS),
        _kv_scalar("qwen35moe.ssm.inner_size", GGUFValueType.UINT32, "I", INNER_SIZE),
        _kv_scalar("qwen35moe.full_attention_interval", GGUFValueType.UINT32, "I", FULL_ATTENTION_INTERVAL),
        _kv_scalar("qwen35moe.vocab_size", GGUFValueType.UINT32, "I", VOCAB),
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

    # GGUF ne-order (fastest-varying first) -- reversed PyTorch shape.
    add("token_embd.weight", (HIDDEN, VOCAB))
    add("output_norm.weight", (HIDDEN,))
    add("output.weight", (HIDDEN, VOCAB))
    for layer in range(LAYERS):
        p = f"blk.{layer}"
        add(f"{p}.attn_norm.weight", (HIDDEN,))
        add(f"{p}.post_attention_norm.weight", (HIDDEN,))
        if layer in FULL_ATTENTION_LAYERS:
            add(f"{p}.attn_q.weight", (HIDDEN, NUM_HEADS * HEAD_DIM * 2))
            add(f"{p}.attn_q_norm.weight", (HEAD_DIM,))
            add(f"{p}.attn_k.weight", (HIDDEN, KV_HEADS * HEAD_DIM))
            add(f"{p}.attn_k_norm.weight", (HEAD_DIM,))
            add(f"{p}.attn_v.weight", (HIDDEN, KV_HEADS * HEAD_DIM))
            add(f"{p}.attn_output.weight", (NUM_HEADS * HEAD_DIM, HIDDEN))
        else:
            add(f"{p}.attn_qkv.weight", (HIDDEN, CONV_DIM))
            add(f"{p}.attn_gate.weight", (HIDDEN, VALUE_DIM))
            add(f"{p}.ssm_alpha.weight", (HIDDEN, NUM_VALUE_HEADS))
            add(f"{p}.ssm_beta.weight", (HIDDEN, NUM_VALUE_HEADS))
            add(f"{p}.ssm_conv1d.weight", (CONV_KERNEL, CONV_DIM))
            add(f"{p}.ssm_dt.bias", (NUM_VALUE_HEADS,))
            add(f"{p}.ssm_a", (NUM_VALUE_HEADS,))
            add(f"{p}.ssm_norm.weight", (VALUE_HEAD_DIM,))
            add(f"{p}.ssm_out.weight", (VALUE_DIM, HIDDEN))
        add(f"{p}.ffn_gate_inp.weight", (HIDDEN, EXPERTS))
        add(f"{p}.ffn_gate_inp_shexp.weight", (HIDDEN,))
        add(f"{p}.ffn_gate_shexp.weight", (HIDDEN, SHARED_INTER))
        add(f"{p}.ffn_up_shexp.weight", (HIDDEN, SHARED_INTER))
        add(f"{p}.ffn_down_shexp.weight", (SHARED_INTER, HIDDEN))
        # MoE expert banks: ne-order [hidden, inter, experts] / [inter, hidden, experts].
        add(f"{p}.ffn_gate_exps.weight", (HIDDEN, INTER, EXPERTS))
        add(f"{p}.ffn_up_exps.weight", (HIDDEN, INTER, EXPERTS))
        add(f"{p}.ffn_down_exps.weight", (INTER, HIDDEN, EXPERTS))

    header = _build_minimal_gguf(kv=kv, tensors=tensors)
    path = tmp_path / "tiny-qwen35moe.gguf"
    path.write_bytes(header + b"".join(data_chunks))
    return str(path)


def _build_minimal_gguf(*, version: int = 3, kv: list, tensors: list, alignment: int | None = None) -> bytes:
    """Copy of ``test_gguf_loader_wiring.py``'s own helper (itself a copy of
    ``test_gguf_reader.py``'s (#270)) -- builds everything up to (not
    including) the tensor-data bytes, which the caller appends immediately
    after."""
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
    return _build_synthetic_qwen35moe_gguf(tmp_path_factory.mktemp("gguf-qwen35moe"))


def test_parse_config_detects_real_layer_types_not_just_the_interval_formula(synthetic_gguf_path):
    # full_attention_interval=4 over 6 layers predicts ONLY layer 3 as
    # full-attention; layer 5 is deliberately full-attention here too
    # (mirroring the real target checkpoint's own "one extra full-attention
    # layer the formula doesn't predict" shape) -- parse_config must read
    # the actual tensor names (via model_path) to get this right.
    cfg = parse_config(synthetic_gguf_path, model_path=synthetic_gguf_path)
    layer_types = cfg.attrs["text_config"]["layer_types"]
    assert layer_types == [
        "full_attention" if i in FULL_ATTENTION_LAYERS else "linear_attention" for i in range(LAYERS)
    ]
    assert cfg.attrs["text_config"]["linear_num_key_heads"] == NUM_KEY_HEADS
    assert cfg.attrs["text_config"]["linear_num_value_heads"] == NUM_VALUE_HEADS
    assert cfg.attrs["text_config"]["linear_key_head_dim"] == KEY_HEAD_DIM
    assert cfg.attrs["text_config"]["linear_value_head_dim"] == VALUE_HEAD_DIM
    assert cfg.attrs["text_config"]["linear_conv_kernel_dim"] == CONV_KERNEL
    assert cfg.attrs["text_config"]["partial_rotary_factor"] == pytest.approx(ROTARY_DIM / HEAD_DIM)
    assert cfg.attrs["text_config"]["shared_expert_intermediate_size"] == SHARED_INTER
    assert cfg.attrs["head_dim"] == HEAD_DIM
    assert cfg.attrs["rope_theta"] == pytest.approx(1000000.0)


def test_iter_weights_yields_correctly_named_and_shaped_tensors(synthetic_gguf_path):
    seen = dict(iter_weights(synthetic_gguf_path, torch.device("cpu")))

    assert seen["model.embed_tokens.weight"].shape == (VOCAB, HIDDEN)
    assert seen["model.norm.weight"].shape == (HIDDEN,)
    assert seen["lm_head.weight"].shape == (VOCAB, HIDDEN)
    for layer in range(LAYERS):
        p = f"model.layers.{layer}"
        assert seen[f"{p}.input_layernorm.weight"].shape == (HIDDEN,)
        assert seen[f"{p}.post_attention_layernorm.weight"].shape == (HIDDEN,)
        assert seen[f"{p}.mlp.gate.weight"].shape == (EXPERTS, HIDDEN)
        assert seen[f"{p}.mlp.experts.gate_up_proj"].shape == (EXPERTS, 2 * INTER, HIDDEN)
        assert seen[f"{p}.mlp.experts.down_proj"].shape == (EXPERTS, HIDDEN, INTER)
        assert seen[f"{p}.mlp.experts.gate_up_proj"].device.type == "cpu"
        assert seen[f"{p}.mlp.shared_expert_gate.weight"].shape == (1, HIDDEN)
        assert seen[f"{p}.mlp.shared_expert.gate_proj.weight"].shape == (SHARED_INTER, HIDDEN)
        assert seen[f"{p}.mlp.shared_expert.up_proj.weight"].shape == (SHARED_INTER, HIDDEN)
        assert seen[f"{p}.mlp.shared_expert.down_proj.weight"].shape == (HIDDEN, SHARED_INTER)
        if layer in FULL_ATTENTION_LAYERS:
            assert seen[f"{p}.self_attn.q_proj.weight"].shape == (NUM_HEADS * HEAD_DIM * 2, HIDDEN)
            assert seen[f"{p}.self_attn.q_norm.weight"].shape == (HEAD_DIM,)
            assert seen[f"{p}.self_attn.k_proj.weight"].shape == (KV_HEADS * HEAD_DIM, HIDDEN)
            assert seen[f"{p}.self_attn.v_proj.weight"].shape == (KV_HEADS * HEAD_DIM, HIDDEN)
            assert seen[f"{p}.self_attn.o_proj.weight"].shape == (HIDDEN, NUM_HEADS * HEAD_DIM)
            assert f"{p}.linear_attn.in_proj_qkv.weight" not in seen
        else:
            assert seen[f"{p}.linear_attn.in_proj_qkv.weight"].shape == (CONV_DIM, HIDDEN)
            assert seen[f"{p}.linear_attn.in_proj_z.weight"].shape == (VALUE_DIM, HIDDEN)
            assert seen[f"{p}.linear_attn.in_proj_a.weight"].shape == (NUM_VALUE_HEADS, HIDDEN)
            assert seen[f"{p}.linear_attn.in_proj_b.weight"].shape == (NUM_VALUE_HEADS, HIDDEN)
            assert seen[f"{p}.linear_attn.conv1d.weight"].shape == (CONV_DIM, 1, CONV_KERNEL)
            assert seen[f"{p}.linear_attn.dt_bias"].shape == (NUM_VALUE_HEADS,)
            assert seen[f"{p}.linear_attn.A_log"].shape == (NUM_VALUE_HEADS,)
            assert seen[f"{p}.linear_attn.norm.weight"].shape == (VALUE_HEAD_DIM,)
            assert seen[f"{p}.linear_attn.out_proj.weight"].shape == (HIDDEN, VALUE_DIM)
            assert f"{p}.self_attn.q_proj.weight" not in seen


def test_iter_weights_respects_include_flags(synthetic_gguf_path):
    dense_only = dict(iter_weights(synthetic_gguf_path, torch.device("cpu"), include_moe_experts=False))
    assert "model.layers.0.mlp.experts.gate_up_proj" not in dense_only
    assert "model.embed_tokens.weight" in dense_only
    assert "model.layers.0.linear_attn.in_proj_qkv.weight" in dense_only
    assert "model.layers.0.mlp.shared_expert.gate_proj.weight" in dense_only

    experts_only = dict(iter_weights(synthetic_gguf_path, torch.device("cpu"), include_non_moe=False))
    assert "model.layers.0.mlp.experts.gate_up_proj" in experts_only
    assert "model.embed_tokens.weight" not in experts_only
    assert "model.layers.0.linear_attn.in_proj_qkv.weight" not in experts_only


def test_load_model_end_to_end_produces_finite_parameters(synthetic_gguf_path):
    model, expert_sources = load_model(synthetic_gguf_path, torch.device("cpu"), dtype=torch.float32)
    gate_up_banks, down_banks = expert_sources
    assert len(gate_up_banks) == LAYERS
    for p in model.parameters():
        assert torch.isfinite(p).all(), "GGUF-loaded qwen35moe model has a non-finite parameter"


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
    serving path: ``load_model()`` loads the real (hybrid, fused-QKV,
    shared-expert) ``qwen35moe`` GGUF tensor set, and ``Engine.generate()``
    (prefill + decode, through both the Gated-Delta-Net and full-attention
    layers) produces finite, in-vocabulary greedy tokens.
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
    """The offload backend routes the shared-expert-alongside-routed-expert
    MoE block through the same ``_attach_offload_cache`` path #273/#278
    already validated for ``qwen3moe`` -- confirm it also works for
    ``qwen35moe``'s hybrid layers + always-on shared expert.
    """
    reset_global_ctx()
    engine = Engine(_engine_config(synthetic_gguf_path, moe_backend="offload"))
    assert engine.model.moe_cache is not None
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
