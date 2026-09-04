"""End-to-end: LFM2(.5)-MoE full model wiring (issue `models-lfm2moe-e2e`,
#232, final child of the LFM2 epic #229).

Small synthetic checkpoint only -- real-checkpoint validation against
`LiquidAI/LFM2.5-8B-A1B-Base` is sequenced separately by the parent session
(deliberately out of scope here, per the issue's own body). Mirrors the
established per-model e2e pattern (`tests/test_models_mellum_e2e.py`):
seeded weights, a real `Engine` prefill+decode run, deterministic greedy.

The tiny config deliberately exercises EVERY structural novelty of this
architecture in one model (all confirmed against the real
LFM2.5-8B-A1B-Base `config.json`, scaled down):

* `layer_types` alternates conv / full_attention (attention every 4th
  starting at index 2 -- the real checkpoint's own pattern) -- BOTH layer
  kinds genuinely run in the e2e forward;
* `num_dense_layers=1` -- layer 0 is a dense w1/w3/w2 MLP, the rest MoE;
* `use_expert_bias=True` + `norm_topk_prob=True` -- the bias-corrected
  router runs in the e2e forward;
* `tie_word_embeddings=True` with NO `lm_head.weight` key in the
  checkpoint -- `iter_weights` must synthesize it (an unfilled `lm_head`
  silently zeros every logit);
* the conv filter is written in the REAL checkpoint layout
  (`conv.conv.weight` `[hidden, 1, kernel]`, an `nn.Conv1d` weight) and
  the fused experts in the REAL packed layout
  (`feed_forward.experts.gate_up_proj` `[E, 2I, H]`) -- `iter_weights`'
  key normalization + the packed-bank streamer path are both exercised.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from freetoken.core import Req, SamplingParams, reset_global_ctx
from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.engine.engine import Engine
from freetoken.models.register import get_model_class

DEVICE = "cpu"

H, V, L = 32, 64, 8
NH, KVH, HEAD_DIM = 4, 2, 8
INTER, MOE_INTER = 48, 24
E, TOPK = 4, 2
K = 3
NUM_DENSE = 1
LAYER_TYPES = ["conv", "conv", "full_attention", "conv", "conv", "conv", "full_attention", "conv"]

TINY_CONFIG = {
    "architectures": ["Lfm2MoeForCausalLM"],
    "model_type": "lfm2_moe",
    "hidden_size": H,
    "vocab_size": V,
    "num_hidden_layers": L,
    "num_attention_heads": NH,
    "num_key_value_heads": KVH,
    "intermediate_size": INTER,
    "moe_intermediate_size": MOE_INTER,
    "num_experts": E,
    "num_experts_per_tok": TOPK,
    "num_dense_layers": NUM_DENSE,
    "layer_types": LAYER_TYPES,
    "conv_L_cache": K,
    "conv_bias": False,
    "norm_topk_prob": True,
    "use_expert_bias": True,
    "routed_scaling_factor": 1.0,
    "max_position_embeddings": 128,
    "norm_eps": 1e-5,
    "tie_word_embeddings": True,
    "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
}


@pytest.fixture(autouse=True)
def _clean_global_ctx():
    yield
    reset_global_ctx()


def _config():
    from freetoken.models.lfm2_moe import parse_config

    return parse_config(type("Hf", (), {"to_dict": lambda self: TINY_CONFIG})())


def _write_tiny_checkpoint(tmp_path) -> str:
    """A real-layout LFM2 checkpoint: HF key spelling (``feed_forward``,
    ``conv.conv.weight`` as an nn.Conv1d ``[H, 1, K]``, fused packed
    experts), tied lm_head (no lm_head key at all)."""
    from safetensors.torch import save_file

    model_path = tmp_path / "ckpt"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(TINY_CONFIG))

    state = {}
    gen = torch.Generator().manual_seed(0)

    def add(name, shape):
        state[name] = torch.randn(shape, generator=gen, dtype=torch.float32) * 0.02

    add("model.embed_tokens.weight", (V, H))
    for l in range(L):
        p = f"model.layers.{l}"
        add(f"{p}.operator_norm.weight", (H,))
        add(f"{p}.ffn_norm.weight", (H,))
        if LAYER_TYPES[l] == "full_attention":
            add(f"{p}.self_attn.q_proj.weight", (NH * HEAD_DIM, H))
            add(f"{p}.self_attn.k_proj.weight", (KVH * HEAD_DIM, H))
            add(f"{p}.self_attn.v_proj.weight", (KVH * HEAD_DIM, H))
            add(f"{p}.self_attn.out_proj.weight", (H, NH * HEAD_DIM))
            add(f"{p}.self_attn.q_layernorm.weight", (HEAD_DIM,))
            add(f"{p}.self_attn.k_layernorm.weight", (HEAD_DIM,))
        else:
            add(f"{p}.conv.in_proj.weight", (3 * H, H))
            add(f"{p}.conv.conv.weight", (H, 1, K))  # real nn.Conv1d layout
            add(f"{p}.conv.out_proj.weight", (H, H))
        if l < NUM_DENSE:
            add(f"{p}.feed_forward.w1.weight", (INTER, H))
            add(f"{p}.feed_forward.w3.weight", (INTER, H))
            add(f"{p}.feed_forward.w2.weight", (H, INTER))
        else:
            add(f"{p}.feed_forward.gate.weight", (E, H))
            add(f"{p}.feed_forward.experts.gate_up_proj", (E, 2 * MOE_INTER, H))
            add(f"{p}.feed_forward.experts.down_proj", (E, H, MOE_INTER))
    add("model.embedding_norm.weight", (H,))
    # No lm_head.weight: tie_word_embeddings=True (the real checkpoint's own
    # spelling) -- iter_weights must synthesize it from embed_tokens.

    save_file(state, str(model_path / "model.safetensors"))
    return str(model_path)


def _engine_config(model_path: str, *, device: str | None = None) -> EngineConfig:
    return EngineConfig(
        model_path=model_path,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.float32,
        device=device,
        attention_backend="auto",
        max_running_req=2,
        page_size=1,
        max_seq_len_override=32,
    )


def _add_prompt(engine: Engine, output_len: int, uid: int = 0) -> None:
    engine.add_request(
        Req(
            input_ids=[1, 2, 3, 4, 5],
            table_idx=0,
            cached_len=0,
            output_len=output_len,
            uid=uid,
            sampling_params=SamplingParams(temperature=0.0, max_tokens=output_len),
            cache_handle=None,
        )
    )


def test_model_class_builds_hybrid_conv_attention_and_dense_moe_layers():
    config = _config()
    model = get_model_class("Lfm2MoeForCausalLM", config, device=torch.device("cpu"))
    for i, layer in enumerate(model.layers):
        assert layer.layer_type == LAYER_TYPES[i]
        if LAYER_TYPES[i] == "full_attention":
            assert layer.self_attn is not None and layer.conv is None
        else:
            assert layer.conv is not None and layer.self_attn is None
        assert isinstance(layer.mlp.layer_id, int)
    dense = model.layers[0].mlp
    assert hasattr(dense, "w1") and hasattr(dense, "w3") and hasattr(dense, "w2")
    moe = model.layers[1].mlp
    assert len(moe.experts) == E
    assert moe.gate.top_k == TOPK
    assert moe.gate.use_expert_bias  # the real checkpoint's own flag
    # The conv-state pool is registered for exactly the conv layers.
    assert set(model.conv_state_pool._layers) == {i for i, t in enumerate(LAYER_TYPES) if t == "conv"}


def test_iter_weights_normalizes_real_checkpoint_layout(tmp_path):
    from freetoken.models.lfm2_moe import iter_weights

    model_path = _write_tiny_checkpoint(tmp_path)
    device = torch.device("cpu")
    all_names = {}
    for name, tensor in iter_weights(model_path, device):
        all_names[name] = tensor
    # feed_forward -> mlp (dense MLP, router, and fused experts alike).
    assert "model.layers.0.mlp.w1.weight" in all_names
    assert "model.layers.1.mlp.gate.weight" in all_names
    assert "model.layers.1.mlp.experts.gate_up_proj" in all_names
    assert all(".feed_forward." not in n for n in all_names)
    # conv.conv.weight [H, 1, K] -> conv.conv_weight [H, K].
    cw = all_names["model.layers.0.conv.conv_weight"]
    assert cw.shape == (H, K)
    # Tied lm_head synthesized from embed_tokens.
    assert torch.equal(all_names["lm_head.weight"], all_names["model.embed_tokens.weight"])
    # The expert/dense split filters are real filters, not no-ops.
    expert_only = {n for n, _ in iter_weights(model_path, device, include_non_moe=False)}
    dense_only = {n for n, _ in iter_weights(model_path, device, include_moe_experts=False)}
    assert expert_only and all(".experts." in n for n in expert_only)
    assert dense_only and all(".experts." not in n for n in dense_only)
    assert expert_only | dense_only == set(all_names)


def test_short_conv_stateful_chunks_match_stateless():
    """The engine-loop contract: a chunked stateful run (fresh ring zeroed,
    then carried) must equal the stateless full-sequence forward."""
    from freetoken.models.lfm2_moe import ShortConv

    gen = torch.Generator().manual_seed(1)
    conv_a = ShortConv(H, K, has_bias=True, dtype=torch.float32)
    conv_b = ShortConv(H, K, has_bias=True, dtype=torch.float32)
    conv_b.load_state_dict(conv_a.state_dict())
    # ShortConv's conv_weight starts as torch.empty (the loader always fills
    # it) -- give both copies the same sane init so the comparison below is
    # about the chunking math, not uninitialized-memory magnitudes.
    with torch.no_grad():
        conv_a.conv_weight.normal_(0, 0.02, generator=gen)
        if conv_a.conv_bias is not None:
            conv_a.conv_bias.normal_(0, 0.02, generator=gen)
    conv_b.load_state_dict(conv_a.state_dict())
    x = torch.randn(11, H, generator=gen)

    stateless = conv_a(x)

    state = torch.zeros(H, K - 1)
    chunks = []  # prefill chunk, mid chunk, then single-token decode steps
    for size in (5, 2, 1, 1, 1, 1):
        chunks.append(size)
    assert sum(chunks) == x.shape[0]
    outs, offset = [], 0
    for size in chunks:
        outs.append(conv_b.forward_stateful(x[offset : offset + size], state))
        offset += size
    stateful = torch.cat(outs, dim=0)
    assert torch.allclose(stateless, stateful, atol=1e-5, rtol=1e-5)


def test_engine_generate_prefill_and_decode(tmp_path):
    model_path = _write_tiny_checkpoint(tmp_path)
    engine = Engine(_engine_config(model_path, device=DEVICE))
    _add_prompt(engine, output_len=4)
    generated = engine.generate()
    vocab = engine.config.model_config.vocab_size
    assert len(generated) == 1
    assert len(generated[0]) == 4
    assert all(0 <= t < vocab for t in generated[0])


def test_engine_greedy_is_deterministic(tmp_path):
    model_path = _write_tiny_checkpoint(tmp_path)

    def build():
        engine = Engine(_engine_config(model_path, device=DEVICE))
        _add_prompt(engine, output_len=3)
        return engine.generate()

    a = build()
    b = build()
    assert a == b
    assert len(a[0]) == 3


def test_engine_second_request_reusing_a_slot_matches_a_fresh_engine(tmp_path):
    """A recycled table row must not leak the previous request's conv ring:
    the SECOND request through the same engine (same table_idx after the
    first completes) generates exactly what a cold engine generates."""
    model_path = _write_tiny_checkpoint(tmp_path)
    engine = Engine(_engine_config(model_path, device=DEVICE))
    _add_prompt(engine, output_len=3, uid=0)
    first = engine.generate()
    _add_prompt(engine, output_len=3, uid=1)
    second = engine.generate()

    cold = Engine(_engine_config(model_path, device=DEVICE))
    _add_prompt(cold, output_len=3, uid=0)
    reference = cold.generate()
    assert second == reference
    assert first == reference
