"""The real DeepSeek-Coder-V2-Lite MoE router (issue `models-dsv2lite-moe`,
#218, second child of the DeepSeek-Coder-V2-Lite epic #216), depends on
`models-dsv2lite-mla` (#217, MLA + config parsing, already merged).

Genuinely different math from every other MoE router in this port
(`deepseek_v4`/`glm_moe_dsa`/`qwen3_5_moe`/`qwen3_moe` all use a sigmoid +
bias-corrected, optionally-grouped router): DeepSeek-Coder-V2-Lite's real
`topk_method="greedy"` router is plain softmax scores + flat top-k, no
renormalization, no bias-correction term -- confirmed byte-for-byte against
the real, installed `transformers.models.deepseek_v2.modeling_deepseek_v2`
(v5.15.1) `DeepseekV2TopkRouter`. The real checkpoint's own config.json also
confirms `n_group: 1`/`topk_group: 1` (genuinely flat, no grouping) and
`norm_topk_prob: false` (consistent with the real reference code having no
renorm branch for the greedy path at all).

Full real-checkpoint end-to-end validation is issue #219's job -- this file
stays scoped to #218's own accept bar: the router itself against a
hand-computed reference, and a small synthetic checkpoint proving the MoE
layer (router + routed experts + shared expert) wires correctly through a
real Engine prefill+decode, mirroring `deepseek_v4`'s own
`test_models_deepseek_v4_moe_e2e.py` in shape.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from freetoken.core import Req, SamplingParams, reset_global_ctx
from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.engine.engine import Engine
from freetoken.models.deepseek_v2_lite import (
    _DeepseekV2LiteMLP,
    _DeepseekV2LiteMoE,
    _DeepseekV2LiteTopkRouter,
    iter_weights,
    parse_config,
)
from freetoken.models.register import get_model_class

DEVICE = "cpu"

H, V, L = 32, 64, 3
NH = 4
KV_LORA = 16
QK_ROPE, QK_NOPE, V_HEAD = 4, 6, 8
INTER = 48
E, MOE_INTER, TOPK = 8, 24, 3
N_SHARED = 2
FIRST_K_DENSE = 1  # layer 0 dense, layers 1-2 MoE -- matches the real checkpoint's shape

TINY_CONFIG = {
    "architectures": ["DeepseekV2ForCausalLM"],
    "model_type": "deepseek_v2",
    "hidden_size": H,
    "vocab_size": V,
    "num_hidden_layers": L,
    "num_attention_heads": NH,
    "q_lora_rank": None,
    "kv_lora_rank": KV_LORA,
    "qk_rope_head_dim": QK_ROPE,
    "qk_nope_head_dim": QK_NOPE,
    "v_head_dim": V_HEAD,
    "attention_bias": False,
    "intermediate_size": INTER,
    "n_routed_experts": E,
    "moe_intermediate_size": MOE_INTER,
    "num_experts_per_tok": TOPK,
    "n_group": 1,
    "topk_group": 1,
    "routed_scaling_factor": 1.0,
    "n_shared_experts": N_SHARED,
    "norm_topk_prob": False,  # the real checkpoint's own value
    "topk_method": "greedy",  # the real checkpoint's own value
    "first_k_dense_replace": FIRST_K_DENSE,
    "max_position_embeddings": 128,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
}


@pytest.fixture(autouse=True)
def _clean_global_ctx():
    yield
    reset_global_ctx()


def _config():
    return parse_config(type("Hf", (), {"to_dict": lambda self: TINY_CONFIG})())


def test_config_carries_the_real_router_fields():
    config = _config()
    assert config.is_moe is True
    assert config.num_experts == E
    assert config.attrs["topk_method"] == "greedy"
    assert config.attrs["n_group"] == 1
    assert config.attrs["norm_topk_prob"] is False


def test_model_class_builds_dense_and_moe_layers():
    config = _config()
    model = get_model_class("DeepseekV2ForCausalLM", config, device=torch.device("cpu"))
    assert isinstance(model.layers[0].mlp, _DeepseekV2LiteMLP)  # dense (first_k_dense_replace)
    assert isinstance(model.layers[1].mlp, _DeepseekV2LiteMoE)
    assert len(model.layers[1].mlp.experts) == E
    assert model.layers[1].mlp.shared_experts is not None


def test_router_matches_hand_computed_greedy_softmax_reference():
    """The real math: plain softmax scores, flat top-k, weights scaled by
    routed_scaling_factor -- NO renormalization (see module docstring)."""
    torch.manual_seed(0)
    router = _DeepseekV2LiteTopkRouter(_config(), dtype=torch.float32)
    router.weight.data = torch.randn(E, H) * 0.1

    x = torch.randn(5, H)
    topk_indices, topk_weights = router(x)
    assert topk_indices.shape == (5, TOPK)
    assert topk_weights.shape == (5, TOPK)

    # Independent hand-computed reference.
    logits = x.float() @ router.weight.float().t()
    scores = torch.softmax(logits, dim=-1)
    expected_weights, expected_indices = torch.topk(scores, k=TOPK, dim=-1, sorted=False)
    expected_weights = expected_weights * TINY_CONFIG["routed_scaling_factor"]

    # topk with sorted=False may order ties differently; compare as sets of
    # (index, weight) pairs per row rather than assuming identical ordering.
    for row in range(5):
        got = sorted(zip(topk_indices[row].tolist(), topk_weights[row].tolist()))
        want = sorted(zip(expected_indices[row].tolist(), expected_weights[row].tolist()))
        assert got == pytest.approx(want, abs=1e-5)


def test_router_selects_the_actual_highest_softmax_scores():
    """A directly interpretable case: 8 experts, top-3 -- the 3 highest raw
    logits must be exactly the 3 selected (softmax is monotonic, so ranking
    survives it)."""
    router = _DeepseekV2LiteTopkRouter(_config(), dtype=torch.float32)
    router.weight.data = torch.eye(E, H)[: , :H]  # expert e's logit = x[e] for e < H
    x = torch.zeros(1, H)
    x[0, 0], x[0, 3], x[0, 5] = 10.0, 9.0, 8.0  # experts 0, 3, 5 should win
    topk_indices, _ = router(x)
    assert set(topk_indices[0].tolist()) == {0, 3, 5}


def test_shared_experts_output_is_unweighted_and_always_added():
    """Zero out every routed expert's weights -- only the shared-expert
    contribution should survive in the MoE block's output."""
    config = _config()
    moe = _DeepseekV2LiteMoE(config, torch.device("cpu"), torch.float32, layer_id=1)
    for e in moe.experts:
        for p in e.parameters():
            p.data.zero_()
    for p in moe.shared_experts.parameters():
        p.data.normal_(std=0.1)
    x = torch.randn(4, H)
    out = moe(x)
    expected = moe.shared_experts(x)
    torch.testing.assert_close(out, expected)


def _write_tiny_checkpoint(tmp_path) -> str:
    from safetensors.torch import save_file

    model_path = tmp_path / "ckpt"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(TINY_CONFIG))

    config = _config()
    state = {}
    gen = torch.Generator().manual_seed(0)

    def add(name, shape):
        state[name] = torch.randn(shape, generator=gen, dtype=torch.float32) * 0.02

    qk_head_dim = QK_NOPE + QK_ROPE
    add("model.embed_tokens.weight", (V, H))
    for l in range(L):
        p = f"model.layers.{l}"
        add(f"{p}.input_layernorm.weight", (H,))
        add(f"{p}.self_attn.q_proj.weight", (NH * qk_head_dim, H))
        add(f"{p}.self_attn.kv_a_proj_with_mqa.weight", (KV_LORA + QK_ROPE, H))
        add(f"{p}.self_attn.kv_a_layernorm.weight", (KV_LORA,))
        add(f"{p}.self_attn.kv_b_proj.weight", (NH * (QK_NOPE + V_HEAD), KV_LORA))
        add(f"{p}.self_attn.o_proj.weight", (H, NH * V_HEAD))
        add(f"{p}.post_attention_layernorm.weight", (H,))
        if l < FIRST_K_DENSE:
            add(f"{p}.mlp.gate_proj.weight", (INTER, H))
            add(f"{p}.mlp.up_proj.weight", (INTER, H))
            add(f"{p}.mlp.down_proj.weight", (H, INTER))
        else:
            add(f"{p}.mlp.gate.weight", (E, H))
            for e in range(E):
                eb = f"{p}.mlp.experts.{e}"
                add(f"{eb}.gate_proj.weight", (MOE_INTER, H))
                add(f"{eb}.up_proj.weight", (MOE_INTER, H))
                add(f"{eb}.down_proj.weight", (H, MOE_INTER))
            add(f"{p}.mlp.shared_experts.gate_proj.weight", (MOE_INTER * N_SHARED, H))
            add(f"{p}.mlp.shared_experts.up_proj.weight", (MOE_INTER * N_SHARED, H))
            add(f"{p}.mlp.shared_experts.down_proj.weight", (H, MOE_INTER * N_SHARED))
    add("model.norm.weight", (H,))
    add("lm_head.weight", (V, H))

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


def _add_prompt(engine: Engine, output_len: int) -> None:
    engine.add_request(
        Req(
            input_ids=[1, 2, 3, 4, 5],
            table_idx=0,
            cached_len=0,
            output_len=output_len,
            uid=0,
            sampling_params=SamplingParams(temperature=0.0, max_tokens=output_len),
            cache_handle=None,
        )
    )


def test_iter_weights_covers_dense_moe_and_shared_expert(tmp_path):
    model_path = _write_tiny_checkpoint(tmp_path)
    names = {n for n, _ in iter_weights(model_path, torch.device("cpu"))}
    assert "model.layers.0.mlp.gate_proj.weight" in names  # dense layer
    assert "model.layers.1.mlp.gate.weight" in names  # MoE router
    assert "model.layers.1.mlp.experts.0.gate_proj.weight" in names
    assert "model.layers.1.mlp.shared_experts.gate_proj.weight" in names


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


def test_moe_backend_guard_rejects_offload():
    """Fail loud, not silently wrong -- same discipline as deepseek_v4's own
    MoE block: this MoE is fused-only, offload/cpu/hybrid must raise."""
    config = _config()
    config.use_offload_moe = True
    with pytest.raises(NotImplementedError):
        _DeepseekV2LiteMoE(config, torch.device("cpu"), torch.float32, layer_id=1)
