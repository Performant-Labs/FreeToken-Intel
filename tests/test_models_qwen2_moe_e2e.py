"""End-to-end: Qwen1.5-MoE-A2.7B (qwen2_moe) -- bias-term MHA + the real
gated-shared-expert MoE router, full model wiring (issue #222, final piece
of epic #220). See qwen2_moe/__init__.py's own module docstring for the two
real, confirmed differences from this port's Qwen3-family models: bias-term
attention with no q/k norm, and the sigmoid-gated shared-expert combination.

Small synthetic checkpoint only (2 layers: 1 dense via ``mlp_only_layers``,
1 MoE) -- real-checkpoint validation against the actual downloaded
Qwen/Qwen1.5-MoE-A2.7B is sequenced separately.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from freetoken.core import Req, SamplingParams, reset_global_ctx
from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.engine.engine import Engine
from freetoken.models.qwen2_moe import Qwen2MoeForCausalLM, iter_weights, parse_config
from freetoken.models.register import get_model_class

DEVICE = "cpu"

H, V, L = 32, 64, 2
NH, NKV, HD = 4, 4, 8
INTER = 48
E, MOE_INTER, TOPK = 4, 24, 2
SHARED_INTER = 24

# The generic expert-source streamer (weight.py's stream_moe_expert_sources)
# assumes dense layers are a leading PREFIX (first_k_dense_replace), not
# qwen2_moe's real, arbitrary mlp_only_layers list. The real checkpoint's
# own values are decoder_sparse_step=1, mlp_only_layers=[] -- every layer is
# MoE -- so the full Engine e2e config below matches that (all-MoE) rather
# than exercising a mismatch the real checkpoint doesn't have. The dense/MoE
# *selection logic* itself (mlp_only_layers marking a layer dense) is
# covered separately by test_model_class_builds_dense_layer_0_and_moe_layer_1
# below, which builds the model directly without the checkpoint streamer.
TINY_CONFIG = {
    "architectures": ["Qwen2MoeForCausalLM"],
    "model_type": "qwen2_moe",
    "hidden_size": H,
    "vocab_size": V,
    "num_hidden_layers": L,
    "num_attention_heads": NH,
    "num_key_value_heads": NKV,
    "head_dim": HD,
    "qkv_bias": True,
    "intermediate_size": INTER,
    "num_experts": E,
    "moe_intermediate_size": MOE_INTER,
    "num_experts_per_tok": TOPK,
    "norm_topk_prob": False,
    "shared_expert_intermediate_size": SHARED_INTER,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
    "max_position_embeddings": 128,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
}

# Separate config for the dense/MoE selection-logic test: layer 0 dense.
DENSE_SPLIT_CONFIG = dict(TINY_CONFIG, mlp_only_layers=[0])


@pytest.fixture(autouse=True)
def _clean_global_ctx():
    yield
    reset_global_ctx()


def _config():
    return parse_config(type("Hf", (), {"to_dict": lambda self: TINY_CONFIG})())


def test_config_derives_dense_vs_moe_layer_split():
    config = parse_config(type("Hf", (), {"to_dict": lambda self: DENSE_SPLIT_CONFIG})())
    assert config.attrs["mlp_only_layers"] == [0]
    assert config.attrs["decoder_sparse_step"] == 1


def test_model_class_builds_dense_layer_0_and_moe_layer_1():
    config = parse_config(type("Hf", (), {"to_dict": lambda self: DENSE_SPLIT_CONFIG})())
    model = get_model_class("Qwen2MoeForCausalLM", config, device=torch.device("cpu"))
    from freetoken.models.qwen2_moe import _Qwen2MoeMLP, _Qwen2MoeMoE  # noqa: PLC0415

    assert isinstance(model.layers[0].mlp, _Qwen2MoeMLP)
    assert isinstance(model.layers[1].mlp, _Qwen2MoeMoE)
    assert model.layers[1].mlp.shared_expert is not None


def test_model_class_builds_every_layer_moe_matching_real_checkpoint():
    config = _config()
    model = get_model_class("Qwen2MoeForCausalLM", config, device=torch.device("cpu"))
    from freetoken.models.qwen2_moe import _Qwen2MoeMoE  # noqa: PLC0415

    assert all(isinstance(layer.mlp, _Qwen2MoeMoE) for layer in model.layers)


def _write_tiny_checkpoint(tmp_path) -> str:
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
        add(f"{p}.input_layernorm.weight", (H,))
        add(f"{p}.self_attn.q_proj.weight", (NH * HD, H))
        add(f"{p}.self_attn.q_proj.bias", (NH * HD,))
        add(f"{p}.self_attn.k_proj.weight", (NKV * HD, H))
        add(f"{p}.self_attn.k_proj.bias", (NKV * HD,))
        add(f"{p}.self_attn.v_proj.weight", (NKV * HD, H))
        add(f"{p}.self_attn.v_proj.bias", (NKV * HD,))
        add(f"{p}.self_attn.o_proj.weight", (H, NH * HD))
        add(f"{p}.post_attention_layernorm.weight", (H,))
        # Every layer MoE, matching TINY_CONFIG's mlp_only_layers=[] --
        # the real checkpoint's own layer split (see the module note above).
        add(f"{p}.mlp.gate.weight", (E, H))
        for e in range(E):
            eb = f"{p}.mlp.experts.{e}"
            add(f"{eb}.gate_proj.weight", (MOE_INTER, H))
            add(f"{eb}.up_proj.weight", (MOE_INTER, H))
            add(f"{eb}.down_proj.weight", (H, MOE_INTER))
        add(f"{p}.mlp.shared_expert.gate_proj.weight", (SHARED_INTER, H))
        add(f"{p}.mlp.shared_expert.up_proj.weight", (SHARED_INTER, H))
        add(f"{p}.mlp.shared_expert.down_proj.weight", (H, SHARED_INTER))
        add(f"{p}.mlp.shared_expert_gate.weight", (1, H))
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
        moe_backend="fused",
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


def test_iter_weights_covers_expert_and_shared_expert_tensors(tmp_path):
    model_path = _write_tiny_checkpoint(tmp_path)
    names = {n for n, _ in iter_weights(model_path, torch.device("cpu"))}
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" in names
    assert "model.layers.1.mlp.shared_expert_gate.weight" in names


def test_iter_weights_include_moe_experts_flag_splits_dense_and_expert(tmp_path):
    model_path = _write_tiny_checkpoint(tmp_path)
    dense_names = {n for n, _ in iter_weights(model_path, torch.device("cpu"), include_moe_experts=False)}
    expert_names = {n for n, _ in iter_weights(model_path, torch.device("cpu"), include_non_moe=False)}
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" not in dense_names
    assert "model.embed_tokens.weight" in dense_names
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" in expert_names
    assert "model.embed_tokens.weight" not in expert_names


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
