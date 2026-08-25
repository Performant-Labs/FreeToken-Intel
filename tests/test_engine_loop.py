"""Tests for the engine loop (issue ``engine-loop``, #14).

The contract: ``Engine`` wires the (real) model forward, the paged KV pool, the
reference attention backend, and the sampler into a prefill/decode loop. The
acceptance bar is that a **dummy-weight** forward produces tokens (even
garbage) without crashing, and that ``Engine.generate`` runs the greedy
sampling path end to end.

These tests fabricate a tiny Qwen3-MoE checkpoint on disk (offline, no
network) and run the engine on the CPU, so they are machine-independent.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.xpu

from freetoken.core import Req, SamplingParams, reset_global_ctx
from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.engine.engine import Engine

# The tests build the model explicitly on the CPU (float32) so they run on any
# machine -- an XPU box would otherwise default the model to the GPU, which is
# the XPU-specific path exercised separately.
DEVICE = "cpu"


@pytest.fixture(autouse=True)
def _clean_global_ctx():
    # Each Engine installs a global context; clear it between tests so two
    # engines in one process never collide.
    yield
    reset_global_ctx()

# A tiny Qwen3-MoE: 2 layers, 4 experts, small hidden/vocab -- cheap to build
# and run on a CPU.
TINY_CONFIG = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "model_type": "qwen3_moe",
    "hidden_size": 64,
    "vocab_size": 32,
    "num_hidden_layers": 2,
    "num_local_experts": 4,
    "num_experts_per_tok": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 32,
    "moe_intermediate_size": 32,
    "max_position_embeddings": 128,
    "rope_theta": 1000000.0,
}


def _write_tiny_checkpoint(tmp_path) -> str:
    """Write a minimal Qwen3-MoE safetensors checkpoint (dummy weights).

    The engine's ``use_dummy_weight`` path fabricates the MoE expert banks from
    the config; the dense weights are still read from the checkpoint, so we
    write every dense parameter as a small zero tensor.
    """
    from freetoken.models.qwen3_moe import parse_config
    from safetensors.torch import save_file

    model_path = tmp_path / "ckpt"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(TINY_CONFIG))
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {}})
    )

    config = parse_config(type("Hf", (), {"to_dict": lambda self: TINY_CONFIG})())
    state = {}

    def add(name, shape):
        state[name] = torch.zeros(shape, dtype=torch.float32)

    hs, vocab = config.hidden_size, config.vocab_size
    heads, kv = config.num_attention_heads, config.num_key_value_heads
    head_dim = hs // heads
    layers = config.num_layers
    experts, inter = config.num_experts, config.moe_intermediate_size

    add("model.embed_tokens.weight", (vocab, hs))
    for l in range(layers):
        prefix = f"model.layers.{l}"
        add(f"{prefix}.input_layernorm.weight", (hs,))
        add(f"{prefix}.self_attn.q_proj.weight", (hs, heads * head_dim))
        add(f"{prefix}.self_attn.k_proj.weight", (hs, kv * head_dim))
        add(f"{prefix}.self_attn.v_proj.weight", (hs, kv * head_dim))
        add(f"{prefix}.self_attn.o_proj.weight", (heads * head_dim, hs))
        add(f"{prefix}.self_attn.q_norm.weight", (head_dim,))
        add(f"{prefix}.self_attn.k_norm.weight", (head_dim,))
        add(f"{prefix}.post_attention_layernorm.weight", (hs,))
        add(f"{prefix}.mlp.gate.weight", (hs, experts))
        for e in range(experts):
            add(f"{prefix}.mlp.experts.{e}.gate_proj.weight", (inter, hs))
            add(f"{prefix}.mlp.experts.{e}.up_proj.weight", (inter, hs))
            add(f"{prefix}.mlp.experts.{e}.down_proj.weight", (hs, inter))
    add("model.norm.weight", (hs,))
    add("lm_head.weight", (vocab, hs))

    # Dense weights (the loader streams these); experts are dummy-fabricated.
    dense = {k: v for k, v in state.items() if ".experts." not in k}
    save_file(dense, str(model_path / "model.safetensors"))
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
        use_dummy_weight=True,
    )


def _add_prompt(engine: Engine, output_len: int) -> None:
    engine.add_request(
        Req(
            input_ids=[1, 2, 3],
            table_idx=0,
            cached_len=0,
            output_len=output_len,
            uid=0,
            sampling_params=SamplingParams(temperature=0.0, max_tokens=output_len),
            cache_handle=None,
        )
    )


def test_engine_generate_prefill_and_decode(tmp_path):
    model_path = _write_tiny_checkpoint(tmp_path)
    engine = Engine(_engine_config(model_path, device=DEVICE))
    _add_prompt(engine, output_len=4)
    generated = engine.generate()
    vocab = engine.config.model_config.vocab_size
    assert len(generated) == 1
    # output_len=4 -> exactly 4 generated tokens, each in [0, vocab).
    assert len(generated[0]) == 4
    assert all(0 <= t < vocab for t in generated[0])


def test_engine_greedy_is_deterministic(tmp_path):
    model_path = _write_tiny_checkpoint(tmp_path)

    def build():
        # The dummy MoE experts are fabricated from a seed derived from the model
        # *config* (see loader._seed_dummy_experts), and the dense weights are
        # zeroed, so two builds of the same checkpoint produce identical weights
        # no matter what RNG state the process is in. Greedy (temperature=0) is
        # then a pure function of (weights, prompt) -> identical tokens.
        engine = Engine(_engine_config(model_path, device=DEVICE))
        _add_prompt(engine, output_len=3)
        return engine.generate()

    a = build()
    b = build()
    assert a == b
    assert len(a[0]) == 3
