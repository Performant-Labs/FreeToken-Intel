"""Forward-correctness tests for the CPU MoE executor (issue #8, ADR 0002).

The contract: ``--moe-backend cpu`` runs the MoE expert GEMM on the host instead
of streaming activated experts over PCIe to the XPU. The host banks are the same
pinned source of truth the ``offload`` backend reads (ADR 0002); the only
difference is *where the GEMM runs*. So the same checkpoint loaded in-VRAM (the
reference) and on the CPU backend (under test) must produce *identical* greedy
tokens end to end (prefill + decode). A wrong bank orientation, an off-by-one in
the layer map, or a mismatched SwiGLU math changes the logits and fails the
comparison, whereas a shape-only check would pass regardless.

These tests are CPU-marked (no ``xpu`` / ``slow``), so they run in the CPU venv
on any box: a tiny Qwen3-MoE is built twice from the same fabricated checkpoint
-- once in-VRAM (the reference) and once on the CPU backend (under test) -- and
the engine's greedy output is compared.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from freetoken.core import Req, SamplingParams, reset_global_ctx
from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.engine.engine import Engine
from freetoken.models.loader import load_model
from freetoken.models.qwen3_moe import parse_config
from freetoken.models.weight import _stack_expert_rows

DEVICE = torch.device("cpu")

# A tiny Qwen3-MoE: 2 MoE layers (first_k_dense_replace=0 -> both layers are
# MoE), 4 experts, top-2, small hidden / vocab -- cheap to build and run on a
# CPU. Mirrors the dense-weight layout the loader streams.
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

# A per-expert checkpoint whose values *identify* (layer, expert, projection),
# so the reference and the CPU path provably consume the same expert bytes.
# gate (expert e, row r): 100*(e+1) + r + layer*1000; up: +1; down: +2.
# All values are small enough to be exact in float32 (the loader's dtype here).
def _distinguishable_experts() -> dict:
    hidden, inter, experts, vocab, layers = 64, 32, 4, 32, 2
    w = {
        "model.embed_tokens.weight": torch.randn(vocab, hidden),
        "lm_head.weight": torch.randn(vocab, hidden),
    }
    for layer in range(layers):
        for e in range(experts):
            prefix = f"model.layers.{layer}.mlp.experts.{e}"
            base = 100 * (e + 1) + 1000 * layer
            w[f"{prefix}.gate_proj"] = torch.arange(inter, dtype=torch.float32)[:, None].repeat(1, hidden) + base
            w[f"{prefix}.up_proj"] = torch.arange(inter, dtype=torch.float32)[:, None].repeat(1, hidden) + base + 1
            # down_proj is the [H, I] weight of nn.Linear(I, H): H rows (output),
            # I cols (input). The CPU executor reads it as a [H, I] bank row and
            # projects ``h @ down_w.t()`` -- the same orientation the in-VRAM
            # expert uses, so a wrong orientation here would diverge the logits.
            w[f"{prefix}.down_proj"] = torch.arange(hidden, dtype=torch.float32)[:, None].repeat(1, inter) + base + 2
    return w


def _dense_weights() -> dict:
    hidden, vocab, heads, kv, inter, experts, layers = 64, 32, 4, 2, 32, 4, 2
    head_dim = hidden // heads
    w = {
        "model.embed_tokens.weight": torch.randn(vocab, hidden),
        "lm_head.weight": torch.randn(vocab, hidden),
    }
    for l in range(layers):
        prefix = f"model.layers.{l}"
        w[f"{prefix}.input_layernorm.weight"] = torch.randn(hidden)
        w[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(heads * head_dim, hidden)
        w[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(kv * head_dim, hidden)
        w[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(kv * head_dim, hidden)
        w[f"{prefix}.self_attn.q_norm.weight"] = torch.randn(head_dim)
        w[f"{prefix}.self_attn.k_norm.weight"] = torch.randn(head_dim)
        w[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(heads * head_dim, hidden)
        w[f"{prefix}.post_attention_layernorm.weight"] = torch.randn(hidden)
        # The MoE router is nn.Linear(hidden -> num_experts): weight [num_experts, hidden].
        w[f"{prefix}.mlp.gate.weight"] = torch.randn(experts, hidden)
    w["model.norm.weight"] = torch.randn(hidden)
    return w


@pytest.fixture(scope="module")
def cpu_ckpt(tmp_path_factory):
    """A checkpoint with distinguishable per-expert weights + random dense weights.

    Both the in-VRAM reference and the CPU-under-test load from *this same*
    path, so they are guaranteed to consume identical dense + expert bytes.
    """
    from safetensors.torch import save_file

    path = tmp_path_factory.mktemp("qwen3moe-cpu")
    (path / "config.json").write_text(json.dumps(TINY_CONFIG))
    # Re-seed so the dense weights are a fixed, order-independent value: the
    # in-VRAM reference and the CPU-under-test both load this same file, so they
    # always consume identical dense bytes regardless of test order.
    torch.manual_seed(1234)
    weights = {**_dense_weights(), **_distinguishable_experts()}
    shards = {k: v.contiguous() for k, v in weights.items()}
    save_file(shards, str(path / "model.safetensors"))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {k: "model.safetensors" for k in shards}})
    )
    return str(path)


@pytest.fixture(autouse=True)
def _clean_global_ctx():
    # Each Engine installs a global context; clear it between tests so two
    # engines in one process never collide.
    yield
    reset_global_ctx()


def _engine_config(
    model_path: str, *, moe_backend: str | None, moe_cpu_layers: str | None = None
) -> EngineConfig:
    return EngineConfig(
        model_path=model_path,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.float32,
        device=DEVICE,
        attention_backend="auto",
        moe_backend=moe_backend,
        moe_cpu_layers=moe_cpu_layers,
        max_running_req=2,
        page_size=1,
        max_seq_len_override=32,
        num_page_override=64,
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


def test_cpu_forward_matches_in_vram_reference(cpu_ckpt):
    """The CPU expert executor must reproduce the in-VRAM forward's logits.

    The CPU backend runs the same SwiGLU expert math over the *same* host banks
    the reference's expert modules hold; only the compute location differs. So
    the same checkpoint loaded in-VRAM (reference) and on the CPU backend (under
    test) must produce identical greedy tokens end to end (prefill + decode).
    """
    # Reference: in-VRAM experts (moe_backend=None -> the default resident path).
    ref_model, _ = load_model(cpu_ckpt, DEVICE, dtype=torch.float32, moe_backend=None)
    assert not getattr(ref_model, "moe_offload", False), "reference must be the in-VRAM path"
    assert ref_model.layers[0].mlp.experts is not None, "the in-VRAM path must build resident experts"

    # Under test: CPU expert executor (ADR 0002). The experts are never
    # device-resident; the forward computes them on the host from the banks.
    cpu_model, cpu_sources = load_model(cpu_ckpt, DEVICE, dtype=torch.float32, moe_backend="cpu")
    assert cpu_model.moe_offload, "the cpu backend must flag the model as host-offloaded"
    assert cpu_model.layers[0].mlp.experts is None, "the cpu path must not build device-resident experts"
    # The host banks must be present and correctly shaped (the CPU's source of
    # truth): one [E, 2I, H] gate_up + [E, H, I] down bank per MoE layer.
    assert len(cpu_sources[0]) == 2  # two MoE layers
    assert cpu_sources[0][0].shape == (4, 2 * 32, 64)  # gate_up [E, 2I, H]
    assert cpu_sources[1][1].shape == (4, 64, 32)  # down [E, H, I]
    assert getattr(cpu_model, "moe_backend", None) == "cpu", "the loader must record the resolved cpu backend"

    ref_engine = Engine(_engine_config(cpu_ckpt, moe_backend=None))
    _add_prompt(ref_engine, output_len=6)
    ref_tokens = ref_engine.generate()

    cpu_engine = Engine(_engine_config(cpu_ckpt, moe_backend="cpu"))
    _add_prompt(cpu_engine, output_len=6)
    cpu_tokens = cpu_engine.generate()

    # Same checkpoint, same prompt -> the CPU path must match the reference.
    assert cpu_tokens == ref_tokens, (
        f"cpu logits diverged from in-VRAM reference: {cpu_tokens} != {ref_tokens}"
    )
    # Both must actually generate (non-empty, in-vocab) tokens.
    assert len(cpu_tokens[0]) == 6
    assert all(0 <= t < TINY_CONFIG["vocab_size"] for t in cpu_tokens[0])


def test_cpu_forward_is_deterministic(cpu_ckpt):
    """Two CPU runs of the same checkpoint must be identical.

    Guards against non-determinism in the CPU executor's per-expert routing (e.g.
    a dict-order dependent expert iteration, or a stale weight from a prior step
    leaking in) -- the same invariant the offload slot-pool test pins.
    """
    from freetoken.engine.engine import Engine as _Engine

    def run():
        engine = _Engine(_engine_config(cpu_ckpt, moe_backend="cpu"))
        _add_prompt(engine, output_len=5)
        return engine.generate()

    a = run()
    b = run()
    assert a == b
    assert len(a[0]) == 5


# =============================================================================
# --moe-cpu-layers partition (issue #8): the per-layer CPU/offload split the
# loader resolves and the blocks dispatch on. (The torch-free spec-parser unit
# test lives in test_moe_cpu_layers_parse.py so it runs in the CPU venv too.)
# =============================================================================


def test_moe_cpu_layers_partial_partition_matches_all_cpu(cpu_ckpt):
    """A partial --moe-cpu-layers partition must not change the forward.

    The tiny model has 2 MoE layers. Partitioning layer 0 to the CPU and leaving
    layer 1 offload-only (``--moe-cpu-layers 1`` -> first 1 MoE layer on CPU) must
    produce the *same* greedy tokens as the all-CPU reference (``--moe-backend
    cpu`` with the default all-CPU partition), because the CPU expert GEMM and the
    XPU-offload slot pool are both exact mirrors of the in-VRAM expert math (ADR
    0002 / issue #7). A partition that mis-wires the layer map, the host banks, or
    the per-layer dispatch would route a layer's tokens through the wrong expert
    bytes and the logits would diverge.
    """
    # Reference: every MoE layer on the CPU (the --moe-backend=cpu default).
    ref_engine = Engine(_engine_config(cpu_ckpt, moe_backend="cpu"))
    _add_prompt(ref_engine, output_len=6)
    ref_tokens = ref_engine.generate()

    # Under test: only the first MoE layer (index 0) on the CPU; layer 1 stays on
    # the XPU offload slot pool. Same checkpoint, same prompt.
    partial_engine = Engine(_engine_config(cpu_ckpt, moe_backend="cpu", moe_cpu_layers="1"))
    _add_prompt(partial_engine, output_len=6)
    partial_tokens = partial_engine.generate()

    assert partial_tokens == ref_tokens, (
        f"partial --moe-cpu-layers partition diverged from the all-CPU reference: "
        f"{partial_tokens} != {ref_tokens}"
    )
    assert len(partial_tokens[0]) == 6


def test_moe_cpu_layers_all_offload_still_correct(cpu_ckpt):
    """--moe-cpu-layers 0 (no CPU layers) must still match the in-VRAM reference.

    The partition is "empty": every MoE layer runs on the XPU offload slot pool
    (the ADR 0002 path), none on the CPU. The forward must still be an exact
    mirror of the in-VRAM reference -- the same invariant the offload slot-pool
    test pins -- so a regression in the per-layer dispatch (e.g. a layer that
    should be offload accidentally routed to the CPU, or vice versa) fails here.
    """
    ref_model, _ = load_model(cpu_ckpt, DEVICE, dtype=torch.float32, moe_backend=None)
    ref_engine = Engine(_engine_config(cpu_ckpt, moe_backend=None))
    _add_prompt(ref_engine, output_len=6)
    ref_tokens = ref_engine.generate()

    offload_engine = Engine(_engine_config(cpu_ckpt, moe_backend="offload", moe_cpu_layers="0"))
    _add_prompt(offload_engine, output_len=6)
    offload_tokens = offload_engine.generate()

    assert offload_tokens == ref_tokens, (
        f"--moe-cpu-layers 0 (all offload) diverged from in-VRAM reference: "
        f"{offload_tokens} != {ref_tokens}"
    )
    assert len(offload_tokens[0]) == 6
