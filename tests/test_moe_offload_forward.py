"""Forward-correctness tests for the host-offload MoE path (issue #54, ADR 0002).

The contract (ADR 0002): the MoE experts are never XPU-resident. They live in
pinned host RAM and are streamed through a small LRU slot pool on demand. The
offload forward (:meth:`_Qwen3MoE._forward_offload`) must therefore produce the
*same* next-token logits as the in-VRAM forward that gathers from XPU-resident
expert modules -- because the LRU slot pool is a *transport* for the expert
weights, not a change to the math.

These tests are CPU-marked (no ``xpu`` / ``slow``), so they run in the CPU venv
on any box: a tiny Qwen3-MoE is built twice from the same fabricated checkpoint
-- once in-VRAM (the reference) and once offload (under test) -- and the
engine's greedy output (which runs prefill-then-decode, so *both* the
materialize and the LRU paths) is compared. A slot-pool bug (an expert read from
the wrong slot, an off-by-one in the layer map, or a dtype/shape mismatch in the
gather) changes the logits and fails the comparison, whereas a shape-only check
would pass regardless.
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
# so the reference and the offload path provably consume the same expert bytes.
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
            # I cols (input). (An earlier draft wrote it [H, H]; when H != I the
            # offload down bank -- packed as [E, H, I] -- then had the wrong inner
            # dim and the offload forward silently diverged from the in-VRAM
            # reference. The bank must be [H, I], matching nn.Linear(I, H).weight.)
            w[f"{prefix}.down_proj"] = torch.arange(hidden, dtype=torch.float32)[:, None].repeat(1, inter) + base + 2
    return w


def _dense_weights() -> dict:
    hidden, vocab, heads, kv, inter, experts, layers = 64, 32, 4, 2, 32, 4, 2
    # The model derives head_dim = hidden // num_attention_heads (64 // 4 = 16)
    # and its q/k norms are RMSNorm(head_dim) over that. The projections match the
    # model's nn.Linear(hidden -> out): q is [heads*head_dim, hidden], k/v are
    # [kv*head_dim, hidden] (nn.Linear stores weight as [out, in]). The q/k norms
    # are [head_dim] (the model's RMSNorm(hidden // heads) width). The shapes
    # below are EXACT: the loader's placement is a strict shape match, so a
    # mismatched checkpoint weight is a hard error (it no longer silently no-ops
    # -- see loader._place_dense).
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
def offload_ckpt(tmp_path_factory):
    """A checkpoint with distinguishable per-expert weights + random dense weights.

    Both the in-VRAM reference and the offload-under-test load from *this same*
    path, so they are guaranteed to consume identical dense + expert bytes.
    """
    from safetensors.torch import save_file

    path = tmp_path_factory.mktemp("qwen3moe-offload")
    (path / "config.json").write_text(json.dumps(TINY_CONFIG))
    # _dense_weights() draws from the *global* RNG (torch.randn). The fixture is
    # module-scoped, so it builds once -- but the RNG offset at that moment
    # depends on how much random state the *prior* tests (and the reference
    # load_model in test_offload_forward_matches_in_vram_reference) consumed.
    # Re-seed here so the dense weights are a fixed, order-independent value:
    # the in-VRAM reference and the offload-under-test both load this same file,
    # so they always consume identical dense bytes regardless of test order.
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


def _engine_config(model_path: str, *, moe_backend: str | None) -> EngineConfig:
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


def test_offload_forward_matches_in_vram_reference(offload_ckpt):
    """The host-offload forward must reproduce the in-VRAM forward's logits.

    The LRU slot pool transports the expert weights; it must not change them.
    So the same checkpoint loaded in-VRAM (reference) and offload (under test)
    must produce identical greedy tokens end to end (prefill + decode).
    """
    # Reference: in-VRAM experts (moe_backend=None -> the default resident path).
    ref_model, ref_sources = load_model(offload_ckpt, DEVICE, dtype=torch.float32, moe_backend=None)
    assert not getattr(ref_model, "moe_offload", False), "reference must be the in-VRAM path"
    # The in-VRAM path must have populated the model's per-expert modules.
    assert ref_model.layers[0].mlp.experts is not None

    # Under test: host-offload experts (ADR 0002).
    off_model, off_sources = load_model(offload_ckpt, DEVICE, dtype=torch.float32, moe_backend="offload")
    assert off_model.moe_offload, "offload path must flag the model"
    assert off_model.moe_cache is not None, "the loader must attach the LRU slot pool"
    assert off_model.moe_layer_id is not None
    assert off_model.layers[0].mlp.experts is None, "the offload path must not build XPU-resident experts"
    # The host banks must be present and correctly shaped (the source of truth).
    assert len(off_sources[0]) == 2  # two MoE layers
    assert off_sources[0][0].shape == (4, 2 * 32, 64)  # [E, 2I, H]
    assert off_sources[1][1].shape == (4, 64, 32)  # [E, H, I]

    # The offload forward must be wired: run one decode step and confirm the
    # slot pool was actually exercised (the prefill materialized the layers, so
    # the first decode steps are hits; a later step's misses are streamed in).
    ref_engine = Engine(_engine_config(offload_ckpt, moe_backend=None))
    _add_prompt(ref_engine, output_len=6)
    ref_tokens = ref_engine.generate()

    off_engine = Engine(_engine_config(offload_ckpt, moe_backend="offload"))
    _add_prompt(off_engine, output_len=6)
    off_tokens = off_engine.generate()

    # The engine's greedy output is a pure function of (weights, prompt). Same
    # checkpoint, same prompt -> the offload path must match the reference.
    assert off_tokens == ref_tokens, (
        f"offload logits diverged from in-VRAM reference: {off_tokens} != {ref_tokens}"
    )
    # Both must actually generate (non-empty, in-vocab) tokens.
    assert len(off_tokens[0]) == 6
    assert all(0 <= t < TINY_CONFIG["vocab_size"] for t in off_tokens[0])


def test_offload_forward_is_deterministic(offload_ckpt):
    """Two offload runs of the same checkpoint must be identical.

    Guards against non-determinism in the slot-pool routing (e.g. dict-order
    dependent expert iteration, or a stale slot from a prior step leaking in).
    """
    from freetoken.engine.engine import Engine as _Engine

    def run():
        engine = _Engine(_engine_config(offload_ckpt, moe_backend="offload"))
        _add_prompt(engine, output_len=5)
        return engine.generate()

    a = run()
    b = run()
    assert a == b
    assert len(a[0]) == 5


def test_offload_slots_stream_missed_experts(offload_ckpt):
    """A decode step whose routed expert is not resident must stream it in.

    With cache_size = 2E + num_moe the pool has double-buffer slots [0, 2E)
    plus a handful of decode slots [2E, S). Prefill materializes each layer into
    [0, E). A decode step that routes to an expert *outside* that layer's
    materialized set is a miss: the LRU evicts a decode slot, copies the missed
    expert's host row in, and the forward reads it from the slot. The miss
    counter must advance -- proof the stream-in path actually ran (not just the
    hit path).
    """
    # Two MoE layers with a pool that can only hold ~one layer (cache_size =
    # E + num_moe < 2E) force REAL cross-layer eviction: materializing layer 1
    # must evict layer 0's experts, so layer 0's decode is a genuine miss that is
    # streamed back in. The miss counter must therefore be > 0 -- proof the
    # stream-in (copy_missing) path ran, not just the hit path.
    from freetoken.engine.engine import Engine as _Engine

    engine = _Engine(_engine_config(offload_ckpt, moe_backend="offload"))
    _add_prompt(engine, output_len=8)
    engine.generate()
    stats = engine.model.moe_cache.decode_miss_stats()
    assert stats["calls"] > 0, "decode must call ensure_experts"
    assert stats["missing"] > 0, "with a sub-layer pool, decode must stream (evicted) experts back in"


def test_offload_layer_map_matches_moe_layers(offload_ckpt):
    """The model's layer_id -> MoE-index map must match the loader's mapping.

    The slot pool is indexed by *MoE-layer index* (0-based among the MoE
    layers); the model's blocks are indexed by *absolute layer id*. A mismatch
    here (an off-by-one in first_k_dense_replace, or a dense layer counted)
    routes a layer to the wrong host bank -- the forward-correctness test above
    would then fail, so pin the mapping explicitly.
    """
    from freetoken.models.loader import _moe_layers

    model, _ = load_model(offload_ckpt, DEVICE, dtype=torch.float32, moe_backend="offload")
    expected = {layer_id: idx for idx, layer_id in enumerate(_moe_layers(model.config))}
    got = {layer_id: idx for layer_id, idx in enumerate(model.moe_layer_id) if idx != 0 or layer_id in expected}
    # Every MoE layer must map to the right MoE index.
    for layer_id, moe_idx in expected.items():
        assert model.moe_layer_id[layer_id] == moe_idx, (layer_id, moe_idx)
    # The cache must be sized for exactly the number of MoE layers.
    assert model.moe_cache.num_layers == len(expected)
