"""Forward-correctness tests for the hybrid MoE backend (issue #9, ADR 0002).

The contract: ``--moe-backend hybrid`` splits each decode step's routed-expert
misses between the two transports the ``offload`` and ``cpu`` backends each own --
a fraction ``f`` (the ``ft bench bw`` profile's fetch fraction) PCIe-fetched into
the XPU LRU slot pool and computed there, the rest ``(1 - f)`` computed on the host
CPU from the same pinned banks. The two halves are *disjoint* expert sets that,
together, cover exactly the routed experts, and each half uses the *same* SwiGLU
math and accumulation order as the pure backend it mirrors -- so the hybrid output
is **numerically identical to pure offload** (and to the in-VRAM reference). The
``q*`` split changes *which* experts ride which transport, never the arithmetic.

These tests are ``xpu``-marked (the B70 nightly, ``.venv-xpu``): they are
deselected in the torch-free CPU venv (see ``conftest.py``). They run a tiny
Qwen3-MoE through the *real* engine with the hybrid backend and compare the
greedy output to the in-VRAM reference. The fetch fraction is controlled with a
per-XPU ``ft bench bw`` profile keyed by the *real* device UUID (the same
identity convention the reader's auto-lookup uses), so the test pins an explicit
``f`` instead of depending on the box's measured bandwidths.

A per-uuid profile is only trusted when its ``xpu.name`` matches the box's card
(``_usable_profile``'s mismatch guard), so the fixture pins the real
``xpu_device_name()`` -- the same real-identity convention the profile-reader
tests use.
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
from freetoken.moe.bench_profile import default_profile_path

# XPU is the point of this module: skip cleanly (not fail) where there is none.
XPU = pytest.mark.skipif(not torch.xpu.is_available(), reason="no XPU available")

DEVICE = torch.device("cpu")

# A tiny Qwen3-MoE: 2 MoE layers (first_k_dense_replace=0 -> both layers are
# MoE), 4 experts, top-2, small hidden / vocab. 4 experts matter here: with a
# fetch fraction of 0.5 the hybrid splits the step's routed ids 2-to-2
# (``n_fetch = round(n * 0.5)``), so *both* the XPU-fetch and the host-CPU halves
# are exercised on every decode step -- the non-degenerate split the parity test
# must pin. (With fewer experts the 0.5 split could collapse to one empty half.)
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


# A per-expert checkpoint whose values *identify* (layer, expert, projection), so
# the reference and the hybrid path provably consume the same expert bytes.
# gate (expert e, row r): 100*(e+1) + r + layer*1000; up: +1; down: +2.
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
        w[f"{prefix}.mlp.gate.weight"] = torch.randn(experts, hidden)
    w["model.norm.weight"] = torch.randn(hidden)
    return w


@pytest.fixture(scope="module")
def hybrid_ckpt(tmp_path_factory):
    """A checkpoint with distinguishable per-expert weights + random dense weights.

    The in-VRAM reference and the hybrid-under-test both load *this same* path, so
    they consume byte-identical dense + expert bytes: any divergence is a hybrid
    split bug, not a weight difference.
    """
    from safetensors.torch import save_file

    path = tmp_path_factory.mktemp("qwen3moe-hybrid")
    (path / "config.json").write_text(json.dumps(TINY_CONFIG))
    # Re-seed so the dense weights are a fixed, order-independent value (the
    # reference and the hybrid both load this file, so they always agree).
    torch.manual_seed(1234)
    weights = {**_dense_weights(), **_distinguishable_experts()}
    shards = {k: v.contiguous() for k, v in weights.items()}
    save_file(shards, str(path / "model.safetensors"))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {k: "model.safetensors" for k in shards}})
    )
    return str(path)


@pytest.fixture(scope="module")
def hybrid_profile(tmp_path_factory):
    """A per-XPU ``ft bench bw`` profile that pins the fetch fraction to 0.5.

    Keyed by the *real* device UUID (the reader's auto-lookup looks it up by
    ``default_profile_path(uuid)``), and its ``xpu.name`` pinned to the real
    device name so ``_usable_profile``'s mismatch guard trusts it. A profile in
    ``$XDG_CACHE_HOME`` would be a shared, mutable file keyed by the box's
    UUID, so the test writes one to a temp dir instead (the same isolation the
    other fixtures use) and points the engine at it with ``FREETOKEN_BENCHBW_PATH``
    (the explicit-path seam the loader's auto-lookup reads before the per-uuid
    default). The overlap pair is written so the reader's ``pcie/(pcie+cpu)``
    formula yields exactly 0.5.
    """
    from freetoken.moe.bench_profile import _xpu_identity

    name, uuid = _xpu_identity()
    if name is None or uuid is None:
        pytest.skip("no XPU (or no readable uuid) on this box")
    # cpu 10, pcie 10 -> 10 / (10 + 10) = 0.5 (the overlap pair, preferred).
    prof = {
        "xpu": {"name": name, "uuid": uuid},
        "dtypes": {"bf16": "hybrid"},
        "dtype_kernels": {
            "bf16": {
                "cpu_moe_gbs": 10.0,
                "pcie_gather_gbs": 10.0,
                "cpu_moe_overlap_gbs": 10.0,
                "pcie_gather_overlap_gbs": 10.0,
            }
        },
    }
    path = tmp_path_factory.mktemp("benchbw") / f"{uuid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prof))
    return str(path)


@pytest.fixture(autouse=True)
def _clean_global_ctx():
    # Each Engine installs a global context; clear it between tests so two engines
    # in one process never collide. (conftest's autouse generation-hook reset also
    # runs; a duplicate reset is harmless.)
    yield
    reset_global_ctx()


def _engine_config(
    model_path: str, *, moe_backend: str | None, moe_hybrid_max_fetch: int = -1
) -> EngineConfig:
    # bf16 (the hero dtype): the hybrid profile is keyed by the model's *effective*
    # dtype (the loader maps the stamped dtype to the bench format, bfloat16 ->
    # "bf16"), so the reference and the hybrid must both run bf16 to share the
    # bf16-keyed profile. The tiny weights are small enough to stay exact in bf16.
    return EngineConfig(
        model_path=model_path,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        device=DEVICE,
        attention_backend="auto",
        moe_backend=moe_backend,
        moe_hybrid_max_fetch=moe_hybrid_max_fetch,
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


@XPU
def test_hybrid_forward_matches_in_vram_reference(hybrid_ckpt, hybrid_profile, monkeypatch):
    """The hybrid split must reproduce the in-VRAM forward's greedy tokens.

    Hybrid's two halves (PCIe-fetch to XPU / host-CPU) are disjoint expert sets
    that cover the routed experts, each mirroring the pure backend it uses; so the
    hybrid output must be *identical* to the in-VRAM reference's greedy tokens
    (prefill + decode). A split that double-counts or drops an expert -- or that
    computes a row through the wrong half -- changes the logits and fails.

    The fetch fraction is pinned to 0.5 by the per-uuid profile, so the split is
    deterministic and both halves are non-empty on every decode step.
    """
    # Reference: in-VRAM experts (moe_backend=None -> the default resident path).
    ref_engine = Engine(_engine_config(hybrid_ckpt, moe_backend=None))
    _add_prompt(ref_engine, output_len=8)
    ref_tokens = ref_engine.generate()

    # Under test: hybrid experts. Point the loader's auto-lookup at the profile
    # (FREETOKEN_BENCHBW_PATH) so the loader reads fetch_fraction 0.5 for this
    # expert format (the profile's xpu.name matches the box -> trusted).
    monkeypatch.setenv("FREETOKEN_BENCHBW_PATH", hybrid_profile)
    model, sources = load_model(hybrid_ckpt, DEVICE, dtype=torch.bfloat16, moe_backend="hybrid")
    assert model.moe_offload, "the hybrid path must flag the model as host-offloaded"
    assert model.moe_cache is not None, "the loader must attach the LRU slot pool"
    assert model.layers[0].mlp.experts is None, "the hybrid path must not build device-resident experts"
    # The per-step fetch fraction the block's forward reads (pinned to 0.5 above).
    assert getattr(model, "moe_hybrid_fetch_fraction", None) == pytest.approx(0.5, abs=1e-9), (
        "the loader must read the profile's fetch fraction (0.5) onto the model"
    )

    hybrid_engine = Engine(_engine_config(hybrid_ckpt, moe_backend="hybrid"))
    _add_prompt(hybrid_engine, output_len=8)
    hybrid_tokens = hybrid_engine.generate()

    assert hybrid_tokens == ref_tokens, (
        f"hybrid logits diverged from the in-VRAM reference: {hybrid_tokens} != {ref_tokens}"
    )
    # Both must actually generate (non-empty, in-vocab) tokens.
    assert len(hybrid_tokens[0]) == 8
    assert all(0 <= t < TINY_CONFIG["vocab_size"] for t in hybrid_tokens[0])
    # The decode steps must have actually driven the slot pool (not just prefill).
    stats = hybrid_engine.model.moe_cache.decode_miss_stats()
    assert stats["calls"] > 0, "decode must call ensure_experts on the slot pool"


@XPU
def test_hybrid_max_fetch_cap_stays_correct(hybrid_ckpt, hybrid_profile, monkeypatch):
    """--moe-hybrid-max-fetch must cap the XPU half without changing the math.

    The cap is the operator's override of the profile's ``q*`` fraction: a hard
    ceiling on the per-step PCIe-fetched expert count. With the profile pinning
    fetch_fraction 0.5 (the 4-expert model's decode step fetches 2 by default),
    capping at 1 shifts the *largest-index* routed expert from the XPU half to the
    CPU half. The two halves stay a disjoint cover, so the forward must still be
    numerically identical to the uncapped hybrid (and to the in-VRAM reference) --
    the cap changes the *transport split*, never the arithmetic.
    """
    monkeypatch.setenv("FREETOKEN_BENCHBW_PATH", hybrid_profile)

    # Uncapped hybrid (fetch fraction 0.5 -> the XPU half fetches 2 experts).
    uncapped_engine = Engine(_engine_config(hybrid_ckpt, moe_backend="hybrid"))
    _add_prompt(uncapped_engine, output_len=8)
    uncapped_tokens = uncapped_engine.generate()

    # Capped hybrid (cap 1 -> the XPU half fetches only 1, the rest rides the CPU).
    capped_engine = Engine(_engine_config(hybrid_ckpt, moe_backend="hybrid", moe_hybrid_max_fetch=1))
    _add_prompt(capped_engine, output_len=8)
    capped_tokens = capped_engine.generate()

    # The engine must have stashed the cap on the model (the block's forward reads
    # it through getattr). A missing/wrong stash would make the cap a no-op.
    assert getattr(capped_engine.model, "moe_hybrid_max_fetch", None) == 1, (
        "the engine must stash the --moe-hybrid-max-fetch cap on the model"
    )

    # The cap must not change the math: capped == uncapped (disjoint cover holds).
    assert capped_tokens == uncapped_tokens, (
        f"--moe-hybrid-max-fetch 1 changed the hybrid output: {capped_tokens} != {uncapped_tokens}"
    )
    assert len(capped_tokens[0]) == 8
    assert all(0 <= t < TINY_CONFIG["vocab_size"] for t in capped_tokens[0])


@XPU
def test_hybrid_is_deterministic(hybrid_ckpt, hybrid_profile, monkeypatch):
    """Two hybrid runs of the same checkpoint must be identical.

    Guards against non-determinism in the hybrid split (e.g. a dict-order-dependent
    expert partition, or a stale slot from a prior step leaking into the CPU half)
    -- the same invariant the offload / cpu determinism tests pin.
    """
    from freetoken.engine.engine import Engine as _Engine

    monkeypatch.setenv("FREETOKEN_BENCHBW_PATH", hybrid_profile)

    def run():
        engine = _Engine(_engine_config(hybrid_ckpt, moe_backend="hybrid"))
        _add_prompt(engine, output_len=5)
        return engine.generate()

    a = run()
    b = run()
    assert a == b
    assert len(a[0]) == 5
