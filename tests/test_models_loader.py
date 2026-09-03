"""Tests for the checkpoint loader (issue ``models-loader``, #17).

Split by environment:

* The config/tokenizer helpers in ``freetoken.utils.hf`` are torch-free and run
  on a CPU-only box. They are exercised against a locally fabricated
  ``config.json`` (offline) so the test needs no network.
* The weight path (safetensors reader, dense->device / experts->host routing,
  and the MoE expert bank builder) needs ``torch``. Those tests are marked
  ``torch`` and are skipped on a box without it.
"""
from __future__ import annotations

import json
import os

import pytest

torch = pytest.importorskip("torch")

# Populated by the ``moe_ckpt`` fixture with the exact expert tensor written to
# the checkpoint, so the "real banks" test can assert the loader round-trips it.
_EXPECTED_L0_GATE_UP = None
# Populated by the ``moe_ckpt_perexpert`` fixture with the per-expert source tensors
# (gate / up / down for layer 0) so the repack test can assert the stacked bank
# equals the concatenation of the exact bytes that were written.
_EXPECTED_L0_PEREXPERT = None
# Populated by the ``moe_ckpt_distinguishable`` fixture: the exact per-expert
# gate/up/down bytes for layer 0, written with *distinguishable* values so a test
# can tell whether the loader routed each byte to the right expert + projection
# (a shape-only check would miss a gate<->up swap or an expert row shift).
_EXPECTED_L0_DISTINGUISHABLE = None

from freetoken.models.loader import load_model
from freetoken.models.weight import _PlainBank, load_moe_expert_sources
from freetoken.utils import cached_load_hf_config
from freetoken.utils.hf import RawConfigShim, load_tokenizer

QWEN3_MOE_CONFIG = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "model_type": "qwen3_moe",
    "hidden_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "intermediate_size": 256,
    "moe_intermediate_size": 32,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "vocab_size": 64,
}


def _write_checkpoint(tmp_path, *, config: dict, weights: dict) -> str:
    """Fabricate a local HF-style checkpoint: a config.json plus one
    ``model.safetensors`` shard (with an index) holding ``weights``."""
    from safetensors.torch import save_file

    cfg = dict(config)
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    shards = {k: v.contiguous() for k, v in weights.items()}
    if shards:
        save_file(shards, str(tmp_path / "model.safetensors"))
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {k: "model.safetensors" for k in shards}})
        )
    return str(tmp_path)


def _qwen3_moe_weights() -> dict:
    """A tiny Qwen3-MoE checkpoint: dense weights + per-layer stacked MoE
    experts (the ``...experts.gate_up_proj`` / ``...experts.down_proj`` layout
    the model adapter normalizes to)."""
    hidden, inter, experts, vocab = 128, 32, 4, 64
    layers = 2
    w = {
        "model.embed_tokens.weight": torch.randn(vocab, hidden),
        "lm_head.weight": torch.randn(vocab, hidden),
        # one dense attention weight (not an expert -> dense path)
        "model.layers.0.self_attn.q_proj.weight": torch.randn(hidden, hidden),
    }
    for layer in range(layers):
        w[f"model.layers.{layer}.mlp.experts.gate_up_proj"] = torch.randn(experts, 2 * inter, hidden)
        w[f"model.layers.{layer}.mlp.experts.down_proj"] = torch.randn(experts, hidden, inter)
    return w


def _qwen3_moe_weights_per_expert() -> dict:
    """A tiny Qwen3-MoE checkpoint in the *raw HF* per-expert layout: one
    ``...experts.{e}.{gate,up,down}_proj`` tensor per expert (NOT the packed
    ``gate_up_proj``/``down_proj`` form). This is what a real HF checkpoint such
    as Qwen3-30B-A3B ships, so the loader must repack it to the packed banks."""
    hidden, inter, experts, vocab = 128, 32, 4, 64
    layers = 2
    w = {
        "model.embed_tokens.weight": torch.randn(vocab, hidden),
        "lm_head.weight": torch.randn(vocab, hidden),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(hidden, hidden),
    }
    for layer in range(layers):
        for e in range(experts):
            prefix = f"model.layers.{layer}.mlp.experts.{e}"
            w[f"{prefix}.gate_proj"] = torch.randn(inter, hidden)
            w[f"{prefix}.up_proj"] = torch.randn(inter, hidden)
            w[f"{prefix}.down_proj"] = torch.randn(hidden, inter)
    return w


def _qwen3_moe_weights_distinguishable() -> dict:
    """A per-expert checkpoint whose values *identify* both the expert and the
    projection, so a placement test can prove each byte landed in exactly the
    right place (catches a gate<->up swap or an off-by-one expert row that a
    shape check would miss).

    Values (exact integers, lossless in bf16, all distinct per (expert,
    projection) and distinct across the two halves of a fused gate_up row):

    * gate_proj (expert e, row r, col c): 100*(e+1) + r   -> in [100, 431]
    * up_proj   (expert e, row r, col c): 100*(e+1) + r + 1  -> in [101, 432]
    * down_proj (expert e, row r, col c): 100*(e+1) + r + 2  -> in [102, 433]

    (Columns are constant within a row; the row index alone distinguishes the
    two I=32-row halves of the fused ``[E, 2I, H]`` gate_up bank.)
    """
    hidden, inter, experts, vocab = 128, 32, 4, 64
    layers = 2
    w = {
        "model.embed_tokens.weight": torch.randn(vocab, hidden),
        "lm_head.weight": torch.randn(vocab, hidden),
    }
    for layer in range(layers):
        for e in range(experts):
            prefix = f"model.layers.{layer}.mlp.experts.{e}"
            base = 100 * (e + 1)
            w[f"{prefix}.gate_proj"] = torch.arange(inter, dtype=torch.float32)[:, None].repeat(1, hidden) + base
            w[f"{prefix}.up_proj"] = torch.arange(inter, dtype=torch.float32)[:, None].repeat(1, hidden) + base + 1
            w[f"{prefix}.down_proj"] = torch.arange(hidden, dtype=torch.float32)[:, None].repeat(1, inter) + base + 2
    return w


@pytest.fixture(scope="module")
def moe_ckpt(tmp_path_factory):
    weights = _qwen3_moe_weights()
    path = _write_checkpoint(tmp_path_factory.mktemp("qwen3moe"), config=QWEN3_MOE_CONFIG, weights=weights)
    # Persist the source expert tensors so the "real banks" test can compare the
    # loaded bank against the exact bytes that were written (randn is not
    # reproducible across calls).
    global _EXPECTED_L0_GATE_UP
    _EXPECTED_L0_GATE_UP = weights["model.layers.0.mlp.experts.gate_up_proj"].to(torch.bfloat16)
    return path


@pytest.fixture(scope="module")
def moe_ckpt_perexpert(tmp_path_factory):
    weights = _qwen3_moe_weights_per_expert()
    path = _write_checkpoint(
        tmp_path_factory.mktemp("qwen3moe-pe"), config=QWEN3_MOE_CONFIG, weights=weights
    )
    global _EXPECTED_L0_PEREXPERT
    _EXPECTED_L0_PEREXPERT = {
        name: weights[name].to(torch.bfloat16)
        for name in (
            "model.layers.0.mlp.experts.0.gate_proj",
            "model.layers.0.mlp.experts.0.up_proj",
            "model.layers.0.mlp.experts.0.down_proj",
        )
    }
    return path


@pytest.fixture(scope="module")
def moe_ckpt_distinguishable(tmp_path_factory):
    weights = _qwen3_moe_weights_distinguishable()
    path = _write_checkpoint(
        tmp_path_factory.mktemp("qwen3moe-dist"), config=QWEN3_MOE_CONFIG, weights=weights
    )
    global _EXPECTED_L0_DISTINGUISHABLE
    _EXPECTED_L0_DISTINGUISHABLE = {
        name: weights[name].to(torch.bfloat16)
        for name in (
            "model.layers.0.mlp.experts.0.gate_proj",
            "model.layers.0.mlp.experts.0.up_proj",
            "model.layers.0.mlp.experts.0.down_proj",
        )
    }
    return path


@pytest.fixture(scope="module")
def dense_ckpt(tmp_path_factory):
    return _write_checkpoint(
        tmp_path_factory.mktemp("qwen3dense"),
        config={**QWEN3_MOE_CONFIG, "architectures": ["Qwen3ForCausalLM"]},
        weights={"model.embed_tokens.weight": torch.randn(64, 128)},
    )


# --- torch-free config helpers (run on CPU-only boxes) -----------------------


def test_raw_config_shim_attribute_access():
    shim = RawConfigShim({"hidden_size": 128, "num_hidden_layers": 2}, architectures=["X"])
    assert shim.hidden_size == 128
    assert shim.num_hidden_layers == 2
    assert "X" in shim.architectures
    with pytest.raises(AttributeError):
        shim.does_not_exist  # noqa: B018


def test_raw_config_shim_wraps_nested_config():
    shim = RawConfigShim({"moe_config": {"num_experts": 4}})
    assert isinstance(shim.moe_config, RawConfigShim)
    assert shim.moe_config.num_experts == 4


def test_download_hf_weight_returns_local_dir(moe_ckpt):
    from freetoken.utils.hf import download_hf_weight

    assert download_hf_weight(moe_ckpt) == moe_ckpt
    assert os.path.isfile(os.path.join(moe_ckpt, "config.json"))


# --- torch-gated checkpoint loading -------------------------------------------


def test_cached_load_hf_config_reads_local(moe_ckpt):
    cfg = cached_load_hf_config(moe_ckpt)
    assert cfg.architectures == ["Qwen3MoeForCausalLM"]
    assert cfg.hidden_size == 128
    assert cfg.num_experts == 4


def test_iter_safetensors_reads_shards_onto_device(dense_ckpt):
    # The loader's shard-reading primitive (safetensors -> named tensors on the
    # destination device) is exercised directly: the Qwen3 dense model's
    # ``iter_weights``/``parse_config`` are owned by a separate stub
    # (``models-dense``), so the dense->device routing is asserted at the
    # primitive the loader is built on.
    from freetoken.models.weight import iter_safetensors

    device = torch.device("cpu")
    got = {name: tensor for name, tensor in iter_safetensors(dense_ckpt, device)}
    assert set(got) == {"model.embed_tokens.weight"}
    assert got["model.embed_tokens.weight"].shape == (64, 128)
    assert got["model.embed_tokens.weight"].device == device


def test_iter_safetensors_opens_each_shard_exactly_once(tmp_path):
    """Real bug found by issue #138's real-checkpoint validation: an earlier
    version opened (mmap'd) the shard file once PER TENSOR instead of once
    per shard. For a checkpoint with many tensors in one shard (a GPTQ MoE
    checkpoint's per-expert layout puts thousands in one file), that
    exhausted a 27GB virtual-address ulimit well before physical RAM was
    ever the constraint -- confirmed against the real
    Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 checkpoint (10,240 tensors in one ~1.4GB
    shard). Spies on safe_open's call count directly, not just correctness
    (the old and new code paths return identical tensors either way)."""
    from safetensors.torch import save_file

    import freetoken.models.weight as weight_mod

    path = tmp_path / "many_tensors_ckpt"
    path.mkdir()
    # 20 small tensors in ONE shard -- if opened per-tensor, safe_open would
    # be called 20 times for this one file; opened per-shard, exactly once.
    tensors = {f"t{i}": torch.randn(2, 2) for i in range(20)}
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path / "model.safetensors"))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}})
    )

    open_count = 0
    real_safe_open = weight_mod.safe_open

    class _CountingSafeOpen:
        def __init__(self, *args, **kwargs):
            nonlocal open_count
            open_count += 1
            self._inner = real_safe_open(*args, **kwargs)

        def __enter__(self):
            return self._inner.__enter__()

        def __exit__(self, *exc):
            return self._inner.__exit__(*exc)

    weight_mod.safe_open = _CountingSafeOpen
    try:
        got = dict(weight_mod.iter_safetensors(str(path), torch.device("cpu")))
    finally:
        weight_mod.safe_open = real_safe_open

    assert open_count == 1, f"expected exactly 1 safe_open call for 1 shard, got {open_count}"
    assert set(got) == set(tensors)
    for name, expected in tensors.items():
        torch.testing.assert_close(got[name], expected)


def test_load_moe_expert_sources_dummy_banks(moe_ckpt):
    gate_up, down = load_moe_expert_sources(moe_ckpt, dtype=torch.bfloat16, dummy=True)
    # Per-layer banks, one per MoE layer, stacked to [num_experts, ...].
    assert len(gate_up) == 2 and len(down) == 2
    for gu, dn in zip(gate_up, down):
        assert gu.shape == (4, 2 * 32, 128)
        assert gu.dtype == torch.bfloat16
        assert dn.shape == (4, 128, 32)
        assert dn.device.type == "cpu"  # host offload banks


def test_load_moe_expert_sources_real_banks(moe_ckpt):
    gate_up, down = load_moe_expert_sources(moe_ckpt, dtype=torch.bfloat16)
    assert len(gate_up) == 2 and len(down) == 2
    # The real path reads the checkpoint, so bank[0] must match the exact bytes
    # that were written (the fixture persisted them as bf16).
    assert torch.equal(gate_up[0], _EXPECTED_L0_GATE_UP)
    assert gate_up[0].device.type == "cpu"


def test_load_model_real_moe_routes_experts_to_host(moe_ckpt):
    model, expert_sources = load_model(moe_ckpt, torch.device("cpu"))
    # MoE checkpoint -> per-layer expert banks, on host memory.
    assert model is not None
    assert len(expert_sources[0]) == 2
    assert expert_sources[0][0].device.type == "cpu"


# --- per-expert (raw HF) checkpoint -> repacked packed banks (#53) -----------


def test_load_moe_expert_sources_real_banks_per_expert_layout(moe_ckpt_perexpert):
    """A raw HF per-expert checkpoint is repacked to the same packed banks the
    ``dummy=True`` path fabricates: gate_up [E, 2I, H] (gate then up) + down
    [E, H, I]."""
    gate_up, down = load_moe_expert_sources(moe_ckpt_perexpert, dtype=torch.bfloat16)
    assert len(gate_up) == 2 and len(down) == 2
    for gu, dn in zip(gate_up, down):
        assert gu.shape == (4, 2 * 32, 128)
        assert dn.shape == (4, 128, 32)
        assert gu.device.type == "cpu" and dn.device.type == "cpu"
    # Round-trip: layer 0's bank row 0 must equal the exact gate/up bytes that
    # were written, fused on the inner (dim 1) axis: the first [I, H] block is the
    # gate row, the second the up row. The expected bank is built by the same
    # reliable 3-D route the loader uses (promote the rows to 3-D, then cat) --
    # a bare 2-D ``torch.cat(..., dim=1)`` is mishandled by the torch XPU build
    # (it flattens), so it must not be used to construct the reference.
    exp = _EXPECTED_L0_PEREXPERT
    gate0 = exp["model.layers.0.mlp.experts.0.gate_proj"]
    up0 = exp["model.layers.0.mlp.experts.0.up_proj"]
    down0 = exp["model.layers.0.mlp.experts.0.down_proj"]
    from freetoken.models.weight import _stack_expert_rows

    expected_gate_up = torch.cat([_stack_expert_rows([gate0]), _stack_expert_rows([up0])], dim=1)[0]
    assert torch.equal(gate_up[0][0], expected_gate_up)
    assert torch.equal(down[0][0], down0)


def test_load_model_real_moe_per_expert_routes_to_host(moe_ckpt_perexpert):
    """The top-level loader path (load_model) handles the per-expert layout and
    routes the repacked banks to host memory, exactly like the packed path."""
    model, expert_sources = load_model(moe_ckpt_perexpert, torch.device("cpu"))
    assert model is not None
    assert len(expert_sources[0]) == 2
    assert expert_sources[0][0].shape == (4, 2 * 32, 128)
    assert expert_sources[0][0].device.type == "cpu"


def test_load_moe_expert_sources_rejects_unknown_expert_key(tmp_path):
    """An expert-looking key that is neither packed nor a known per-expert
    projection still raises (the loader must not silently mis-route weights)."""
    cfg = dict(QWEN3_MOE_CONFIG)
    weights = {
        "model.embed_tokens.weight": torch.randn(64, 128),
        "model.layers.0.mlp.experts.0.bogus_proj": torch.randn(32, 128),
    }
    path = _write_checkpoint(tmp_path, config=cfg, weights=weights)
    with pytest.raises(ValueError, match="Unexpected expert weight key"):
        load_moe_expert_sources(path, dtype=torch.bfloat16)


def test_place_expert_weights_routes_each_byte_correctly(moe_ckpt_distinguishable):
    """``load_model`` must place the repacked packed banks into the model's
    per-expert modules with the *exact* byte-to-place routing: gate row ->
    ``gate_proj``, the following up row -> ``up_proj``, down -> ``down_proj``,
    one row per expert.

    The checkpoint's values distinguish (expert, projection), so a routing
    mistake -- e.g. the old code splitting a fused ``[E, 2I, H]`` row as
    ``gu[e,0]``/``gu[e,1]`` (a 4-D ``[E,2,I,H]`` layout), which would swap or
    truncate the gate/up halves -- makes an assertion below fail, whereas the
    shape-only tests pass regardless. This is the regression guard for the
    packed-bank contract (#53 / ADR 0002).
    """
    hidden, inter, experts = 128, 32, 4
    model, _ = load_model(moe_ckpt_distinguishable, torch.device("cpu"))
    for layer in range(2):
        exp = _EXPECTED_L0_DISTINGUISHABLE  # layer-0 reference (values are layer-independent here)
        for e in range(experts):
            m = model.layers[layer].mlp.experts[e]
            base = 100 * (e + 1)
            exp_gate = torch.arange(inter, dtype=torch.float32)[:, None].repeat(1, hidden) + base
            exp_up = torch.arange(inter, dtype=torch.float32)[:, None].repeat(1, hidden) + base + 1
            exp_down = torch.arange(hidden, dtype=torch.float32)[:, None].repeat(1, inter) + base + 2
            # The model params are bf16; the expected ints are exactly representable.
            assert torch.equal(m.gate_proj.weight, exp_gate.to(torch.bfloat16)), (layer, e, "gate")
            assert torch.equal(m.up_proj.weight, exp_up.to(torch.bfloat16)), (layer, e, "up")
            assert torch.equal(m.down_proj.weight, exp_down.to(torch.bfloat16)), (layer, e, "down")
