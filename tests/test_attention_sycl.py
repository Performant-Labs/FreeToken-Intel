"""Tests for the native SYCL attention backend (issue ``attn-sycl``, #5).

Two halves:

* CPU-safe (the per-PR ``ci`` job, torch-free): the backend imports and registers
  cleanly, and ``sycl`` is wired as a spec-consuming backend (the kernel needs
  ``attn_spec.sliding_window`` for SWA). These touch no torch and run on any box.

* ``xpu``-marked (the B70 nightly, ``.venv-xpu``): the backend actually compiles
  ``attention.cpp`` and drives the kernel. The correctness bar is that a dummy-
  weight engine running attention on the SYCL kernel produces *exactly* the same
  greedy tokens as the same engine running the reference (pure-torch) backend.
  Both run the identical model/weights/prompts, so the only difference is the
  attention math -- a mismatch means the kernel is wrong.

  The reference engine runs on the **CPU** (not the XPU): this is the
  stream-desync-safe pattern. The SYCL kernel enqueues on its own SYCL queue,
  which desyncs torch's XPU stream, so a torch reference computed on the XPU
  *after* the kernel read stale USM and would falsely "diverge". A CPU reference
  is on a different, unaffected device, so the comparison is sound.
"""
from __future__ import annotations

import json

import pytest

# --- Import + registration (torch is a hard dependency of the sycl backend) --
#
# sycl.py imports torch at module scope, so these "CPU-safe" tests only run where
# torch is installed (the .venv-xpu / B70 nightly); in the torch-free per-PR
# venv importorskip skips them (same belt-and-suspenders as test_xpu_smoke.py).

def test_sycl_backend_module_imports():
    pytest.importorskip("torch")
    import freetoken.attention as attention

    for name in ("SyclAttentionBackend", "SyclMetadata"):
        assert hasattr(attention, name), name
    # The backend class must satisfy the BaseAttnBackend contract.
    from freetoken.attention.base import BaseAttnBackend

    assert issubclass(attention.SyclAttentionBackend, BaseAttnBackend)


def test_sycl_backend_registered_with_spec_consumption():
    pytest.importorskip("torch")
    from freetoken.attention import attention_backend_info
    from freetoken.attention.base import AttnType

    info = attention_backend_info("sycl")
    assert info.requires_sycl
    assert info.consumes_attn_spec  # the kernel needs attn_spec.sliding_window for SWA
    assert AttnType.SWA in info.supported_types
    assert AttnType.FULL in info.supported_types


def test_sycl_backend_raises_without_xpu_on_first_use(monkeypatch):
    pytest.importorskip("torch")
    # The constructor is safe (lazy); first use must raise a clear error rather
    # than dlopen a missing toolchain. _xpu_available is forced False so this
    # holds even on a box that *does* have an XPU.
    import freetoken.attention.sycl as sycl_mod

    monkeypatch.setattr(sycl_mod, "_xpu_available", lambda: False)
    backend = sycl_mod.SyclAttentionBackend(object())
    with pytest.raises(RuntimeError):
        backend._ensure_loaded()


# --- XPU: the kernel drives real attention ------------------------------------

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


def _write_tiny_checkpoint(tmp_path, *, random: bool = False) -> str:
    """A minimal Qwen3-MoE checkpoint on disk.

    ``random=False`` zeroes the dense weights (fast; the zero-weight engine is
    enough to exercise the engine loop + a valid attention *shape*). ``random``
    seeds the weights from a fixed RNG so a K/V-slot off-by-one in the attention
    kernel -- invisible with all-zero weights (every slot reads 0.0 either way)
    -- actually changes the logits and is caught by the token-equality assertion.
    """
    import torch
    from freetoken.models.qwen3_moe import parse_config
    from safetensors.torch import save_file

    model_path = tmp_path / "ckpt"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(TINY_CONFIG))

    config = parse_config(type("Hf", (), {"to_dict": lambda self: TINY_CONFIG})())
    state = {}
    gen = torch.Generator().manual_seed(0)

    def add(name, shape):
        if random:
            state[name] = torch.randn(shape, dtype=torch.float32, generator=gen) * 0.5
        else:
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
        # nn.Linear.weight is [out_features, in_features]. q_proj: [heads*head_dim,
        # hs]; k/v_proj: [kv*head_dim, hs]; o_proj: [hs, heads*head_dim]. q_proj's
        # two dims are equal here (hs == heads*head_dim) so a transposed q_proj
        # would silently copy, but k/v_proj ([32, 64] vs a transposed [64, 32])
        # would not -- which is what makes a wrong shape fail loudly.
        add(f"{prefix}.self_attn.q_proj.weight", (heads * head_dim, hs))
        add(f"{prefix}.self_attn.k_proj.weight", (kv * head_dim, hs))
        add(f"{prefix}.self_attn.v_proj.weight", (kv * head_dim, hs))
        add(f"{prefix}.self_attn.o_proj.weight", (hs, heads * head_dim))
        add(f"{prefix}.self_attn.q_norm.weight", (head_dim,))
        add(f"{prefix}.self_attn.k_norm.weight", (head_dim,))
        add(f"{prefix}.post_attention_layernorm.weight", (hs,))
        # The MoE router: [num_experts, hidden_size] (one logit per expert per token).
        add(f"{prefix}.mlp.gate.weight", (experts, hs))
        for e in range(experts):
            # Per-expert source keys are the *raw HF* form: the trailing token is
            # the projection name with NO trailing ".weight" (unlike the dense
            # params, which carry ".weight"). The loader's _expert_source_info
            # recognizes exactly "…experts.{e}.{gate|up|down}_proj".
            add(f"{prefix}.mlp.experts.{e}.gate_proj", (inter, hs))
            add(f"{prefix}.mlp.experts.{e}.up_proj", (inter, hs))
            add(f"{prefix}.mlp.experts.{e}.down_proj", (hs, inter))
    add("model.norm.weight", (hs,))
    add("lm_head.weight", (vocab, hs))

    save_file(state, str(model_path / "model.safetensors"))
    # A populated weight_map is required when an index file is present: the
    # loader's iter_safetensors resolves every tensor through the index's
    # weight_map (and only falls back to a header scan when no index file exists).
    # An *empty* weight_map would make the loader yield zero tensors -> "Missing
    # MoE expert source layers". Map each tensor to the single shard it lives in.
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {name: "model.safetensors" for name in state}})
    )
    return str(model_path)


def _engine_config(model_path, *, device, attention_backend, dummy_weight: bool = True):
    import torch
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    return EngineConfig(
        model_path=model_path,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.float32,
        device=device,
        attention_backend=attention_backend,
        max_running_req=2,
        page_size=1,
        max_seq_len_override=32,
        use_dummy_weight=dummy_weight,
    )


def _add_prompt(engine, output_len, prompt_ids):
    from freetoken.core import Req, SamplingParams

    engine.add_request(
        Req(
            input_ids=list(prompt_ids),
            table_idx=0,
            cached_len=0,
            output_len=output_len,
            uid=0,
            sampling_params=SamplingParams(temperature=0.0, max_tokens=output_len),
            cache_handle=None,
        )
    )


def _mk_req(prompt_ids, output_len, uid):
    from freetoken.core import Req, SamplingParams

    return Req(
        input_ids=list(prompt_ids),
        table_idx=0,
        cached_len=0,
        output_len=output_len,
        uid=uid,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=output_len),
        cache_handle=None,
    )


@pytest.fixture(autouse=True)
def _clean_global_ctx():
    from freetoken.core import reset_global_ctx

    yield
    reset_global_ctx()


@pytest.mark.xpu
def test_sycl_backend_matches_reference_on_xpu(tmp_path):
    import torch

    assert torch.xpu.is_available(), "this test runs on the B70 XPU nightly"
    model_path = _write_tiny_checkpoint(tmp_path)

    from freetoken.core import reset_global_ctx
    from freetoken.engine.engine import Engine

    # Reference (the correctness bar) on the CPU -- see the module docstring for
    # why the reference must not share the XPU stream with the kernel.
    reset_global_ctx()
    ref_engine = Engine(_engine_config(model_path, device="cpu", attention_backend="auto"))
    _add_prompt(ref_engine, output_len=4, prompt_ids=[1, 2, 3])
    ref_tokens = ref_engine.generate()
    reset_global_ctx()

    # The SYCL backend on the XPU must produce the identical tokens.
    sycl_engine = Engine(_engine_config(model_path, device="xpu", attention_backend="sycl"))
    _add_prompt(sycl_engine, output_len=4, prompt_ids=[1, 2, 3])
    sycl_tokens = sycl_engine.generate()
    reset_global_ctx()

    assert len(sycl_tokens) == 1
    assert sycl_tokens == ref_tokens, (
        f"SYCL attention diverged from the reference backend: {sycl_tokens} != {ref_tokens}"
    )


@pytest.mark.xpu
def test_sycl_backend_multi_request_decode_on_xpu(tmp_path):
    """Two requests in one decode step (bs>1) -- exercises the kernel's batch axis."""
    import torch

    assert torch.xpu.is_available()
    model_path = _write_tiny_checkpoint(tmp_path)

    from freetoken.core import reset_global_ctx
    from freetoken.engine.engine import Engine

    reset_global_ctx()
    ref_engine = Engine(_engine_config(model_path, device="cpu", attention_backend="auto"))
    _add_prompt(ref_engine, output_len=3, prompt_ids=[1, 2, 3])
    ref_engine.add_request(_mk_req(prompt_ids=[5, 6], output_len=3, uid=1))
    ref_tokens = ref_engine.generate()
    reset_global_ctx()

    sycl_engine = Engine(_engine_config(model_path, device="xpu", attention_backend="sycl"))
    _add_prompt(sycl_engine, output_len=3, prompt_ids=[1, 2, 3])
    sycl_engine.add_request(_mk_req(prompt_ids=[5, 6], output_len=3, uid=1))
    sycl_tokens = sycl_engine.generate()
    reset_global_ctx()

    assert sycl_tokens == ref_tokens


@pytest.mark.xpu
def test_sycl_backend_matches_reference_random_weights_on_xpu(tmp_path):
    """Non-zero weights: a K/V-slot off-by-one changes the logits and the tokens.

    The two tests above use zeroed weights, under which *every* KV slot reads
    0.0 -- so a bug that reads the wrong slot (e.g. using the query position as a
    slot index, or dropping the newest key) is invisible and the attention output
    is the same either way. With random weights the logits are non-trivial, so
    any wrong slot read diverges the greedy tokens and the equality assertion
    fires. Multi-step decode (output_len=5) is required: the slot layout only
    differs from a correct one once a request attends over a multi-token history
    (kv_len > 1).

    The reference runs on the XPU (not the CPU) on purpose: with random weights
    the model's *non-attention* ops (RMSNorm / MoE / RoPE) round differently on
    CPU than on the XPU, so a CPU-vs-XPU token comparison can flip a single
    greedy token on a near-tie even when the attention kernel is exactly correct.
    Comparing both engines on the XPU isolates the attention backend (the thing
    this PR changes) from that device-level float nondeterminism. The kernel's
    USM work is fully synchronized by the engine before generate() returns, so
    the token output is stable across runs.
    """
    import torch

    assert torch.xpu.is_available()
    model_path = _write_tiny_checkpoint(tmp_path, random=True)

    from freetoken.core import reset_global_ctx
    from freetoken.engine.engine import Engine

    # Reference (correctness bar) on the XPU -- see the docstring for why a CPU
    # reference is the wrong bar for non-zero weights.
    reset_global_ctx()
    ref_engine = Engine(_engine_config(model_path, device="xpu", attention_backend="auto", dummy_weight=False))
    _add_prompt(ref_engine, output_len=5, prompt_ids=[1, 2, 3])
    ref_tokens = ref_engine.generate()
    reset_global_ctx()

    sycl_engine = Engine(_engine_config(model_path, device="xpu", attention_backend="sycl", dummy_weight=False))
    _add_prompt(sycl_engine, output_len=5, prompt_ids=[1, 2, 3])
    sycl_tokens = sycl_engine.generate()
    reset_global_ctx()

    assert len(sycl_tokens) == 1
    assert sycl_tokens == ref_tokens, (
        f"SYCL attention diverged from the reference under random weights: {sycl_tokens} != {ref_tokens}"
    )
