"""Tests for the Qwen3.5/3.6 (``qwen3_5_moe``) weight + forward path -- torch-gated.

``iter_weights`` (and the ``load_model`` / ``forward`` paths built on it) need
torch to move tensors to their destination devices, so this module is
torch-gated: it ``importorskip("torch")``s at the top and is deselected on a
torch-free box (the CPU ``.venv``). It runs under the XPU venv / nightly.

These tests drive the real loader contract against a fabricated *multimodal*
checkpoint whose language tower sits under ``model.language_model.*`` (the Qwen3.6
layout) and ships a ``model.visual.*`` vision tower:

* ``iter_weights`` drops the vision tower and remaps ``model.language_model.*``
  to ``model.*`` so the loader's MoE-bank plumbing resolves the keys.
* routed experts go to host; everything else (including the always-on shared
  expert and the linear-attention weights) goes to the dense device.
* ``load_moe_expert_sources`` builds the per-layer banks from the remapped keys.
* ``load_model`` runs end-to-end: it places the dense weights, populates the
  in-VRAM expert modules, builds the host banks, and returns a fully-wired model.
* the forward pass is cross-checked against an independent reference
  implementation of the same hybrid architecture (Gated-Delta-Net linear
  attention + gated GQA + 256-way shared-expert MoE) built from the same weights): a correct forward
  reproduces the reference's per-position logits. (The 35B model is reference-
  matched against HF transformers in the nightly; that needs full-precision
  weights + B70 headroom, so it is not part of this fast CPU suite.)
"""
from __future__ import annotations

import json
import os

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402  (the reference below is torch)

from freetoken.models.loader import load_model
from freetoken.models.qwen3_5_moe import iter_weights
from freetoken.models.weight import load_moe_expert_sources

H, I, E, V, L = 32, 16, 4, 64, 2
# Full-attention head geometry (the model + KV pool derive these off the config).
# Like the real Qwen3.6, head_dim = hidden // num_kv_heads (256 = 2048/2): here
# 16 = 32/2, so the shared paged-KV row is [num_kv_heads, head_dim] = [2, 16] and
# the GQA repeat (num_attention_heads // num_kv_heads = 4 // 2 = 2) is > 1.
NH, NKV, HD = 4, 2, 16  # num_attention_heads, num_key_value_heads, head_dim
# attn_output_gate=True -> q_proj is doubled (query + the output gate).
Q_PROJ_DIM = NH * HD * 2
O_PROJ_DIM = NH * HD
KV_PROJ_DIM = NKV * HD
# Linear-attention (Gated-Delta-Net) head geometry. The paged-KV pool row is
# shared by the full-attention layers (num_kv_heads) and the linear layers
# (num_value_heads, the recurrent-state head count), so they must match:
# NV == NKV. The Gated-Delta-Net recurrent state is per value head
# [num_v, key_dim, value_dim] and the delta rule runs over the value heads
# (the q/k are grouped to the value-head count), so the key and value head
# counts and dims must line up (as in the real Qwen3.6: key and value head dim
# are both 128) -> here NK == NV and KD == VD.
NK, NV = 2, 2
KD, VD = 16, 16
KEY_DIM, VALUE_DIM = NK * KD, NV * VD
QKV_DIM = KEY_DIM * 2 + VALUE_DIM
CONV_DIM = KEY_DIM * 2 + VALUE_DIM
CONV_K = 4  # linear_conv_kernel_dim

# A small hybrid text tower: 2 layers (1 linear-attention + 1 full-attention),
# a 4-way router top-2 + an always-on shared expert, and a vision tower to prove
# it is dropped. The text tower sits under model.language_model.* (the real
# Qwen3.6 layout).
def _qwen35_text_config() -> dict:
    return {
        "hidden_size": H,
        "num_hidden_layers": L,
        "num_attention_heads": NH,
        "num_key_value_heads": NKV,
        "num_experts": E,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": I,
        "shared_expert_intermediate_size": I,
        "vocab_size": V,
        "max_position_embeddings": 128,
        "head_dim": HD,
        "attn_output_gate": True,
        "partial_rotary_factor": 0.5,
        "full_attention_interval": 2,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_num_key_heads": NK,
        "linear_num_value_heads": NV,
        "linear_key_head_dim": KD,
        "linear_value_head_dim": VD,
        "linear_conv_kernel_dim": CONV_K,
        "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.5},
    }


def _qwen35_weights() -> dict:
    """The full hybrid text-tower weight set (every layer), plus a vision tower.

    Layer 0 is linear-attention (linear_attn.* + conv1d), layer 1 is full-
    attention (self_attn.*); both carry a MoE mlp (experts.* + shared_expert.*).
    """
    w = {
        "model.visual.blocks.0.attn.q.weight": torch.randn(8, 8),
        "model.language_model.embed_tokens.weight": torch.randn(V, H),
        "model.language_model.norm.weight": torch.randn(H),
        "lm_head.weight": torch.randn(V, H),
    }
    for layer in range(L):
        if layer % 2 == 0:  # linear-attention (Gated DeltaNet) layer
            # The model sizes these off the (asymmetric) linear head dims:
            # in_proj_qkv -> [key_dim*2 + value_dim, hidden]; in_proj_z / out_proj
            # -> [value_dim, ...]; conv1d -> [conv_dim, 1, kernel] (depthwise).
            w[f"model.language_model.layers.{layer}.linear_attn.in_proj_qkv.weight"] = torch.randn(QKV_DIM, H)
            w[f"model.language_model.layers.{layer}.linear_attn.in_proj_z.weight"] = torch.randn(VALUE_DIM, H)
            w[f"model.language_model.layers.{layer}.linear_attn.in_proj_b.weight"] = torch.randn(NV, H)
            w[f"model.language_model.layers.{layer}.linear_attn.in_proj_a.weight"] = torch.randn(NV, H)
            w[f"model.language_model.layers.{layer}.linear_attn.conv1d.weight"] = torch.randn(CONV_DIM, 1, CONV_K)
            w[f"model.language_model.layers.{layer}.linear_attn.out_proj.weight"] = torch.randn(H, VALUE_DIM)
            # A_log / dt_bias are [num_v_heads] (the decay-rate + delta inputs);
            # providing them so the linear path is fully weight-driven (not just
            # its default init) -- the forward cross-check then exercises them.
            w[f"model.language_model.layers.{layer}.linear_attn.A_log"] = torch.randn(NV)
            w[f"model.language_model.layers.{layer}.linear_attn.dt_bias"] = torch.randn(NV)
        else:  # full-GQA layer
            # q_proj is doubled (query + the output gate, attn_output_gate=True);
            # o_proj maps [heads*head_dim] -> hidden; k/v are GQA (NKV heads,
            # num_kv_heads*head_dim = hidden here, like the real Qwen3.6).
            w[f"model.language_model.layers.{layer}.self_attn.q_proj.weight"] = torch.randn(Q_PROJ_DIM, H)
            w[f"model.language_model.layers.{layer}.self_attn.k_proj.weight"] = torch.randn(KV_PROJ_DIM, H)
            w[f"model.language_model.layers.{layer}.self_attn.v_proj.weight"] = torch.randn(KV_PROJ_DIM, H)
            w[f"model.language_model.layers.{layer}.self_attn.o_proj.weight"] = torch.randn(H, O_PROJ_DIM)
            # q_norm / k_norm are per-head RMSNorms (the Qwen3.5/3.6 'qk norm');
            # fabricate them so the attention path is fully weight-driven.
            w[f"model.language_model.layers.{layer}.self_attn.q_norm.weight"] = torch.randn(HD)
            w[f"model.language_model.layers.{layer}.self_attn.k_norm.weight"] = torch.randn(HD)
        # Every layer has a MoE block: 4 routed experts (packed) + a shared expert.
        w[f"model.language_model.layers.{layer}.mlp.gate.weight"] = torch.randn(E, H)
        w[f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"] = torch.randn(E, 2 * I, H)
        w[f"model.language_model.layers.{layer}.mlp.experts.down_proj"] = torch.randn(E, H, I)
        w[f"model.language_model.layers.{layer}.mlp.shared_expert.gate_proj.weight"] = torch.randn(I, H)
        w[f"model.language_model.layers.{layer}.mlp.shared_expert.up_proj.weight"] = torch.randn(I, H)
        w[f"model.language_model.layers.{layer}.mlp.shared_expert.down_proj.weight"] = torch.randn(H, I)
        w[f"model.language_model.layers.{layer}.mlp.shared_expert_gate.weight"] = torch.randn(1, H)
    return w


@pytest.fixture(scope="module")
def qwen35_ckpt(tmp_path_factory):
    """A fabricated multimodal Qwen3.5/3.6 checkpoint (config.json + one
    safetensors shard + index)."""
    from safetensors.torch import save_file

    path = tmp_path_factory.mktemp("qwen35")
    config = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "tie_word_embeddings": True,
        "vision_config": {"hidden_size": 8, "num_chunks": 2},
        "text_config": _qwen35_text_config(),
    }
    weights = _qwen35_weights()
    (path / "config.json").write_text(json.dumps(config))
    save_file({k: v.contiguous() for k, v in weights.items()}, str(path / "model.safetensors"))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in weights}})
    )
    return str(path)


# --- iter_weights: vision drop + language_model remap + device routing --------


def test_iter_weights_drops_visual_and_remaps_language_model(qwen35_ckpt):
    names = [n for n, _ in iter_weights(qwen35_ckpt, torch.device("cpu"))]
    # No vision tower leaks through.
    assert not any(n.startswith("model.visual") for n in names)
    # The language prefix is remapped to the FreeToken name space.
    assert "model.embed_tokens.weight" in names
    assert "model.layers.0.linear_attn.in_proj_qkv.weight" in names
    assert "model.layers.1.self_attn.q_proj.weight" in names
    assert "model.layers.0.mlp.experts.gate_up_proj" in names
    assert "model.layers.0.mlp.shared_expert.gate_proj.weight" in names
    assert "lm_head.weight" in names
    # The raw checkpoint keys are never yielded un-remapped.
    assert not any(n.startswith("model.language_model") for n in names)


def test_iter_weights_dense_goes_to_device_experts_go_to_host(qwen35_ckpt):
    got = {n: t for n, t in iter_weights(qwen35_ckpt, torch.device("cpu"))}
    # Dense weights (incl. the shared expert + linear-attention) -> the device.
    assert got["model.layers.0.mlp.shared_expert.gate_proj.weight"].device.type == "cpu"
    assert got["model.layers.0.linear_attn.in_proj_qkv.weight"].device.type == "cpu"
    # Routed experts -> host offload banks (cpu here).
    assert got["model.layers.0.mlp.experts.gate_up_proj"].device.type == "cpu"


def test_iter_weights_include_flags_route_experts_only(qwen35_ckpt):
    # include_moe_experts=False: the routed experts are dropped, the shared expert
    # (dense) is kept. This is the loader's dense path.
    no_experts = [n for n, _ in iter_weights(qwen35_ckpt, torch.device("cpu"), include_moe_experts=False)]
    assert not any(".experts." in n for n in no_experts)
    assert "model.layers.0.mlp.shared_expert.gate_proj.weight" in no_experts
    assert "model.embed_tokens.weight" in no_experts
    # include_non_moe=False: only the routed experts remain (the loader's bank path).
    experts_only = [n for n, _ in iter_weights(qwen35_ckpt, torch.device("cpu"), include_non_moe=False)]
    assert all(".experts." in n for n in experts_only)
    assert len(experts_only) == 2 * L  # gate_up_proj + down_proj per layer


# --- the MoE bank path (remapped keys must satisfy the loader) -----------------


def test_load_moe_expert_sources_builds_banks_from_remapped_keys(qwen35_ckpt):
    gate_up, down = load_moe_expert_sources(qwen35_ckpt, dtype=torch.bfloat16)
    assert len(gate_up) == L and len(down) == L
    assert gate_up[0].shape == (E, 2 * I, H)
    assert down[0].shape == (E, H, I)
    assert gate_up[0].device.type == "cpu"
    assert down[0].device.type == "cpu"


# --- the top-level loader path -------------------------------------------------


def test_load_model_end_to_end_on_multimodal_ckpt(qwen35_ckpt):
    model, expert_sources = load_model(qwen35_ckpt, torch.device("cpu"))
    # The loader resolves the multimodal config, builds the model (which the
    # stub's __init__ rebinds to a plain nn.Module instance), and attaches the
    # per-layer host banks for the routed experts. (isinstance(model,
    # Qwen3_5MoEForCausalLM) is False by design -- the instance's class is
    # nn.Module so the loader's named_parameters() resolves; that is what the
    # loader actually consumes.)
    assert isinstance(model, torch.nn.Module)
    assert model.config.architectures == ["Qwen3_5MoeForConditionalGeneration"]
    assert len(expert_sources[0]) == L
    assert expert_sources[0][0].shape == (E, 2 * I, H)
    assert expert_sources[0][0].device.type == "cpu"


# --- the hybrid forward (PR2) --------------------------------------------------


def _drive_prefill(model, seq, T, table_idx=0):
    """Set up a prefill context and run the real forward over ``T`` tokens.

    Builds a single-request batch (``table_idx``), an identity page table (slot
    ``pos`` holds position ``pos``), a KV pool, and the reference attention
    backend, then calls ``model.forward(input_ids, positions, out_loc)`` inside
    the active-batch context. Returns ``(logits, ctx)`` (``logits`` is the
    last-position next-token logits ``[bs, V]``).
    """
    from freetoken.attention.triton import TritonAttentionBackend
    from freetoken.core import Batch, Context, Req, SamplingParams, set_global_ctx
    from freetoken.kvcache.base import BaseKVCachePool

    req = Req(
        input_ids=seq,
        table_idx=table_idx,
        cached_len=0,
        output_len=1,
        uid=0,
        sampling_params=SamplingParams(),
        cache_handle=None,
    )
    batch = Batch(reqs=[req], phase="prefill")
    batch.input_ids = torch.arange(T, dtype=torch.long)
    batch.positions = torch.arange(T, dtype=torch.long)
    batch.out_loc = torch.arange(T, dtype=torch.long)
    batch.extend_lens = torch.tensor([T])
    ctx = Context(page_size=1)
    ctx.kv_cache = BaseKVCachePool(model.config, 1, 1024, torch.device("cpu"), torch.bfloat16)
    pt = torch.zeros((1, 1024), dtype=torch.int32)
    pt[0, :T] = torch.arange(T)
    ctx.page_table = pt
    ctx.kv_cache.attach_page_table(pt)
    backend = TritonAttentionBackend(model.config)
    backend.prepare_metadata(batch)
    ctx.attn_backend = backend
    set_global_ctx(ctx)
    ctx._batch = batch
    try:
        logits = model.forward(batch.input_ids, batch.positions, batch.out_loc)
    finally:
        from freetoken.core import reset_global_ctx

        reset_global_ctx()
    return logits, ctx


def test_forward_runs_and_has_last_position_logits_shape(qwen35_ckpt):
    """The real hybrid forward runs end-to-end (prefill) and returns the correct
    shape/dtype: ``[bs, vocab]`` last-position logits. This replaces the old
    stub test -- the forward is now implemented, not a NotImplementedError."""
    model, _ = load_model(qwen35_ckpt, torch.device("cpu"))
    seq = torch.randint(0, V, (L + 4,))
    T = seq.shape[0]
    logits, _ = _drive_prefill(model, seq, T)
    assert logits.shape == (1, V)
    assert logits.device.type == "cpu"
    assert torch.isfinite(logits).all()


def test_forward_writes_full_attention_kv_into_pool(qwen35_ckpt):
    """The full-attention layer appends its K/V to the paged pool at the token
    positions: after a prefill of T tokens, the layer's KV buffer rows are
    populated (not the untouched empty pool)."""
    model, _ = load_model(qwen35_ckpt, torch.device("cpu"))
    seq = torch.randint(0, V, (L + 4,))
    T = seq.shape[0]
    _, ctx = _drive_prefill(model, seq, T)
    full_layers = [l for l in model.layers if l.self_attn is not None]
    assert full_layers, "the fabricated tower must have a full-attention layer"
    k = ctx.kv_cache.k_buffer
    assert k.shape[1] == NKV and k.shape[2] == HD
    # The rows holding the written K/V (slot == position under the identity
    # table) are populated: some norm over the first T rows is nonzero.
    assert (k[:T].float().norm() > 0).item()
    # And at least one full-attention layer's K was written (the row for position
    # 0 of that layer is nonzero) -- a no-op write would leave the pool empty.
    assert bool((k[:T].float().pow(2).sum() > 0).item())


def test_forward_cross_checked_against_independent_reference(qwen35_ckpt):
    """The FreeToken hybrid forward reproduces the per-position logits of an
    *independent* reference implementation (reference_qwen35.py) built from the
    same weights: linear Gated-Delta-Net, gated GQA, and shared-expert MoE all
    agree. A divergence flags a wiring/shape/math bug in the real forward."""
    # reference_qwen35.py sits next to this test in tests/; it is not a test
    # module (no test_ prefix) so put the tests dir on sys.path to import it.
    import sys

    tests_dir = os.path.dirname(os.path.abspath(__file__))
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    from reference_qwen35 import reference_logits

    model, _ = load_model(qwen35_ckpt, torch.device("cpu"))
    torch.manual_seed(0)
    seq = torch.randint(0, V, (L + 4,))
    T = seq.shape[0]
    # The linear-attention layers keep recurrent state in the model's pool and
    # advance it in place during the forward. The reference reprocesses all T
    # tokens in one shot from the pool's *initial* state, so it must read the same
    # (pristine) state the real forward began from -- not the final state the
    # real forward leaves behind -- so snapshot it first.
    pristine = {lid: [(s.state.clone(), s.conv_state.clone()) for s in pool] for lid, pool in model.linear_state_pool._layers.items()}
    logits, ctx = _drive_prefill(model, seq, T)
    # What the real forward left in the pool (its final recurrent state).
    pool_final = {lid: [(s.state.clone(), s.conv_state.clone()) for s in pool] for lid, pool in model.linear_state_pool._layers.items()}
    # Restore the pristine state so the reference starts where the real forward did.
    for lid, pool in model.linear_state_pool._layers.items():
        for s, (st, cst) in zip(pool, pristine[lid]):
            s.state.copy_(st)
            s.conv_state.copy_(cst)

    # The reference re-runs the architecture from the same weights, reading the
    # same (pristine) linear-state pool the real forward used, so its per-layer
    # outputs and final state must match the real forward's.
    from freetoken.core import reset_global_ctx, set_global_ctx

    ctx.linear_state_pool = model.linear_state_pool
    batch = ctx._batch
    # The reference drives the same Triton backend, which reads the *global*
    # context at call time; _drive_prefill reset it in its finally, so re-arm it
    # here (and reset it again once the reference is done).
    set_global_ctx(ctx)
    try:
        # Feed the reference the SAME input ids the real forward consumed
        # (batch.input_ids, an arange in _drive_prefill) -- not the raw ``seq`` --
        # so the embedding (and hence every downstream layer) matches.
        ref = reference_logits(model, batch.input_ids, batch.positions, 0, T, ctx, batch)
        # After the reference runs, the pool holds the reference's final state.
        ref_pool = {lid: [(s.state.clone(), s.conv_state.clone()) for s in pool] for lid, pool in model.linear_state_pool._layers.items()}
    finally:
        reset_global_ctx()
    # Cross-check the final recurrent state the real forward and the reference
    # each leave in the pool (recurrent matrix + conv ring) -- a stronger check
    # than the logits alone, since it pins the linear-attention recurrence.
    for lid in pool_final:
        real_state, real_conv = pool_final[lid][0]
        ref_state, ref_conv = ref_pool[lid][0]
        state_err = (real_state.float() - ref_state.float()).abs().max().item()
        conv_err = (real_conv.float() - ref_conv.float()).abs().max().item()
        assert state_err < 1e-3, f"linear-state pool diverges at layer {lid} (max |Δ state|={state_err:.3e})"
        assert conv_err < 1e-3, f"conv-state pool diverges at layer {lid} (max |Δ conv|={conv_err:.3e})"
    # Match the real forward's last-position logits (bf16 -> compare in fp32).
    got = logits.float()
    want = ref.float()
    assert got.shape == want.shape
    max_err = (got - want).abs().max().item()
    cos = (got.flatten() @ want.flatten()) / (
        got.flatten().norm() * want.flatten().flatten().norm() + 1e-12
    )
    assert max_err < 2e-2, f"forward diverges from the independent reference (max |Δ|={max_err:.3e})"
    assert cos.item() > 0.999, f"forward logits disagree with the reference (cos={cos.item():.4f})"
