"""End-to-end: a rewritten tool-call resumes GDN state from the anchor
instead of recomputing it (issue `semantic-cache-e2e`, #172 -- the final
child of the `semantic-cache` epic, #32).

The actual point of the whole epic, proven with a real forward pass (not
just the isolated-piece unit tests #169/#170/#171 already wrote): a small
fabricated Qwen3.5-shaped hybrid (linear-attention + full-attention)
checkpoint, run through the real ``Engine``, mirroring this project's own
established e2e style (``test_qwen35_gptq_e2e_loader.py`` et al: a few-KB
synthetic checkpoint, not a real multi-GB one).

Scenario:
  1. Request 1 prefills a prompt and decodes to completion (greedy,
     deterministic -- same weights + prompt always sample the same
     tokens). ``engine.toolcall_anchor_id`` is armed to the token this
     model actually samples FIRST (learned via a throwaway probe run,
     matching this port's own established technique for "inject a real
     anchor without depending on a model choosing to call a tool" --
     see tests/test_toolcall_anchor.py), so the anchor lands right after
     the prompt: a real, working (not synthetic-forced) GDN snapshot.
  2. Request 2's prompt is request 1's prompt + its first generated
     token (the shared "echoed tool call") + ONE new, different token
     (the "client-side rewrite") -- exactly reproducing request 1 up to
     the anchor, then diverging.
  3. Confirms: engine.add_request restores request 2's GDN state from the
     frozen snapshot (not from zero) -- proven the strong way, by matching
     request 2's ENTIRE generated continuation against a COLD run of the
     same (rewritten) prompt on a fresh engine, bit-identical. GDN
     recurrence is a pure function of token history, so a wrong (zero or
     partial) restore would diverge these two runs; only a byte-correct
     restore reproduces the cold run exactly.
  4. A coarse, real "does less work" signal (this issue's own Accept
     bar): a spy on ``_GatedDeltaNet._delta_rule`` counts the total
     tokens the recurrent core actually processes for request 2's
     prefill step -- the cache/restore path must process strictly fewer
     than request 2's full prompt length (it only extends the tail past
     ``cached_len``), unlike a cold run which processes the whole thing.

Deliberate scope cut (documented in engine.py's own comments): restore
routing goes through a flat, never-evicted ``Engine._mamba_anchor_snapshots``
lookaside keyed by the exact token prefix, not the full HybridRadixCache
tree-donation/eviction machinery #169 already built for the KV+mamba-node
case -- proving the restore MECHANISM end-to-end doesn't need that tree's
ownership/eviction machinery; wiring HybridRadixCache into CacheManager in
place of the plain RadixPrefixCache it wraps today is real, separable
follow-up work.

CPU-only (DEVICE = "cpu", like test_engine_loop.py) -- runs in the
CPU-only CI runner; an XPU run is the same engine/model path on a
different device, not a new code path to test separately here.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from freetoken.core import Req, SamplingParams, reset_global_ctx
from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.engine.engine import Engine

DEVICE = "cpu"

H, I, E, V, L = 32, 16, 4, 64, 2
NH, NKV, HD = 4, 2, 16
NK, NV = 2, 2
KD, VD = 16, 16
KEY_DIM, VALUE_DIM = NK * KD, NV * VD
QKV_DIM = KEY_DIM * 2 + VALUE_DIM
CONV_DIM = KEY_DIM * 2 + VALUE_DIM
CONV_K = 4
Q_PROJ_DIM = NH * HD * 2  # q_proj always fuses query + output gate (_Qwen35Attention)
O_PROJ_DIM = NH * HD
KV_PROJ_DIM = NKV * HD


@pytest.fixture(autouse=True)
def _clean_global_ctx():
    yield
    reset_global_ctx()


def _text_config() -> dict:
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
        "attn_output_gate": False,
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


def _weights() -> dict:
    g = torch.Generator().manual_seed(0)
    w = {
        "model.language_model.embed_tokens.weight": torch.randn(V, H, generator=g),
        "model.language_model.norm.weight": torch.randn(H, generator=g),
        "lm_head.weight": torch.randn(V, H, generator=g),
    }
    for layer in range(L):
        if layer % 2 == 0:  # linear-attention (Gated-Delta-Net)
            p = f"model.language_model.layers.{layer}.linear_attn"
            w[f"{p}.in_proj_qkv.weight"] = torch.randn(QKV_DIM, H, generator=g)
            w[f"{p}.in_proj_z.weight"] = torch.randn(VALUE_DIM, H, generator=g)
            w[f"{p}.in_proj_b.weight"] = torch.randn(NV, H, generator=g)
            w[f"{p}.in_proj_a.weight"] = torch.randn(NV, H, generator=g)
            w[f"{p}.conv1d.weight"] = torch.randn(CONV_DIM, 1, CONV_K, generator=g)
            w[f"{p}.out_proj.weight"] = torch.randn(H, VALUE_DIM, generator=g)
            w[f"{p}.A_log"] = torch.randn(NV, generator=g)
            w[f"{p}.dt_bias"] = torch.randn(NV, generator=g)
        else:  # full-attention (gated GQA)
            p = f"model.language_model.layers.{layer}.self_attn"
            w[f"{p}.q_proj.weight"] = torch.randn(Q_PROJ_DIM, H, generator=g)
            w[f"{p}.k_proj.weight"] = torch.randn(KV_PROJ_DIM, H, generator=g)
            w[f"{p}.v_proj.weight"] = torch.randn(KV_PROJ_DIM, H, generator=g)
            w[f"{p}.o_proj.weight"] = torch.randn(H, O_PROJ_DIM, generator=g)
            w[f"{p}.q_norm.weight"] = torch.randn(HD, generator=g)
            w[f"{p}.k_norm.weight"] = torch.randn(HD, generator=g)
        m = f"model.language_model.layers.{layer}.mlp"
        w[f"{m}.gate.weight"] = torch.randn(E, H, generator=g)
        w[f"{m}.experts.gate_up_proj"] = torch.randn(E, 2 * I, H, generator=g)
        w[f"{m}.experts.down_proj"] = torch.randn(E, H, I, generator=g)
        w[f"{m}.shared_expert.gate_proj.weight"] = torch.randn(I, H, generator=g)
        w[f"{m}.shared_expert.up_proj.weight"] = torch.randn(I, H, generator=g)
        w[f"{m}.shared_expert.down_proj.weight"] = torch.randn(H, I, generator=g)
        w[f"{m}.shared_expert_gate.weight"] = torch.randn(1, H, generator=g)
    return w


@pytest.fixture(scope="module")
def hybrid_ckpt(tmp_path_factory):
    from safetensors.torch import save_file

    path = tmp_path_factory.mktemp("qwen35_hybrid_e2e")
    config = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "tie_word_embeddings": True,
        "text_config": _text_config(),
    }
    weights = _weights()
    (path / "config.json").write_text(json.dumps(config))
    save_file({k: v.contiguous() for k, v in weights.items()}, str(path / "model.safetensors"))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in weights}})
    )
    return str(path)


def _cfg(model_path: str, **overrides) -> EngineConfig:
    kwargs = dict(
        model_path=model_path,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.float32,
        device=DEVICE,
        attention_backend="auto",
        max_running_req=2,
        page_size=1,
        max_seq_len_override=32,
        num_page_override=256,
    )
    kwargs.update(overrides)
    return EngineConfig(**kwargs)


def _req(uid: int, ids: list[int], output_len: int) -> Req:
    return Req(
        input_ids=list(ids),
        table_idx=0,
        cached_len=0,
        output_len=output_len,
        uid=uid,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=output_len),
        cache_handle=None,
    )


PROMPT = [5, 9, 13, 17]
REWRITE_TAIL = 42  # a token that never appears anywhere in PROMPT/the probe run


def _delta_rule_token_count_spy():
    """Patches _GatedDeltaNet._delta_rule to also record the total T
    (sequence length) it is called with, across every call -- a coarse,
    real measure of how many tokens the recurrent core actually
    processed, this issue's own Accept-bar signal for "less recompute"."""
    from freetoken.models.qwen3_5_moe import _GatedDeltaNet

    counts: list[int] = []
    original = _GatedDeltaNet._delta_rule

    def spy(self, q, k, v, g, beta, slot, out_dtype=None):
        counts.append(q.shape[1])  # q is [B, T, num_v, D] on entry
        return original(self, q, k, v, g, beta, slot, out_dtype=out_dtype)

    return patch.object(_GatedDeltaNet, "_delta_rule", spy), counts


def test_rewritten_toolcall_restores_gdn_state_from_the_anchor(hybrid_ckpt):
    # --- Learn the real (deterministic) first-sampled token, to arm a
    # real anchor without depending on this random-weight model actually
    # choosing to call a tool (mirrors test_toolcall_anchor.py's own
    # technique).
    probe = Engine(_cfg(hybrid_ckpt))
    probe.add_request(_req(0, PROMPT, output_len=1))
    first_token = probe.generate()[0][0]
    reset_global_ctx()
    assert first_token != REWRITE_TAIL

    # --- Request 1: prefill + decode to completion, with the anchor armed
    # at the model's own real first-sampled token. Finishing commits its
    # full sequence into the prefix cache AND (issue #172) freezes its GDN
    # state at the anchor into a ping-pong track slot.
    engine = Engine(_cfg(hybrid_ckpt, enable_prefix_cache=True))
    engine.toolcall_anchor_id = first_token
    engine.add_request(_req(0, PROMPT, output_len=3))
    engine.generate()
    assert len(engine._mamba_anchor_snapshots) == 1  # noqa: SLF001
    anchor_prefix = next(iter(engine._mamba_anchor_snapshots))  # noqa: SLF001
    assert anchor_prefix == tuple(PROMPT + [first_token])

    # --- Request 2: request 1's prompt, plus its first generated token
    # (the shared echoed tool call), plus ONE new diverging token (the
    # client-side rewrite) -- reproduces request 1 exactly up to the
    # anchor, then diverges.
    rewritten_prompt = PROMPT + [first_token, REWRITE_TAIL]
    spy_ctx, counts = _delta_rule_token_count_spy()
    with spy_ctx:
        engine.add_request(_req(1, rewritten_prompt, output_len=3))
        restored_out = engine.generate()[0]
    # The cache/restore path only extends the un-cached tail (the one
    # diverging token) during prefill, not the whole 6-token prompt --
    # the coarse "less recompute" signal this issue's Accept bar asks for.
    assert counts[0] < len(rewritten_prompt)
    assert counts[0] == 1

    # --- Cold baseline: a FRESH engine (no prior history, prefix caching
    # off -- nothing to restore from), fed the exact same rewritten
    # prompt from scratch. A wrong (zero-state, or stale) restore would
    # make request 2's continuation diverge from this; only a byte-
    # correct restore reproduces it exactly, since GDN recurrence is a
    # pure function of token history.
    cold = Engine(_cfg(hybrid_ckpt, enable_prefix_cache=False))
    cold.add_request(_req(0, rewritten_prompt, output_len=3))
    cold_out = cold.generate()[0]

    assert restored_out == cold_out
