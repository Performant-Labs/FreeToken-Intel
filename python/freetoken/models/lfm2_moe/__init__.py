"""LFM2(.5)-8B-A1B (``lfm2_moe``) -- Intel Arc Pro B70 port. Epic #229.

Upstream reference: HF ``transformers.models.lfm2_moe`` (real, registered
``Lfm2MoeConfig``/``Lfm2MoeForCausalLM``, not this session's cloned
FLashML/FreeToken upstream tree -- that project has no LFM2 port at all;
this is a different upstream entirely). Ground truth for the conv math
below is ``transformers/models/lfm2_moe/modeling_lfm2_moe.py``'s own
``Lfm2MoeShortConv``.

Fill in: GitHub issues `models-lfm2moe-conv` (#230), `models-lfm2moe-attn-moe`
(#231), `models-lfm2moe-e2e` (#232) -- one package, built up incrementally
(this port's established one-``__init__.py``-per-model convention).

This file currently ships #230 only: ``parse_config`` (real) and the short
gated-conv layer primitive (``ShortConv`` / ``short_conv_forward``) below.
``iter_weights``/``Lfm2MoeForCausalLM`` stay stubs (``unimplemented``) until
#232 wires the full model -- the conv primitive is unit-tested standalone
in the meantime (see ``tests/test_models_lfm2moe_conv.py``).

## Hybrid backbone

Unlike every other MoE model in this port, LFM2-MoE's decoder alternates
TWO different layer kinds per ``layer_types`` (confirmed against the real
``LiquidAI/LFM2.5-8B-A1B-Base`` checkpoint's own ``config.json``, NOT
assumed from the class default): most layers are a short causal gated
convolution (this issue), a periodic few (roughly every 4th, starting at
index 2) are full attention (issue #231). The first ``num_dense_layers``
layers use a plain dense MLP; the rest use the sparse MoE (also #231).

## Short gated conv (``ShortConv``, #230)

Ground truth: ``Lfm2MoeShortConv.forward`` in the real
``modeling_lfm2_moe.py``::

    BCx = in_proj(x).transpose(-1, -2)      # [.., 3*hidden, T]
    B, C, x = BCx.chunk(3, dim=-2)          # each [.., hidden, T]
    h = B * x                                # first gate
    h = causal_depthwise_conv1d(h, conv_weight, conv_bias, K=conv_L_cache)
    y = C * h                                # second gate
    y = out_proj(y.transpose(-1, -2))

The causal depthwise conv itself (the real upstream implementation calls
an external fused kernel, ``causal_conv1d_fn``/``causal_conv1d_update``,
whose semantics this port reproduces in pure torch): a standard
left-padded-only depthwise ``Conv1d`` -- ``nn.Conv1d(groups=hidden,
kernel_size=K, padding=K-1)(h)`` produces ``T+K-1`` outputs; the LAST
``K-1`` of those read from the (irrelevant) right zero-padding, so taking
only the first ``T`` outputs gives the causal result: position ``t``'s
output depends only on ``h[t-K+1 .. t]`` (zero-padded on the left for
``t < K-1``). This is exactly ``F.conv1d(..., padding=K-1)[..., :T]``,
confirmed by direct derivation from ``nn.Conv1d``'s own cross-correlation
definition -- ``output[t] = bias + sum_k weight[k] * input[t-(K-1)+k]``,
which for ``t <= T-1`` and ``k <= K-1`` never indexes past the original
input's last position, so the right-padding zeros are never read.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from freetoken._stub import unimplemented
from freetoken.models.config import ModelConfig


# --------------------------------------------------------------------------- #
# Checkpoint side (config parsing)
# --------------------------------------------------------------------------- #


def parse_config(hf_config: Any, model_path: str | None = None, **_kwargs) -> ModelConfig:
    """Build a :class:`ModelConfig` from a real HF ``Lfm2MoeConfig``.

    ``layer_types`` is read verbatim from the checkpoint (a real list of
    ``"conv"``/``"full_attention"`` strings, one per layer) -- do NOT
    assume a fixed ratio; the real ``LiquidAI/LFM2.5-8B-A1B-Base``
    checkpoint's own pattern is `[conv, conv, full_attention, conv, conv,
    conv, full_attention, ...]` (attention roughly every 4th layer,
    starting at index 2), confirmed directly from its ``config.json`` --
    but the config class ships no default builder for this field at all
    (unlike e.g. Mellum's ``["full_attention"] * num_hidden_layers``
    default), so a checkpoint that omits it entirely has no sane fallback
    and this raises rather than guessing.
    """
    src = hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(hf_config)
    layer_types = src.get("layer_types")
    num_layers = src.get("num_hidden_layers")
    if not layer_types:
        raise ValueError(
            "lfm2_moe checkpoint's config.json has no 'layer_types' -- this "
            "architecture has no derivable default (unlike e.g. Mellum's "
            "all-full_attention fallback), so the real conv/attention "
            "pattern must be present, not guessed"
        )
    if num_layers is not None and len(layer_types) != num_layers:
        raise ValueError(
            f"layer_types has {len(layer_types)} entries but num_hidden_layers={num_layers}"
        )

    cfg = ModelConfig(
        architectures=["Lfm2MoeForCausalLM"],
        hidden_size=src.get("hidden_size"),
        vocab_size=src.get("vocab_size"),
        num_layers=num_layers,
        num_experts=src.get("num_experts") or src.get("num_local_experts"),
        num_attention_heads=src.get("num_attention_heads"),
        num_key_value_heads=src.get("num_key_value_heads"),
        intermediate_size=src.get("intermediate_size"),
        moe_intermediate_size=src.get("moe_intermediate_size"),
        num_experts_per_tok=src.get("num_experts_per_tok"),
        first_k_dense_replace=src.get("num_dense_layers") or 0,
        is_moe=True,
        max_position_embeddings=src.get("max_position_embeddings"),
        tie_word_embeddings=src.get("tie_word_embeddings", False),
        rope_theta=(src.get("rope_parameters") or {}).get("rope_theta"),
        hidden_act=src.get("hidden_act", "silu"),
        dtype=src.get("dtype") or src.get("torch_dtype"),
    )
    cfg.attrs["layer_types"] = list(layer_types)
    cfg.attrs["conv_bias"] = bool(src.get("conv_bias", False))
    cfg.attrs["conv_L_cache"] = int(src.get("conv_L_cache", 3))
    cfg.attrs["num_dense_layers"] = int(src.get("num_dense_layers") or 0)
    cfg.attrs["norm_eps"] = src.get("norm_eps", 1e-5)
    cfg.attrs["norm_topk_prob"] = bool(src.get("norm_topk_prob", True))
    cfg.attrs["use_expert_bias"] = bool(src.get("use_expert_bias", False))
    cfg.attrs["routed_scaling_factor"] = src.get("routed_scaling_factor", 1.0)
    return cfg


def iter_weights(*args, **kwargs):
    unimplemented("iter_weights", "models-lfm2moe-e2e")


# --------------------------------------------------------------------------- #
# Forward side: the short gated-conv primitive (#230 only)
# --------------------------------------------------------------------------- #


def causal_depthwise_conv1d(
    h: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, kernel_size: int
) -> torch.Tensor:
    """Left-padded-only depthwise causal conv over the time axis.

    ``h [.., C, T]`` (channels, time -- matches ``nn.Conv1d``'s own layout),
    ``weight [C, kernel_size]`` (one filter per channel, depthwise),
    ``bias [C]`` or ``None``. Returns ``[.., C, T]``: position ``t`` depends
    only on ``h[.., t-kernel_size+1 .. t]`` (left-zero-padded).
    """
    padded = F.pad(h, (kernel_size - 1, 0))
    out = F.conv1d(padded, weight.unsqueeze(1), bias=bias, groups=h.shape[-2])
    return out


class ShortConv(nn.Module):
    """LFM2-MoE's short gated causal convolution (real math: see module docstring).

    Standalone/testable: takes and returns plain ``[T, hidden]`` (no batch
    dim, matching this port's own single-request-per-call convention used
    by every other model package here). Not yet wired into a decoder layer
    or the engine -- that is #232's job.
    """

    def __init__(self, hidden_size: int, kernel_size: int, has_bias: bool, dtype=None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.in_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=has_bias, dtype=dtype)
        self.conv_weight = nn.Parameter(torch.empty(hidden_size, kernel_size, dtype=dtype))
        self.conv_bias = nn.Parameter(torch.empty(hidden_size, dtype=dtype)) if has_bias else None
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=has_bias, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x [T, hidden] -> [T, hidden]``."""
        bcx = self.in_proj(x).transpose(0, 1)  # [3*hidden, T]
        b, c, gx = bcx.chunk(3, dim=0)  # each [hidden, T]
        h = b * gx
        h = causal_depthwise_conv1d(h.unsqueeze(0), self.conv_weight, self.conv_bias, self.kernel_size).squeeze(0)
        y = c * h
        return self.out_proj(y.transpose(0, 1))


# --------------------------------------------------------------------------- #
# GQA attention with per-head q/k RMSNorm (#231)
#
# Ground truth: ``Lfm2MoeAttention.forward`` in the real ``modeling_lfm2_moe.py``.
# The issue that filed this work assumed "no QK-norm per the config class" --
# that is WRONG: the config class has no boolean flag for it, but the real
# modeling code unconditionally builds ``q_layernorm``/``k_layernorm``
# (``Lfm2MoeRMSNorm(head_dim, eps=norm_eps)``) and applies them before RoPE,
# every layer. Confirmed by reading the real forward, not assumed from the
# issue text. Otherwise this is the same GQA + half-split (rotate_half) RoPE
# shape as this port's own ``qwen3_moe`` attention -- adapted from it
# (identical KV-pool contract, including the physical-slot ``write_kv`` fix),
# not shared by reference (every model package in this port stands alone).
# --------------------------------------------------------------------------- #


class Lfm2MoeAttention(nn.Module):
    """LFM2-MoE full-attention layer: GQA + per-head q/k RMSNorm + half-split RoPE."""

    def __init__(self, config: ModelConfig, device, dtype, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads or config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.num_heads
        norm_eps = config.attrs.get("norm_eps", 1e-5)

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False, dtype=dtype)
        self.q_layernorm = nn.RMSNorm(self.head_dim, eps=norm_eps, dtype=dtype)
        self.k_layernorm = nn.RMSNorm(self.head_dim, eps=norm_eps, dtype=dtype)

        theta = config.rope_theta or 10000.0
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def _rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # Half-split (rotate_half), matching the real ``modeling_lfm2_moe.py``'s
        # own ``rotate_half``/``apply_rotary_pos_emb`` -- byte-for-byte the same
        # formula as HF Qwen3/Llama-family rope (this port's ``qwen3_moe`` had a
        # real, confirmed bug here from an interleaved split mismatching this
        # cos/sin layout; use the already-fixed half-split form from the start).
        freqs = torch.outer(pos.to(torch.float32), self.inv_freq)  # [N, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [N, D]
        cos = emb.cos()[None, :, :]
        sin = emb.sin()[None, :, :]
        x_f = x.to(torch.float32)
        half = x_f.shape[-1] // 2
        x1, x2 = x_f[..., :half], x_f[..., half:]
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x_f * cos + rotated * sin).to(x.dtype)

    def forward(self, hidden_states, positions, table_idx, ctx, batch):
        bsz, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(hidden_states).view(bsz, self.num_kv_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(hidden_states).view(bsz, self.num_kv_heads, self.head_dim).transpose(0, 1)
        q = self._rope(self.q_layernorm(q), positions)
        k = self._rope(self.k_layernorm(k), positions)
        # write_kv's out_loc is a PHYSICAL pool slot, not a logical token
        # position -- MHAKVCache's real allocator reserves slot 0 as padding,
        # so the two never coincide for a real request (see qwen3_moe's own
        # extensive comment on this, PR #234/#236 -- the same fix applied here
        # from the start rather than retrofit).
        out_loc = ctx.page_table[table_idx, positions.long()]
        ctx.kv_cache.write_kv(k, v, out_loc, self.layer_id)
        out = ctx.attn_backend.forward(q, k, v, self.layer_id, batch, table_idx=table_idx)
        return self.out_proj(out.transpose(0, 1).contiguous().reshape(bsz, -1))


# --------------------------------------------------------------------------- #
# Bias-corrected sigmoid top-k router + expert combination (#231)
#
# Ground truth: ``Lfm2MoeTopKRouter``/``Lfm2MoeExperts``/``Lfm2MoeSparseMoeBlock``
# in the real ``modeling_lfm2_moe.py``. Confirmed against this port's own
# ``deepseek_v4``/``glm_moe_dsa`` ``noaux_tc``-style routers: the SAME shape
# (sigmoid scores; select top-k on scores+bias but gather WEIGHTS from the
# raw, uncorrected sigmoid scores; optional renorm; scale by
# ``routed_scaling_factor``) with two real differences confirmed from the
# actual HF source, not assumed: (1) no grouped-expert pre-filtering at all
# (LFM2's config has no ``n_group``/``topk_group`` -- flat top-k over every
# expert, unlike DeepSeek-V3/V4's group-then-topk), and (2) no shared expert
# (``Lfm2MoeSparseMoeBlock`` has none, unlike ``deepseek_v4``/``glm_moe_dsa``).
# The bias itself is a plain zero-initialized buffer (``expert_bias``, only
# present when ``use_expert_bias`` is set), not a checkpoint-trained weight --
# same convention as DeepSeek's ``e_score_correction_bias``.
# --------------------------------------------------------------------------- #


class Lfm2MoeTopKRouter(nn.Module):
    """Real HF ``Lfm2MoeTopKRouter`` math: sigmoid scores, optional additive
    bias for SELECTION only (weights are gathered from the raw, uncorrected
    scores), optional renorm, then ``routed_scaling_factor``."""

    def __init__(self, config: ModelConfig, dtype) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.attrs.get("norm_topk_prob", True)
        self.routed_scaling_factor = config.attrs.get("routed_scaling_factor", 1.0)
        self.use_expert_bias = config.attrs.get("use_expert_bias", False)
        self.weight = nn.Parameter(torch.empty(self.num_experts, config.hidden_size, dtype=dtype))
        if self.use_expert_bias:
            self.register_buffer("expert_bias", torch.zeros(self.num_experts, dtype=torch.float32))
        else:
            self.expert_bias = None

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``hidden_states [T, H]``. Returns ``(selected_experts [T, k],
        topk_weights [T, k])``."""
        router_logits = F.linear(hidden_states.float(), self.weight.float())  # [T, E]
        routing_weights = router_logits.sigmoid()
        if self.use_expert_bias:
            scores_for_routing = routing_weights + self.expert_bias.unsqueeze(0)
            _, selected_experts = torch.topk(scores_for_routing, k=self.top_k, dim=-1)
            topk_weights = torch.gather(routing_weights, dim=1, index=selected_experts)
        else:
            topk_weights, selected_experts = torch.topk(routing_weights, k=self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)
        topk_weights = topk_weights * self.routed_scaling_factor
        return selected_experts, topk_weights


class Lfm2MoeExperts(nn.Module):
    """Real HF ``Lfm2MoeExperts``: one fused 3D parameter per projection
    (``gate_up_proj [E, 2*inter, H]``, ``down_proj [E, H, inter]``) -- matches
    the real checkpoint's own weight layout (``base_model_ep_plan``:
    ``feed_forward.experts.gate_up_proj``/``down_proj``), not per-expert
    ``nn.Linear`` submodules. One-hot + ``index_add_`` dispatch, confirmed
    byte-for-byte against the real forward (not a reference reimplementation
    with different math -- this port's usual "pure-torch reference" cut is
    the fused-kernel absence, not the dispatch algorithm itself here)."""

    def __init__(self, config: ModelConfig, dtype) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim, dtype=dtype)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim, dtype=dtype)
        )

    def forward(
        self, hidden_states: torch.Tensor, selected_experts: torch.Tensor, topk_weights: torch.Tensor
    ) -> torch.Tensor:
        """``hidden_states [T, H]``, ``selected_experts``/``topk_weights [T, k]``."""
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = F.silu(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * topk_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


class Lfm2MoeSparseMoeBlock(nn.Module):
    """Router + experts, standalone (not yet wired into a decoder layer --
    that is #232's job)."""

    def __init__(self, config: ModelConfig, dtype) -> None:
        super().__init__()
        self.gate = Lfm2MoeTopKRouter(config, dtype)
        self.experts = Lfm2MoeExperts(config, dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        selected_experts, topk_weights = self.gate(hidden_states)
        return self.experts(hidden_states, selected_experts, topk_weights)


# --------------------------------------------------------------------------- #
# Not yet implemented: full model wiring (#232).
# --------------------------------------------------------------------------- #


class Lfm2MoeForCausalLM:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(self, *args, **kwargs):
        unimplemented("Lfm2MoeForCausalLM.forward", "models-lfm2moe-e2e")


__all__ = [
    "ShortConv",
    "causal_depthwise_conv1d",
    "iter_weights",
    "Lfm2MoeAttention",
    "Lfm2MoeExperts",
    "Lfm2MoeForCausalLM",
    "Lfm2MoeSparseMoeBlock",
    "Lfm2MoeTopKRouter",
    "parse_config",
]
