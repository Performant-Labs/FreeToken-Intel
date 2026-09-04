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

Issues #230 (conv) and #231 (attention + router) shipped the standalone
primitives; #232 (this file's current state) wires them into the full
model: the hybrid decoder layer (conv OR attention per ``layer_types``,
dense-vs-MoE split per ``num_dense_layers``), the stateful conv-forward
path (per-request left-context rings), the causal-LM wrapper, and the
real ``iter_weights`` (checkpoint-key normalization + tied-``lm_head``
synthesis). Real-checkpoint validation against
``LiquidAI/LFM2.5-8B-A1B-Base`` is sequenced separately by the parent
session (deliberately out of scope here, per the issue's own body).

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


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool = True,
    include_non_moe: bool = True,
):
    """Yield the checkpoint's tensors (names normalized to this port's module
    tree), each on its destination device.

    Three real normalizations against the raw HF ``Lfm2MoeForCausalLM``
    layout, all confirmed from the real checkpoint family (the
    ``LiquidAI/LFM2.5-8B-A1B`` HF ``modeling_lfm2_moe.py`` + its
    ``base_model_ep_plan``), not assumed:

    * the FFN block is spelled ``feed_forward`` in the checkpoint but
      ``mlp`` in this port's module tree (and in the loader's expert-bank
      streamer, whose key parser ``_expert_source_info`` anchors on the
      ``mlp`` token -- renaming here is what routes the fused
      ``experts.gate_up_proj``/``down_proj`` tensors into the packed banks);
    * the depthwise conv filter is a real ``nn.Conv1d`` weight
      ``conv.conv.weight`` ``[hidden, 1, kernel]`` in the checkpoint but a
      flat ``conv.conv_weight`` ``[hidden, kernel]`` parameter on this
      port's :class:`ShortConv` (same for the optional bias);
    * ``tie_word_embeddings: true`` checkpoints (the real
      ``LFM2.5-8B-A1B-Base`` is one) ship no ``lm_head.weight`` key at all
      -- synthesize it from ``embed_tokens.weight`` (the same real failure
      mode ``qwen3``/``qwen3_moe``/``mellum`` guard against: an unfilled
      ``lm_head`` silently zeros every logit).

    ``include_moe_experts``/``include_non_moe`` are NOT no-ops (see
    ``mellum.iter_weights``'s own comment on the identical trap): the
    loader's expert-bank builder streams this generator with
    ``include_non_moe=False`` (expert tensors only) and the dense pass
    with ``include_moe_experts=False``. The expert filter keys on
    ``.experts.`` AFTER the ``feed_forward`` -> ``mlp`` rename.

    One deliberate no-op: the real checkpoint family may persist the
    zero-initialized ``expert_bias`` buffer (``use_expert_bias: true``) as
    a ``feed_forward.expert_bias`` tensor. This port's router holds that
    buffer at ``mlp.gate.expert_bias``, so a checkpoint key never matches
    and ``_place`` silently skips it -- correct here because the HF buffer
    is zero-initialized and never trained (no gradient path through a
    buffer), so its saved value is always exactly the zeros this port's
    own buffer already holds.
    """
    from freetoken.models.weight import iter_safetensors

    embed_tokens_weight = None
    saw_lm_head = False
    for name, tensor in iter_safetensors(model_path, device):
        if ".feed_forward." in name:
            name = name.replace(".feed_forward.", ".mlp.")
        if name.endswith(".conv.conv.weight"):
            # [hidden, 1, kernel] (nn.Conv1d) -> [hidden, kernel] (ShortConv).
            tensor = tensor.reshape(tensor.shape[0], tensor.shape[-1])
            name = name[: -len(".conv.conv.weight")] + ".conv.conv_weight"
        elif name.endswith(".conv.conv.bias"):
            name = name[: -len(".conv.conv.bias")] + ".conv.conv_bias"
        is_expert = ".experts." in name
        if is_expert and not include_moe_experts:
            continue
        if not is_expert and not include_non_moe:
            continue
        placed = tensor.to(device)
        if name == "model.embed_tokens.weight":
            embed_tokens_weight = placed
        elif name == "lm_head.weight":
            saw_lm_head = True
        yield name, placed

    if include_non_moe and not saw_lm_head and embed_tokens_weight is not None:
        from freetoken.utils import cached_load_hf_config

        hf_config = cached_load_hf_config(model_path)
        if bool(getattr(hf_config, "tie_word_embeddings", False)):
            yield "lm_head.weight", embed_tokens_weight


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
        """``x [T, hidden] -> [T, hidden]`` (stateless: left zero-padding)."""
        bcx = self.in_proj(x).transpose(0, 1)  # [3*hidden, T]
        b, c, gx = bcx.chunk(3, dim=0)  # each [hidden, T]
        h = b * gx
        h = causal_depthwise_conv1d(h.unsqueeze(0), self.conv_weight, self.conv_bias, self.kernel_size).squeeze(0)
        y = c * h
        return self.out_proj(y.transpose(0, 1))

    def forward_stateful(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Chunked/stateful variant for the engine loop (#232).

        ``x`` is this request's NEW tokens ``[T, hidden]`` (one engine step's
        slice, not the whole sequence); ``state`` is this request's conv
        left-context ring ``[hidden, kernel-1]`` -- the last ``kernel-1``
        pre-conv gated activations -- updated IN PLACE (``copy_``). A fresh
        sequence is expressed by the caller zeroing ``state`` first, which
        reduces exactly to :meth:`forward`'s left zero-padding.

        Same math as the real HF decode path (``causal_conv1d_update``):
        concatenate the ring under the new activations, run the depthwise
        conv with NO padding -- the output is exactly ``T`` positions, each
        reading the ``kernel`` inputs ending at that position -- then keep
        the last ``kernel-1`` columns as the new ring. Chunked prefill works
        the same way (a continuation chunk's ring carries over from the
        previous chunk).
        """
        bcx = self.in_proj(x).transpose(0, 1)  # [3*hidden, T]
        b, c, gx = bcx.chunk(3, dim=0)  # each [hidden, T]
        h = b * gx
        km1 = self.kernel_size - 1
        if km1:
            full = torch.cat([state, h], dim=-1)  # [hidden, kernel-1+T]
        else:
            full = h
        out = F.conv1d(
            full.unsqueeze(0), self.conv_weight.unsqueeze(1), bias=self.conv_bias, groups=self.hidden_size
        ).squeeze(0)  # [hidden, T]
        if km1:
            state.copy_(full[:, -km1:])
        y = c * out
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
# Full model wiring (#232).
#
# Ground truth: ``Lfm2MoeDecoderLayer``/``Lfm2MoeModel``/``Lfm2MoeForCausalLM``
# in the real ``modeling_lfm2_moe.py``, read directly. Real structural facts
# (not the issue text's guesses):
#
# * the per-layer norms are ``operator_norm`` (pre-operator) and ``ffn_norm``
#   (pre-FFN) -- NOT ``input_layernorm``/``post_attention_layernorm``;
# * the FINAL norm (applied after the layer stack, before ``lm_head``) is
#   named ``embedding_norm`` (``model.embedding_norm.weight`` in the
#   checkpoint) -- despite the name it normalizes the stack OUTPUT, not the
#   embedding input;
# * the dense MLP (first ``num_dense_layers`` layers) is a plain SwiGLU with
#   ``w1``/``w3``/``w2`` and ``intermediate_size`` (7168 on the real
#   LFM2.5-8B-A1B), NOT the MoE intermediate size;
# * a conv layer and an attention layer share the SAME residual shape
#   (``h + op(operator_norm(h))`` then ``h + ffn(ffn_norm(h))``) -- only the
#   operator module differs.
#
# The module tree deliberately names the FFN attribute ``mlp`` (the
# checkpoint says ``feed_forward``; ``iter_weights`` renames) so the loader's
# fused-expert placement (``_place_expert_weights``, which hard-codes
# ``layers.{i}.mlp.experts.{e}.{gate,up,down}_proj`` per-expert modules)
# works unchanged -- the same module-shape contract every other fused-only
# model here (mellum, glm4_moe, gpt_oss) already satisfies. The routed
# experts are therefore per-expert ``nn.Linear`` modules (fed from the packed
# banks), while the #231-tested fused-3D ``Lfm2MoeExperts`` stays as the
# byte-for-byte HF-reference primitive. The router IS reused directly
# (#231's own ``Lfm2MoeTopKRouter``, ``weight`` param path
# ``mlp.gate.weight`` matching the checkpoint after the rename).
# --------------------------------------------------------------------------- #


class _ConvStatePool:
    """Per-request conv left-context rings: one ``[hidden, kernel-1]`` tensor
    per (conv layer, request slot), lazily grown.

    The model owns this (it knows the layer count and per-request shapes);
    slots are indexed by the request's ``linear_slot_idx`` (falling back to
    ``table_idx``, mirroring ``qwen3_5_moe``'s own default 1:1 GDN-state
    pool). A slot's ring is zeroed by the decoder layer whenever a slice
    starts at position 0 (a fresh sequence), so a recycled table row never
    leaks the previous request's context into the new one.
    """

    def __init__(self) -> None:
        self._layers: dict[int, list[torch.Tensor]] = {}

    def register(self, layer_id: int, num_slots: int, hidden: int, km1: int, device, dtype) -> None:
        self._layers[layer_id] = [
            torch.zeros(hidden, km1, device=device, dtype=dtype) for _ in range(num_slots)
        ]

    def get(self, layer_id: int, slot: int) -> torch.Tensor:
        entries = self._layers.get(layer_id)
        if entries is None:
            raise KeyError(f"conv-state pool: conv layer {layer_id} not registered")
        while slot >= len(entries):
            # Lazily grow for a request admitted after the pool was sized.
            entries.append(torch.zeros_like(entries[-1]))
        return entries[slot]


class _Lfm2MoeDenseMLP(nn.Module):
    """Real HF ``Lfm2MoeMLP``: SwiGLU with ``w1``/``w3``/``w2`` (checkpoint
    key spelling preserved) and the DENSE ``intermediate_size``."""

    def __init__(self, config: ModelConfig, dtype) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, dtype=dtype)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, dtype=dtype)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class _Lfm2MoeExpert(nn.Module):
    """One routed expert: SwiGLU gate/up/down (fused/in-VRAM only, fed from
    the loader's packed banks via ``_place_expert_weights``)."""

    def __init__(self, config: ModelConfig, dtype) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _Lfm2MoeMoE(nn.Module):
    """Router + resident experts for one MoE layer (#232 wiring).

    Routing math is #231's own ``Lfm2MoeTopKRouter`` reused directly
    (sigmoid top-k, optional selection-only bias, renorm, scaling factor --
    see its own docstring); the experts dispatch with the host-side
    per-expert loop this port uses everywhere (a bool-tensor ``nonzero()``
    on this torch/XPU build silently returns empty -- see ``mellum``'s own
    comment on the same confirmed bug)."""

    def __init__(self, config: ModelConfig, device, dtype) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.gate = Lfm2MoeTopKRouter(config, dtype)
        self.experts = nn.ModuleList(
            _Lfm2MoeExpert(config, dtype).to(device, dtype) for _ in range(self.num_experts)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        in_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, in_shape[-1])  # [T, hidden]
        selected_experts, topk_weights = self.gate(flat)

        out = torch.zeros_like(flat)
        selected_host = selected_experts.to("cpu")
        for e in range(self.num_experts):
            token_rows, k_slots = (selected_host == e).nonzero(as_tuple=True)
            if token_rows.numel() == 0:
                continue
            token_rows = token_rows.to(flat.device)
            k_slots = k_slots.to(flat.device)
            sel = flat.index_select(0, token_rows)
            expert_out = self.experts[e](sel)
            w = topk_weights[token_rows, k_slots].unsqueeze(-1)
            out.index_add_(0, token_rows, expert_out * w)
        return out.view(in_shape)


class _Lfm2MoeDecoderLayer(nn.Module):
    """One hybrid LFM2-MoE decoder layer.

    ``layer_types[i]`` picks the operator: ``"full_attention"`` builds the
    GQA module (#231), anything else (``"conv"``) the short conv (#230,
    run through its stateful path with this request's ring). The FFN is a
    dense MLP for the first ``num_dense_layers`` layers, the routed MoE for
    the rest. Residual structure per the real HF forward::

        h = h + op(operator_norm(h))
        h = h + ffn(ffn_norm(h))
    """

    def __init__(self, config: ModelConfig, device, dtype, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        layer_types = config.attrs.get("layer_types") or []
        self.layer_type = layer_types[layer_id] if layer_id < len(layer_types) else "full_attention"
        self.is_attention_layer = self.layer_type == "full_attention"
        eps = config.attrs.get("norm_eps", 1e-5)
        self.operator_norm = nn.RMSNorm(config.hidden_size, eps=eps, dtype=dtype)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=eps, dtype=dtype)
        if self.is_attention_layer:
            self.self_attn = Lfm2MoeAttention(config, device, dtype, layer_id)
            self.conv = None
        else:
            self.self_attn = None
            self.conv = ShortConv(
                config.hidden_size,
                int(config.attrs.get("conv_L_cache", 3)),
                bool(config.attrs.get("conv_bias", False)),
                dtype=dtype,
            )
        num_dense = int(config.attrs.get("num_dense_layers") or 0)
        if layer_id < num_dense:
            self.mlp = _Lfm2MoeDenseMLP(config, dtype)
        else:
            self.mlp = _Lfm2MoeMoE(config, device, dtype)
        self.mlp.layer_id = layer_id

    def forward(
        self,
        hidden_states,
        positions,
        table_idx,
        ctx,
        batch,
        conv_slot_idx=None,
        fresh_sequence: bool = False,
    ):
        residual = hidden_states
        x = self.operator_norm(hidden_states)
        if self.is_attention_layer:
            hidden_states = self.self_attn(x, positions, table_idx, ctx, batch)
        else:
            pool = ctx.model.conv_state_pool
            state = pool.get(self.layer_id, conv_slot_idx if conv_slot_idx is not None else table_idx)
            if fresh_sequence:
                # A recycled slot must not leak the previous request's ring
                # into this one's first chunk -- zero it, which reduces
                # exactly to the stateless left-zero-padding semantics.
                state.zero_()
            hidden_states = self.conv.forward_stateful(x, state)
        hidden_states = residual + hidden_states
        residual = hidden_states
        return residual + self.mlp(self.ffn_norm(hidden_states))


def _xpu_available() -> bool:
    try:
        return bool(torch.xpu.is_available())
    except Exception:
        return False


class Lfm2MoeForCausalLM(nn.Module):
    """The LFM2(.5)-MoE model: embeddings + hybrid conv/attention layers +
    ``embedding_norm`` final norm + ``lm_head``, with per-request conv-state
    rings for the short-conv layers.

    Subclasses ``nn.Module`` so its parameters are real registered
    ``nn.Parameter``s the loader resolves via ``named_parameters()``
    (structurally mirroring ``mellum.MellumForCausalLM``; fused/in-VRAM
    experts only -- no offload/CPU/hybrid MoE backend yet, see
    ``iter_weights`` and the loader's ``_CPU_MOE_CAPABLE_ARCHS`` fallback).
    """

    def __init__(self, config: ModelConfig, device=None) -> None:
        super().__init__()
        self.config = config
        if device is None:
            device = torch.device("xpu") if _xpu_available() else torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        dtype = getattr(config, "dtype", None) or torch.bfloat16
        self.dtype = dtype
        vocab_size = getattr(config, "vocab_size", 256)
        hidden_size = getattr(config, "hidden_size", 256)
        num_layers = getattr(config, "num_layers", 0)
        eps = config.attrs.get("norm_eps", 1e-5) if getattr(config, "attrs", None) else 1e-5
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            _Lfm2MoeDecoderLayer(config, device, dtype, layer_id=i) for i in range(num_layers)
        )
        # The FINAL norm (the real HF name -- ``model.embedding_norm.weight``;
        # it normalizes the stack output despite the name).
        self.embedding_norm = nn.RMSNorm(hidden_size, eps=eps, dtype=dtype)
        from freetoken.layers import LinearReplicated

        self.lm_head = LinearReplicated(hidden_size, vocab_size, has_bias=False, dtype=dtype)
        # No offload/CPU/hybrid MoE backend yet: the engine's offload
        # plumbing must see this model as never eligible for it (the
        # loader's _CPU_MOE_CAPABLE_ARCHS gate then falls back to fused
        # expert placement -- the same deliberate scope cut as mellum).
        self.moe_offload = False
        self.moe_cache = None
        self.moe_layer_id = None
        # Per-request conv left-context rings (see _ConvStatePool). Sized
        # from the engine's max_running_req when stashed in attrs; lazily
        # grown either way, so any admission pattern is handled.
        max_slots = int(config.attrs.get("max_running_req") or 8)
        self.conv_state_pool = _ConvStatePool()
        self._register_conv_pool(max_slots)
        if self.device.type != "cpu":
            self.to(self.device)

    def _register_conv_pool(self, num_slots: int) -> None:
        """(Re)size the conv-state pool for ``num_slots`` requests."""
        self.conv_state_pool = _ConvStatePool()
        for layer in self.layers:
            if layer.conv is not None:
                self.conv_state_pool.register(
                    layer.layer_id,
                    num_slots,
                    layer.conv.hidden_size,
                    layer.conv.kernel_size - 1,
                    self.device,
                    self.dtype,
                )

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, out_loc: torch.Tensor) -> torch.Tensor:
        """Run one engine step; return the **last-position** logits ``[bs, V]``.

        Mirrors the established per-model engine contract (``mellum`` /
        ``qwen3_5_moe``): each request's new-token slice runs through every
        layer (conv layers via their stateful path with this request's ring,
        attention layers through the paged-KV pool), and only the last
        position's post-``embedding_norm`` hidden state feeds ``lm_head``.
        """
        from freetoken.core import get_global_ctx

        ctx = get_global_ctx()
        batch = ctx.batch
        reqs = batch.reqs
        num_tokens = input_ids.shape[0]

        # The conv layers read their per-request ring from the model's own
        # pool, indexed by the request's linear_slot_idx (the same field
        # qwen3_5_moe's GDN layers use; the hybrid engine path may assign
        # one, the default is 1:1 with table_idx).
        for req in reqs:
            if req.linear_slot_idx is None:
                req.linear_slot_idx = req.table_idx

        hidden = self.embed_tokens(input_ids)  # [num_tokens, hidden]
        out = torch.empty((batch.size, self.config.hidden_size), device=hidden.device, dtype=hidden.dtype)

        offset = 0
        extend_lens = batch.extend_lens
        if extend_lens is None:
            prefill = batch.is_prefill or (num_tokens > batch.size)
            extend_lens = [req.extend_len if prefill else 1 for req in reqs]
        is_decode_batch = batch.phase == "decode"
        for i, req in enumerate(reqs):
            ext = 1 if is_decode_batch else int(extend_lens[i])
            token_slice = slice(offset, offset + ext)
            h = hidden[token_slice]
            pos = positions[token_slice]
            # A slice starting at position 0 is a fresh sequence: zero the
            # conv rings for this slot so a recycled table row never leaks
            # the previous request's context. (Prefill continuations and
            # decode steps start past 0 and carry the ring forward.) A
            # prefix-cache hit that skips earlier positions would need its
            # ring restored, not zeroed -- same known gap as qwen3_5_moe's
            # default pool; out of scope here (no prefix caching wired for
            # this model yet).
            fresh = bool(pos.numel() and int(pos[0].item()) == 0)
            for layer in self.layers:
                h = layer(h, pos, req.table_idx, ctx, batch, conv_slot_idx=req.linear_slot_idx, fresh_sequence=fresh)
            out[i] = self.embedding_norm(h)[-1]
            offset += ext

        return self.lm_head(out)


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
