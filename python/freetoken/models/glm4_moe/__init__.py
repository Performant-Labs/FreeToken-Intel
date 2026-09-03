"""GLM-4.7 (``Glm4MoeForCausalLM``, model_type ``glm4_moe``) -- Intel Arc Pro
B70 port.

Upstream NVIDIA path: python/freetoken/models/glm4_moe/
Fill in: GitHub issue `models-glm` (see docs/architecture.md).

Real forward-pass math grounded directly against HF transformers'
``src/transformers/models/glm4_moe/modeling_glm4_moe.py`` (fetched and read
this session, not guessed) -- two features new to this port, absent from
every Qwen3 family model already here:

* **Grouped, sigmoid, bias-corrected top-k MoE routing**
  (``Glm4MoeTopkRouter``, DeepSeek-V3-style): ``sigmoid(router_logits)`` (not
  softmax), a learned per-expert ``e_score_correction_bias`` buffer added
  ONLY for the group/expert *selection* (never applied to the combine
  weights, which are gathered from the raw unbiased sigmoid scores), a
  group-then-expert two-stage top-k (degenerate to plain top-k when
  ``n_group == topk_group == 1``, still true for GLM-4.7's own config), an
  optional post-hoc renormalization (``norm_topk_prob``), and a final
  ``routed_scaling_factor`` multiply applied LAST. The always-on shared
  expert combines by plain unweighted sum (no gate), unlike Qwen3.5/3.6's
  sigmoid-gated shared expert.
* **Leading dense layers** (``first_k_dense_replace``): the first N layers
  run a plain SwiGLU MLP, never a router -- a real architectural feature
  the Qwen3 family ports here never needed (their own ``first_k_dense_
  replace`` is always 0).

Attention (biased q/k/v projections, QK-norm applied BEFORE partial RoPE
over the full head_dim) reuses the exact partial-RoPE math already proven
in ``qwen3_5_moe``'s ``_Qwen35Attention`` (``rotate_half``-style, half-split
rotation, NOT interleaved pairs -- confirmed to match GLM-4.7's real
``apply_rotary_pos_emb`` exactly), duplicated rather than imported (every
model package in this port stands alone, matching the existing
convention).

Deliberate scope cut, documented and left as real follow-up work: this
port only builds the in-VRAM (fused) forward path -- no offload/CPU/hybrid
backend (ADR 0002) for GLM-4.7's own real 92-layer / 160-expert scale,
which obviously cannot run in-VRAM on a 32 GB B70. Wiring GLM-4.7 into the
existing offload machinery (already generic across every other MoE model
this port has) is separable, mechanical follow-up, not a new design
problem -- see loader.py's own `_place_expert_weights_any`/
`_attach_offload_cache`, both already architecture-agnostic.
``num_nextn_predict_layers`` (multi-token-prediction) is a separate,
optional training head untouched by ``Glm4MoeForCausalLM``'s own real
forward -- confirmed absent from the real modeling file entirely -- so it
is not modeled here either.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from freetoken.models.config import ModelConfig
from freetoken.models.weight import iter_safetensors
from freetoken.utils import cached_load_hf_config

# --------------------------------------------------------------------------- #
# Checkpoint side (loader contract, from `#17`)
# --------------------------------------------------------------------------- #


def parse_config(hf_config, model_path: str | None = None, **_kwargs) -> ModelConfig:
    """Build a :class:`ModelConfig` from a real HF ``Glm4MoeConfig``.

    ``**_kwargs`` absorbs the MoE-only kwargs (``use_offload_moe`` etc.)
    ``load_model`` passes to every architecture's ``parse_config`` when
    re-parsing for a resolved MoE backend -- this port doesn't build those
    backends for GLM-4.7 yet (see this module's own docstring), so they are
    accepted and ignored rather than crashing construction.
    """
    del model_path  # no extended-head checkpoint probe needed: head_dim is always explicit
    src = hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(hf_config)
    cfg = ModelConfig(
        architectures=["Glm4MoeForCausalLM"],
        hidden_size=src.get("hidden_size"),
        vocab_size=src.get("vocab_size"),
        num_layers=src.get("num_hidden_layers"),
        num_experts=src.get("n_routed_experts"),
        num_attention_heads=src.get("num_attention_heads"),
        num_key_value_heads=src.get("num_key_value_heads"),
        head_dim=src.get("head_dim"),
        intermediate_size=src.get("intermediate_size"),
        moe_intermediate_size=src.get("moe_intermediate_size"),
        num_experts_per_tok=src.get("num_experts_per_tok"),
        first_k_dense_replace=src.get("first_k_dense_replace") or 0,
        max_position_embeddings=src.get("max_position_embeddings"),
        tie_word_embeddings=src.get("tie_word_embeddings", False),
        rope_theta=src.get("rope_theta"),
        rope_scaling=src.get("rope_scaling"),
        hidden_act=src.get("hidden_act", "silu"),
        dtype=src.get("torch_dtype") or src.get("dtype"),
    )
    cfg.is_moe = True
    # GLM-4.7-specific fields this port's shared ModelConfig has no named
    # field for -- stashed in .attrs (the existing catch-all every model
    # package already uses for its own architecture-specific extras, e.g.
    # qwen3_5_moe's text_config/linear_* fields).
    cfg.attrs["attention_bias"] = bool(src.get("attention_bias", False))
    cfg.attrs["partial_rotary_factor"] = float(src.get("partial_rotary_factor") or 1.0)
    cfg.attrs["use_qk_norm"] = bool(src.get("use_qk_norm", False))
    cfg.attrs["n_group"] = int(src.get("n_group") or 1)
    cfg.attrs["topk_group"] = int(src.get("topk_group") or 1)
    cfg.attrs["n_shared_experts"] = int(src.get("n_shared_experts") or 0)
    cfg.attrs["routed_scaling_factor"] = float(src.get("routed_scaling_factor") or 1.0)
    cfg.attrs["norm_topk_prob"] = bool(src.get("norm_topk_prob", True))
    cfg.attrs["rms_norm_eps"] = float(src.get("rms_norm_eps") or 1e-5)
    return cfg


def iter_weights(model_path: str, device: torch.device, *, include_moe_experts: bool = True, include_non_moe: bool = True):
    """Yield the checkpoint's tensors, each on its destination device.

    Same dense/expert split as ``qwen3_moe``'s own ``iter_weights``: expert
    tensors (``...mlp.experts...``) stay on host memory, everything else
    (including a leading dense layer's plain ``mlp.{gate,up,down}_proj``,
    the router's ``mlp.gate.weight``, and its
    ``mlp.gate.e_score_correction_bias`` buffer -- none of which contain
    ``.experts.``) goes to ``device``. A ``tie_word_embeddings: true``
    checkpoint's missing ``lm_head.weight`` is synthesized from
    ``embed_tokens.weight``, mirroring every other model port's own fix for
    this real failure mode (an unfilled lm_head silently zeros every logit).
    """
    embed_tokens_weight = None
    saw_lm_head = False
    for name, tensor in iter_safetensors(model_path, device):
        is_expert = ".experts." in name
        if is_expert and not include_moe_experts:
            continue
        if not is_expert and not include_non_moe:
            continue
        dest = torch.device("cpu") if is_expert else device
        placed = tensor.to(dest)
        if name == "model.embed_tokens.weight":
            embed_tokens_weight = placed
        elif name == "lm_head.weight":
            saw_lm_head = True
        yield name, placed

    if include_non_moe and not saw_lm_head and embed_tokens_weight is not None:
        hf_config = cached_load_hf_config(model_path)
        if bool(getattr(hf_config, "tie_word_embeddings", False)):
            yield "lm_head.weight", embed_tokens_weight


# --------------------------------------------------------------------------- #
# Forward side (the real model the engine runs, `#14`)
# --------------------------------------------------------------------------- #


def _xpu_available() -> bool:
    try:
        return bool(torch.xpu.is_available())
    except Exception:
        return False


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Half-split rotation (``cat(-x2, x1)`` over the two halves), NOT
    interleaved pairs -- GLM-4.7's real ``rotate_half`` (confirmed against
    the real modeling file), same convention qwen3_5_moe's partial-RoPE
    already uses."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    """Partial-RoPE: rotate only the first ``cos.shape[-1]`` head dims,
    passing the rest through unchanged. Identical contract to
    qwen3_5_moe's own ``_apply_rotary_pos_emb`` (duplicated, not imported --
    see this module's own docstring on why each model package stands
    alone)."""
    out_dtype = q.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    q_embed = torch.cat([q_embed, q_pass.to(q_embed.dtype)], dim=-1).to(out_dtype)
    k_embed = torch.cat([k_embed, k_pass.to(k_embed.dtype)], dim=-1).to(out_dtype)
    return q_embed, k_embed


def _rope_for_positions(inv_freq: torch.Tensor, positions: torch.Tensor, rotary_dim: int) -> tuple:
    del rotary_dim  # kept for signature parity with qwen3_5_moe's own helper
    freqs = torch.outer(positions.to(torch.float32), inv_freq)  # [N, rotary_dim//2]
    freqs_full = torch.cat((freqs, freqs), dim=-1)  # [N, rotary_dim]
    cos = freqs_full.cos()[None, :, :]
    sin = freqs_full.sin()[None, :, :]
    return cos, sin


class _Glm4Attention(nn.Module):
    """GLM-4.7 grouped-query attention: biased q/k/v/o projections
    (``config.attrs["attention_bias"]``), QK-norm over the FULL head_dim
    applied BEFORE partial RoPE (confirmed order against the real modeling
    file -- Qwen3's own attention applies norm-then-rope too, but with FULL
    rope, not partial), KV-pool driven exactly like every other attention
    block in this port.
    """

    def __init__(self, config: ModelConfig, device, dtype, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads or config.num_attention_heads
        self.head_dim = config.head_dim or (config.hidden_size // self.num_heads)
        self.rotary_dim = int(self.head_dim * config.attrs.get("partial_rotary_factor", 1.0))
        bias = bool(config.attrs.get("attention_bias", False))
        use_qk_norm = bool(config.attrs.get("use_qk_norm", False))
        eps = config.attrs.get("rms_norm_eps", 1e-5)

        from freetoken.layers import LinearOProj, LinearReplicated

        self.q_proj = LinearReplicated(config.hidden_size, self.num_heads * self.head_dim, has_bias=bias, dtype=dtype)
        self.k_proj = LinearReplicated(config.hidden_size, self.num_kv_heads * self.head_dim, has_bias=bias, dtype=dtype)
        self.v_proj = LinearReplicated(config.hidden_size, self.num_kv_heads * self.head_dim, has_bias=bias, dtype=dtype)
        # o_proj carries no bias even when q/k/v do (confirmed against the
        # real modeling file).
        self.o_proj = LinearOProj(self.num_heads * self.head_dim, config.hidden_size, has_bias=False, dtype=dtype)
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=eps, dtype=dtype)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=eps, dtype=dtype)
        theta = config.rope_theta or 10000.0
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32, device=device) / self.rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, hidden_states, positions, table_idx, ctx, batch):
        bsz, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(hidden_states).view(bsz, self.num_kv_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(hidden_states).view(bsz, self.num_kv_heads, self.head_dim).transpose(0, 1)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        cos, sin = _rope_for_positions(self.inv_freq, positions, self.rotary_dim)
        cos = cos.reshape(bsz, -1)
        sin = sin.reshape(bsz, -1)
        q, k = _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=0)
        ctx.kv_cache.write_kv(k, v, positions, self.layer_id)
        out = ctx.attn_backend.forward(q, k, v, self.layer_id, batch, table_idx=table_idx)
        return self.o_proj(out.transpose(0, 1).contiguous().reshape(bsz, -1))


class _Glm4MLP(nn.Module):
    """Plain SwiGLU MLP -- the leading (``first_k_dense_replace``) dense
    layers' only feed-forward block, and (with a scaled intermediate size)
    the MoE block's always-on shared expert."""

    def __init__(self, hidden_size: int, intermediate_size: int, dtype) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _Glm4TopkRouter(nn.Module):
    """DeepSeek-V3-style grouped, sigmoid, bias-corrected top-k router
    (``Glm4MoeTopkRouter``, real math -- see this module's own docstring).
    ``weight``/``e_score_correction_bias`` are named to match the real
    checkpoint's ``mlp.gate.weight``/``mlp.gate.e_score_correction_bias``
    keys exactly (this class instance IS the model's ``mlp.gate``)."""

    def __init__(self, config: ModelConfig, dtype) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.n_group = config.attrs.get("n_group", 1)
        self.topk_group = config.attrs.get("topk_group", 1)
        self.norm_topk_prob = config.attrs.get("norm_topk_prob", True)
        self.routed_scaling_factor = config.attrs.get("routed_scaling_factor", 1.0)
        self.weight = nn.Parameter(torch.empty(self.num_experts, config.hidden_size, dtype=dtype))
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``hidden_states`` is ``[T, H]``. Returns ``(topk_indices [T, k],
        topk_weights [T, k])`` -- weights already renormalized and scaled."""
        router_logits = F.linear(hidden_states.float(), self.weight.float())  # [T, E]
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        if self.n_group > 1:
            T = scores_for_choice.shape[0]
            group_scores = (
                scores_for_choice.view(T, self.n_group, self.num_experts // self.n_group)
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )  # [T, n_group]
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1)[1]  # [T, topk_group]
            group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1.0)  # [T, n_group]
            expert_mask = (
                group_mask.unsqueeze(-1)
                .expand(T, self.n_group, self.num_experts // self.n_group)
                .reshape(T, self.num_experts)
            )
            scores_for_choice = scores_for_choice.masked_fill(expert_mask == 0, float("-inf"))
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1)[1]  # [T, k], NOT sorted
        topk_weights = scores.gather(1, topk_indices)  # gathered from the RAW (uncorrected) sigmoid scores
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


class _Glm4MoE(nn.Module):
    """A GLM-4.7 MoE block: the grouped-topk router (:class:`_Glm4TopkRouter`)
    + N routed experts + an always-on shared expert combined by plain
    unweighted sum (no gate -- unlike Qwen3.5/3.6's sigmoid-gated shared
    expert). In-VRAM only (see this module's own docstring on the offload
    scope cut) -- every expert is a real device-resident module."""

    def __init__(self, config: ModelConfig, device, dtype, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.gate = _Glm4TopkRouter(config, dtype)
        self.experts = nn.ModuleList(
            _Glm4MLP(config.hidden_size, config.moe_intermediate_size, dtype) for _ in range(config.num_experts)
        )
        n_shared = config.attrs.get("n_shared_experts", 0)
        self.shared_experts = (
            _Glm4MLP(config.hidden_size, config.moe_intermediate_size * n_shared, dtype) if n_shared > 0 else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        in_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, in_shape[-1])  # [n_tok, H]
        topk_indices, topk_weights = self.gate(flat)
        topk_weights = topk_weights.to(flat.dtype)

        # Host-built integer indices (never nonzero()/boolean masking on the
        # device -- see qwen3_moe's own _forward_inram for the real bug this
        # avoids: an XPU bool-tensor nonzero() silently returns empty).
        top_idx_cpu = topk_indices.to("cpu")
        out = torch.zeros_like(flat)
        for e in range(len(self.experts)):
            for slot in range(topk_indices.shape[1]):
                sel_cpu = top_idx_cpu[:, slot] == e
                if not bool(sel_cpu.any()):
                    continue
                idx = sel_cpu.nonzero(as_tuple=True)[0].to(flat.device)
                w = topk_weights.index_select(0, idx)[:, slot, None]
                y = self.experts[e](flat.index_select(0, idx))
                out.index_add_(0, idx, w * y)

        if self.shared_experts is not None:
            out = out + self.shared_experts(flat)
        return out.view(in_shape)


class _Glm4DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, device, dtype, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        eps = config.attrs.get("rms_norm_eps", 1e-5)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=eps, dtype=dtype)
        self.self_attn = _Glm4Attention(config, device, dtype, layer_id)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=eps, dtype=dtype)
        is_dense = layer_id < int(config.first_k_dense_replace or 0)
        self.mlp = (
            _Glm4MLP(config.hidden_size, config.intermediate_size, dtype)
            if is_dense
            else _Glm4MoE(config, device, dtype, layer_id)
        )

    def forward(self, hidden_states, positions, table_idx, ctx, batch):
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), positions, table_idx, ctx, batch)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class Glm4MoeForCausalLM(nn.Module):
    """The GLM-4.7 model: real forward pass for the Intel engine loop
    (``#14``). Subclasses ``nn.Module`` directly (not the torch-free
    ``BaseLLMModel`` stub) so its parameters are real registered
    ``nn.Parameter``s -- the loader resolves ``named_parameters()``/
    ``named_buffers()`` to fill weights."""

    def __init__(self, config, device=None) -> None:
        super().__init__()
        self.config = config
        if device is None:
            device = torch.device("xpu") if _xpu_available() else torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        dtype = getattr(config, "dtype", None) or torch.bfloat16
        vocab_size = getattr(config, "vocab_size", 256)
        hidden_size = getattr(config, "hidden_size", 256)
        num_layers = getattr(config, "num_layers", 0)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            _Glm4DecoderLayer(config, device, dtype, layer_id=i) for i in range(num_layers)
        )
        eps = getattr(config, "attrs", {}).get("rms_norm_eps", 1e-5)
        self.norm = nn.RMSNorm(hidden_size, eps=eps, dtype=dtype)
        from freetoken.layers import LinearReplicated

        self.lm_head = LinearReplicated(hidden_size, vocab_size, has_bias=False, dtype=dtype)
        # No offload machinery yet (see this module's own docstring) -- the
        # engine's guarded getattr reads see the defaults it already treats
        # as "nothing to offload".
        self.moe_offload = False
        self.moe_cache = None
        if self.device.type != "cpu":
            self.to(self.device)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, out_loc: torch.Tensor) -> torch.Tensor:
        from freetoken.core import get_global_ctx

        ctx = get_global_ctx()
        batch = ctx.batch
        reqs = batch.reqs
        num_tokens = input_ids.shape[0]

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
            for layer in self.layers:
                h = layer(h, positions[token_slice], req.table_idx, ctx, batch)
            out[i] = self.norm(h)[-1]
            offset += ext

        return self.lm_head(out)


__all__ = ["parse_config", "iter_weights", "Glm4MoeForCausalLM"]
