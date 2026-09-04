"""Mellum2-12B-A2.5B (issue #227, `models-mellum-attn`) -- config parsing +
GQA/QK-norm attention + the flat top-8/64 MoE router.

Real checkpoint `JetBrains/Mellum2-12B-A2.5B-Base` (`model_type=mellum`,
`transformers.MellumConfig`, v5.15.1): GQA (32 Q / 4 KV heads, head_dim=128,
explicit -- hidden_size(2304)/num_heads(32)=72 != 128), per-head q_norm/k_norm,
64 routed experts, flat top-8 (`norm_topk_prob=True`, no `n_group` -- ungrouped,
matching this port's own `qwen3_moe` router), no shared expert.

Two things confirmed from the REAL checkpoint's own config.json (not the HF
class defaults, which differ): `layer_types` genuinely alternates 3
sliding_attention + 1 full_attention every 4 layers (NOT the class default of
all full_attention), and `mlp_layer_types` is genuinely all "sparse" (matches
the class default -- every layer is MoE, no dense layers). Sliding-attention
layers use plain RoPE; full-attention layers use YaRN-scaled RoPE with a
DIFFERENT rope_theta config than the class default alone would suggest -- two
independent per-layer-type RoPE tables, not one shared table.

Attention and router math are adapted from this port's own `qwen3_moe`
(`_Qwen3Attention`/`_Qwen3MoE`), which is the closest existing architecture
(GQA + per-head QK-norm + flat-renormalized top-k, identical shape) --
duplicated rather than imported (every model package in this port stands
alone). Critically this reuse also carries forward PR #234's real KV-cache
fix (`write_kv`'s `out_loc` must be the physical pool slot from
`ctx.page_table`, not the logical token position -- the previous
`qwen3`/`qwen3_moe` bug, invisible under a synthetic identity page table).

Standalone primitives only (`MellumAttention`, `mellum_moe_router`) -- not
yet wired into a decoder layer or the engine (that's #228's job).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

from freetoken.models.config import ModelConfig as _ModelConfig


def _yarn_inv_freq_and_scaling(
    head_dim: int, rope_theta: float, rope_params: dict, device
) -> Tuple[torch.Tensor, float]:
    """The real YaRN formula (HF ``modeling_rope_utils.py``'s
    ``_compute_yarn_parameters``) -- identical to this port's own
    ``gpt_oss._yarn_inv_freq_and_scaling``, duplicated per this port's
    "every model package stands alone" convention. Mellum's real
    checkpoint already provides ``attention_factor`` directly (no need to
    infer it from ``mscale``/``mscale_all_dim``, which it doesn't set)."""
    dim = head_dim
    base = float(rope_theta)
    factor = float(rope_params["factor"])
    original_max = float(rope_params["original_max_position_embeddings"])
    beta_fast = float(rope_params.get("beta_fast") or 32)
    beta_slow = float(rope_params.get("beta_slow") or 1)
    truncate = bool(rope_params.get("truncate", True))
    attention_factor = rope_params.get("attention_factor")
    mscale = rope_params.get("mscale")
    mscale_all_dim = rope_params.get("mscale_all_dim")

    def get_mscale(scale, m=1.0):
        if scale <= 1:
            return 1.0
        return 0.1 * m * math.log(scale) + 1.0

    if attention_factor is None:
        if mscale and mscale_all_dim:
            attention_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)
        else:
            attention_factor = get_mscale(factor)

    def find_correction_dim(num_rotations):
        return (dim * math.log(original_max / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    low = find_correction_dim(beta_fast)
    high = find_correction_dim(beta_slow)
    if truncate:
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(lo, hi, n):
        if lo == hi:
            hi += 0.001
        return torch.clamp((torch.arange(n, dtype=torch.float32) - lo) / (hi - lo), 0, 1)

    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)
    extrap_factor = 1 - linear_ramp_factor(low, high, dim // 2).to(device)
    inv_freq = inv_freq_interpolation * (1 - extrap_factor) + inv_freq_extrapolation * extrap_factor
    return inv_freq, float(attention_factor)


def parse_config(hf_config, model_path: str | None = None, **_kwargs) -> "ModelConfig":
    """Build a :class:`ModelConfig` from a real Mellum ``config.json``.

    ``rope_parameters`` (per-attention-type: ``full_attention`` uses YaRN,
    ``sliding_attention`` uses plain RoPE) and ``layer_types`` are stashed
    verbatim on ``cfg.attrs`` -- read the REAL checkpoint's own values here,
    never the HF class defaults (all-full-attention / all-plain-RoPE),
    which do not match the real shipped checkpoint.
    """
    src = hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(hf_config)
    n = int(src.get("num_hidden_layers") or 0)
    layer_types = src.get("layer_types") or (["full_attention"] * n)
    mlp_layer_types = src.get("mlp_layer_types") or (["sparse"] * n)
    rope_params = src.get("rope_parameters") or {}

    cfg = _ModelConfig(
        architectures=["MellumForCausalLM"],
        hidden_size=src.get("hidden_size"),
        vocab_size=src.get("vocab_size"),
        num_layers=n,
        num_experts=src.get("num_experts") or src.get("num_local_experts"),
        num_attention_heads=src.get("num_attention_heads"),
        num_key_value_heads=src.get("num_key_value_heads"),
        head_dim=int(src["head_dim"]) if src.get("head_dim") else None,
        intermediate_size=src.get("intermediate_size"),
        moe_intermediate_size=src.get("moe_intermediate_size"),
        num_experts_per_tok=src.get("num_experts_per_tok"),
        max_position_embeddings=src.get("max_position_embeddings"),
        tie_word_embeddings=src.get("tie_word_embeddings", False),
        hidden_act=src.get("hidden_act", "silu"),
        dtype=src.get("torch_dtype") or src.get("dtype"),
    )
    cfg.is_moe = True
    cfg.attrs["layer_types"] = layer_types
    cfg.attrs["mlp_layer_types"] = mlp_layer_types
    cfg.attrs["rope_parameters"] = rope_params
    cfg.attrs["sliding_window"] = int(src.get("sliding_window") or 0)
    cfg.attrs["norm_topk_prob"] = bool(src.get("norm_topk_prob", True))
    cfg.attrs["rms_norm_eps"] = float(src.get("rms_norm_eps", 1e-6))
    return cfg


class MellumAttention(nn.Module):
    """GQA + per-head q/k RMSNorm attention for one Mellum layer.

    ``layer_type`` ("full_attention" | "sliding_attention") picks this
    layer's own RoPE table (YaRN for full, plain for sliding -- two
    independent tables, per the real checkpoint's own per-type
    ``rope_parameters``) and its sliding-window size (0 for full
    attention). Adapted from ``qwen3_moe``'s own ``_Qwen3Attention``
    (identical GQA/QK-norm math and KV-pool contract -- see that module's
    own ``write_kv``/``out_loc`` comments, carried forward here verbatim:
    ``out_loc`` must be the PHYSICAL pool slot from ``ctx.page_table``,
    never the raw logical position, per PR #234's real fix).
    """

    def __init__(self, config: "ModelConfig", device, dtype, layer_id: int, layer_type: str) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.layer_type = layer_type
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads or config.num_attention_heads
        self.head_dim = config.head_dim or (config.hidden_size // self.num_heads)
        self.sliding_window = config.attrs.get("sliding_window", 0) if layer_type == "sliding_attention" else 0

        from freetoken.layers import LinearOProj, LinearReplicated

        self.q_proj = LinearReplicated(config.hidden_size, self.num_heads * self.head_dim, has_bias=False, dtype=dtype)
        self.k_proj = LinearReplicated(config.hidden_size, self.num_kv_heads * self.head_dim, has_bias=False, dtype=dtype)
        self.v_proj = LinearReplicated(config.hidden_size, self.num_kv_heads * self.head_dim, has_bias=False, dtype=dtype)
        self.o_proj = LinearOProj(self.num_heads * self.head_dim, config.hidden_size, has_bias=False, dtype=dtype)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.attrs.get("rms_norm_eps", 1e-6), dtype=dtype)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.attrs.get("rms_norm_eps", 1e-6), dtype=dtype)

        rope_params = (config.attrs.get("rope_parameters") or {}).get(layer_type, {})
        theta = float(rope_params.get("rope_theta") or 10000.0)
        rope_type = str(rope_params.get("rope_type") or "default").lower()
        if rope_type == "yarn":
            inv_freq, self.attention_scaling = _yarn_inv_freq_and_scaling(
                self.head_dim, theta, rope_params, device
            )
        else:
            inv_freq = 1.0 / (
                theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim)
            )
            self.attention_scaling = 1.0
        self.register_buffer("inv_freq", inv_freq)

    def _rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # Half-split (rotate_half), matching the real HF convention (see
        # qwen3/qwen3_moe's own docstrings on this exact bug class).
        freqs = torch.outer(pos.to(torch.float32), self.inv_freq)  # [N, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [N, D]
        cos = (emb.cos() * self.attention_scaling)[None, :, :]
        sin = (emb.sin() * self.attention_scaling)[None, :, :]
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
        q = self._rope(self.q_norm(q), positions)
        k = self._rope(self.k_norm(k), positions)
        # write_kv's out_loc is the PHYSICAL pool slot (ctx.page_table),
        # never the raw logical position -- PR #234's real fix, carried
        # forward here from day one rather than reintroducing the bug.
        out_loc = ctx.page_table[table_idx, positions.long()]
        ctx.kv_cache.write_kv(k, v, out_loc, self.layer_id)
        from freetoken.attention.base import AttentionSpec

        attn_spec = AttentionSpec(sliding_window=self.sliding_window or None)
        out = ctx.attn_backend.forward(q, k, v, self.layer_id, batch, attn_spec=attn_spec, table_idx=table_idx)
        return self.o_proj(out.transpose(0, 1).contiguous().reshape(bsz, -1))


def mellum_moe_router(hidden_states: torch.Tensor, gate_weight: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flat top-k router, no shared expert, always renormalized (matches
    the real checkpoint's ``norm_topk_prob=True`` and this port's own
    ``qwen3_moe`` router math exactly -- confirmed direct reuse).

    ``hidden_states`` is ``[T, hidden]``, ``gate_weight`` is
    ``[num_experts, hidden]`` (a plain linear, no bias). Returns
    ``(top_w [T, top_k], top_idx [T, top_k])``.
    """
    routing = F.linear(hidden_states, gate_weight)  # [T, num_experts]
    gate_log = F.softmax(routing, dim=-1)
    top_w, top_idx = torch.topk(gate_log, top_k, dim=-1)
    top_w = (top_w / top_w.sum(dim=-1, keepdim=True)).to(hidden_states.dtype)
    return top_w, top_idx


__all__ = ["MellumAttention", "mellum_moe_router", "parse_config"]


# --------------------------------------------------------------------------- #
# Not yet implemented: full model wiring (#228).
# --------------------------------------------------------------------------- #

from freetoken._stub import unimplemented


def iter_weights(*args, **kwargs):
    unimplemented("iter_weights", "models-mellum-e2e")
