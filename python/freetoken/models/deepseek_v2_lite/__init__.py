"""DeepSeek-Coder-V2-Lite -- Intel Arc Pro B70 port.

**Current scope: Multi-head Latent Attention (MLA) + config parsing only**
(issue `models-dsv2lite-mla`, #217, first child of the DeepSeek-Coder-V2-Lite
epic #216). The real MoE router (`models-dsv2lite-moe`, #218) and full
engine wiring against a real downloaded checkpoint (`models-dsv2lite-e2e`,
#219) are deliberate, separate follow-up issues -- every layer here runs a
plain dense MLP (same scope-cut shape as `deepseek_v4`'s own #190), so this
module's own accept bar (a real MLA forward, end to end, numerically sound)
is provable in isolation.

Real DeepSeek-Coder-V2-Lite checkpoint config (pulled directly from HF,
``deepseek-ai/DeepSeek-Coder-V2-Lite-Base``, not guessed):
``q_lora_rank: null`` (unlike DeepSeek-V3/V4, which always set this --
V2-Lite genuinely has no query low-rank compression at this scale, so this
module's own ``_DeepseekV2LiteMLA`` must treat the plain ``q_proj`` path as
the REAL common case here, not an edge case), ``kv_lora_rank: 512``,
``qk_rope_head_dim: 64``, ``qk_nope_head_dim: 128``, ``v_head_dim: 128``,
and (unlike V4) real YaRN rope scaling
(``rope_scaling: {"type": "yarn", "factor": 40, "beta_fast": 32,
"beta_slow": 1, "mscale": 0.707, "mscale_all_dim": 0.707,
"original_max_position_embeddings": 4096}``).

Two real, confirmed architectural differences from `deepseek_v4`'s own MLA
(#190), both grounded directly against the real, installed
``transformers.models.deepseek_v2.modeling_deepseek_v2`` (v5.15.1) rather
than assumed to match V3/V4:

1. **RoPE convention**: DeepSeek-V2's real ``apply_rotary_emb`` uses the
   INTERLEAVED-PAIRS / complex-multiplication convention
   (``torch.polar``/``view_as_complex``, adjacent-pair rotation), NOT the
   half-split ``rotate_half`` (NeoX-style) convention `deepseek_v4`'s own
   MLA uses. Implemented here as the real-valued equivalent of that complex
   multiplication (``_rotate_interleaved`` below) to avoid a complex-dtype
   round trip -- numerically identical, confirmed against the real
   ``apply_rotary_emb`` in this module's own tests.
2. **YaRN rope scaling**: V2-Lite's real checkpoint sets a non-default
   ``rope_type``, unlike every #190-only V4 checkpoint this port has
   exercised so far (which only ever used plain, unscaled rope). Two
   SEPARATE corrections apply, both grounded directly against
   ``transformers.modeling_rope_utils._compute_yarn_parameters`` and
   ``modeling_deepseek_v2.yarn_apply_mscale``:

   * the inverse frequencies themselves are NTK-ramp-interpolated between an
     "extrapolation" (unscaled) and "interpolation" (``factor``-scaled)
     branch, blended per-dimension by ``beta_fast``/``beta_slow`` correction
     bounds (``_yarn_inv_freq`` below) -- this changes what gets rotated;
   * the resulting cos/sin are then scaled by an `attention_factor` derived
     from ``mscale``/``mscale_all_dim`` (``_yarn_attention_factor`` below);
   * the softmax attention SCALE itself (the ``1/sqrt(d)`` term) gets a
     SEPARATE, independently-computed mscale correction
     (``_yarn_softmax_scale`` below, mirrors ``yarn_apply_mscale`` exactly)
     -- this is not the same number as the cos/sin scaling above, despite
     both being called "mscale" corrections; conflating them was the first
     mistake caught while grounding this against the real reference.

Everything else (compression, caching shape, decompression, the KV-pool
``num_key_value_heads=1``/``head_dim=kv_lora_rank+qk_rope_head_dim`` reuse,
the no-DSA scope) is identical in spirit to `deepseek_v4`'s own MLA -- see
that module's docstring for the fuller memory-shape rationale, not
repeated here.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from freetoken.models.config import ModelConfig
from freetoken.models.weight import iter_safetensors
from freetoken.utils import cached_load_hf_config

# --------------------------------------------------------------------------- #
# Checkpoint side (loader contract, from `#17`)
# --------------------------------------------------------------------------- #


def _raw_checkpoint_json(model_path: str) -> dict:
    import json
    import os

    path = os.path.join(model_path, "config.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def parse_config(hf_config, model_path: str | None = None, **_kwargs) -> ModelConfig:
    """Build a :class:`ModelConfig` from a real HF ``DeepseekV2Config``-shaped
    config. See this module's own docstring for the real-math grounding.

    ``config.num_key_value_heads``/``config.head_dim`` are deliberately set
    to the KV-POOL's storage shape (``1`` / ``kv_lora_rank +
    qk_rope_head_dim``), NOT the real attention head count -- identical
    reasoning to `deepseek_v4`'s own MLA (see that module's docstring). The
    real head count lives in ``config.num_attention_heads``.
    """
    raw = hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(hf_config)
    # q_lora_rank specifically: the installed transformers' real
    # DeepseekV2Config class has a non-None default (1536) that leaks
    # through both getattr(hf_config, ...) and to_dict() even for a
    # checkpoint whose own config.json sets it to null -- confirmed
    # directly against the real downloaded DeepSeek-Coder-V2-Lite-Base
    # checkpoint (`q_lora_rank: null`). Same defensive pattern established
    # for DeepseekV4Config's index_topk/n_routed_experts and
    # GlmMoeDsaConfig earlier this session. Without a model_path (a
    # mock/unit-test hf_config with no backing file) fall back to `raw`,
    # which is not polluted for a hand-built mock object.
    file_raw = _raw_checkpoint_json(model_path) if model_path else raw

    def field(name, default=None):
        val = getattr(hf_config, name, None)
        return val if val is not None else raw.get(name, default)

    src = {k: field(k) for k in (
        "hidden_size", "vocab_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim",
        "v_head_dim", "attention_bias", "intermediate_size",
        "max_position_embeddings", "tie_word_embeddings", "rope_theta",
        "rope_scaling", "hidden_act", "torch_dtype", "dtype", "rms_norm_eps",
        "n_routed_experts", "moe_intermediate_size", "num_experts_per_tok",
        "first_k_dense_replace", "n_group", "topk_group",
        "routed_scaling_factor", "n_shared_experts", "norm_topk_prob",
        "topk_method",
    )}
    kv_lora_rank = int(src.get("kv_lora_rank") or 0)
    qk_rope_head_dim = int(src.get("qk_rope_head_dim") or 0)
    # q_lora_rank: read from file_raw (the real checkpoint file, or the mock
    # dict for unit tests), not `field()` -- see the docstring above.
    q_lora_rank_raw = file_raw.get("q_lora_rank")
    is_moe = bool(file_raw.get("n_routed_experts"))
    cfg = ModelConfig(
        architectures=["DeepseekV2ForCausalLM"],
        hidden_size=src.get("hidden_size"),
        vocab_size=src.get("vocab_size"),
        num_layers=src.get("num_hidden_layers"),
        num_attention_heads=src.get("num_attention_heads"),
        num_key_value_heads=1,  # MLA: one shared compressed latent, not per-head K/V
        head_dim=kv_lora_rank + qk_rope_head_dim,  # the pool's real per-token row width
        intermediate_size=src.get("intermediate_size"),
        moe_intermediate_size=int(src["moe_intermediate_size"]) if is_moe and src.get("moe_intermediate_size") else None,
        num_experts=int(src["n_routed_experts"]) if is_moe else None,
        num_experts_per_tok=int(src.get("num_experts_per_tok") or 0) if is_moe else None,
        first_k_dense_replace=int(src.get("first_k_dense_replace") or 0),
        is_moe=is_moe,
        max_position_embeddings=src.get("max_position_embeddings"),
        tie_word_embeddings=src.get("tie_word_embeddings", False),
        kv_lora_rank=kv_lora_rank or None,
        q_lora_rank=int(q_lora_rank_raw) if q_lora_rank_raw else None,
        rope_theta=src.get("rope_theta"),
        rope_scaling=src.get("rope_scaling"),
        hidden_act=src.get("hidden_act", "silu"),
        dtype=src.get("torch_dtype") or src.get("dtype"),
    )
    cfg.attrs["qk_rope_head_dim"] = qk_rope_head_dim
    cfg.attrs["qk_nope_head_dim"] = int(src.get("qk_nope_head_dim") or 0)
    cfg.attrs["v_head_dim"] = int(src.get("v_head_dim") or 0)
    cfg.attrs["attention_bias"] = bool(src.get("attention_bias", False))
    cfg.attrs["rms_norm_eps"] = float(src.get("rms_norm_eps") or 1e-6)
    # Real MoE router (issue #218) -- not implemented here, only the fields
    # a future router needs are parsed and stashed, so #218 does not need
    # to touch parse_config again.
    if is_moe:
        cfg.attrs["n_group"] = int(src.get("n_group") or 1)
        cfg.attrs["topk_group"] = int(src.get("topk_group") or 1)
        cfg.attrs["routed_scaling_factor"] = float(src.get("routed_scaling_factor") or 1.0)
        cfg.attrs["n_shared_experts"] = int(src.get("n_shared_experts") or 0)
        cfg.attrs["norm_topk_prob"] = bool(src.get("norm_topk_prob", False))
        cfg.attrs["topk_method"] = src.get("topk_method") or "greedy"
    return cfg


def iter_weights(model_path: str, device: torch.device, *, include_moe_experts: bool = True, include_non_moe: bool = True):
    """Yield the checkpoint's tensors, each on its destination device.

    Same dense/expert split as `deepseek_v4`'s own ``iter_weights``: expert
    tensors (``...mlp.experts...``) stay on host memory, everything else
    goes to ``device``. This issue's own scope (#217) never builds a real
    MoE block, so a dense-only checkpoint (or the leading dense layers of a
    real MoE checkpoint) has no ``.experts.`` tensors, a no-op split.
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

    if not saw_lm_head and embed_tokens_weight is not None:
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


def apply_interleaved_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Real-valued equivalent of DeepSeek-V2's complex-multiplication RoPE
    (``torch.view_as_complex``/``torch.polar`` in the real reference) --
    adjacent-PAIR rotation applied to ``x[..., D]`` given ``cos``/``sin``
    (each ``[..., D/2]``, one angle per pair), NOT the half-split
    ``rotate_half`` convention `deepseek_v4`'s own MLA uses. Per pair
    ``(x_{2i}, x_{2i+1})``::

        out_{2i}   = x_{2i}   * cos_i - x_{2i+1} * sin_i
        out_{2i+1} = x_{2i+1} * cos_i + x_{2i}   * sin_i

    which is exactly ``(x_{2i} + i*x_{2i+1}) * (cos_i + i*sin_i)`` expanded
    -- the same complex multiplication the real reference performs via
    ``torch.polar``/``view_as_complex``, confirmed numerically identical in
    this module's own tests.
    """
    x1 = x[..., 0::2].to(torch.float32)
    x2 = x[..., 1::2].to(torch.float32)
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    out = torch.empty_like(x, dtype=torch.float32)
    out[..., 0::2] = out1
    out[..., 1::2] = out2
    return out.to(x.dtype)


def _yarn_get_mscale(scale: float, mscale: float = 1.0) -> float:
    """``transformers.modeling_deepseek_v2.yarn_get_mscale`` / the inline
    ``get_mscale`` in ``_compute_yarn_parameters`` -- identical formula,
    ported verbatim (not guessed)."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_rope_params(
    rotary_dim: int,
    base: float,
    max_position_embeddings: int,
    rope_scaling: dict,
) -> tuple[torch.Tensor, float]:
    """Real YaRN inverse-frequency + attention_factor computation, ported
    verbatim from ``transformers.modeling_rope_utils._compute_yarn_parameters``
    (v5.15.1) -- NOT the separate softmax-scale mscale correction (see
    :func:`yarn_softmax_scale` for that one, this module's own docstring
    explains why they are different numbers).

    Returns ``(inv_freq [rotary_dim/2], attention_factor)``.
    """
    factor = rope_scaling.get("factor")
    if factor is None:
        original = rope_scaling["original_max_position_embeddings"]
        factor = max_position_embeddings / original
    attention_factor = rope_scaling.get("attention_factor")
    mscale = rope_scaling.get("mscale")
    mscale_all_dim = rope_scaling.get("mscale_all_dim")
    original_max_position_embeddings = rope_scaling.get(
        "original_max_position_embeddings", max_position_embeddings
    )
    if attention_factor is None:
        if mscale and mscale_all_dim:
            attention_factor = _yarn_get_mscale(factor, mscale) / _yarn_get_mscale(factor, mscale_all_dim)
        else:
            attention_factor = _yarn_get_mscale(factor)

    beta_fast = rope_scaling.get("beta_fast") or 32
    beta_slow = rope_scaling.get("beta_slow") or 1
    truncate = rope_scaling.get("truncate", True)

    def find_correction_dim(num_rotations: float) -> float:
        return (rotary_dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    low = find_correction_dim(beta_fast)
    high = find_correction_dim(beta_slow)
    if truncate:
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, rotary_dim - 1)

    def linear_ramp_factor(lo: float, hi: float, n: int) -> torch.Tensor:
        if lo == hi:
            hi += 0.001
        linear = (torch.arange(n, dtype=torch.float32) - lo) / (hi - lo)
        return torch.clamp(linear, 0, 1)

    pos_freqs = base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)
    inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, rotary_dim // 2)
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )
    return inv_freq, float(attention_factor)


def yarn_softmax_scale(base_scale: float, rope_scaling: dict | None) -> float:
    """The SEPARATE softmax-scale mscale correction, ported verbatim from
    ``transformers.modeling_deepseek_v2.yarn_apply_mscale`` -- a DIFFERENT
    number from :func:`yarn_rope_params`'s own ``attention_factor`` (see
    module docstring). No-op (returns ``base_scale`` unchanged) for
    ``rope_type in (None, "default")`` or when ``mscale_all_dim`` is unset."""
    if not rope_scaling or rope_scaling.get("rope_type", rope_scaling.get("type", "default")) in (None, "default"):
        return base_scale
    mscale_all_dim = rope_scaling.get("mscale_all_dim", 0)
    factor = rope_scaling.get("factor")
    if factor and mscale_all_dim:
        mscale = _yarn_get_mscale(factor, mscale_all_dim)
        return base_scale * mscale * mscale
    return base_scale


class _DeepseekV2LiteMLA(nn.Module):
    """Multi-head Latent Attention for DeepSeek-Coder-V2-Lite (issue
    `models-dsv2lite-mla`, #217). See this module's own docstring for the
    full real-math grounding, and `deepseek_v4`'s own MLA docstring for the
    shared compression/cache/decompression rationale not repeated here."""

    def __init__(self, config: ModelConfig, device, dtype, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.num_heads = config.num_attention_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.attrs["qk_rope_head_dim"]
        self.qk_nope_head_dim = config.attrs["qk_nope_head_dim"]
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.attrs["v_head_dim"]
        self.q_lora_rank = config.q_lora_rank
        bias = bool(config.attrs.get("attention_bias", False))
        eps = config.attrs.get("rms_norm_eps", 1e-6)
        hidden = config.hidden_size

        if self.q_lora_rank:
            self.q_a_proj = nn.Linear(hidden, self.q_lora_rank, bias=bias, dtype=dtype)
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=eps, dtype=dtype)
            self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False, dtype=dtype)
        else:
            # The real DeepSeek-Coder-V2-Lite-Base checkpoint's own case
            # (`q_lora_rank: null`) -- no query low-rank compression at all.
            self.q_proj = nn.Linear(hidden, self.num_heads * self.qk_head_dim, bias=False, dtype=dtype)
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden, self.kv_lora_rank + self.qk_rope_head_dim, bias=bias, dtype=dtype
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=eps, dtype=dtype)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False, dtype=dtype
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, hidden, bias=bias, dtype=dtype)

        theta = config.rope_theta or 10000.0
        rope_scaling = config.rope_scaling or {}
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        if rope_type in (None, "default"):
            inv_freq = 1.0 / (
                theta ** (torch.arange(0, self.qk_rope_head_dim, 2, dtype=torch.float32, device=device) / self.qk_rope_head_dim)
            )
            self.attention_factor = 1.0
        else:
            inv_freq, self.attention_factor = yarn_rope_params(
                self.qk_rope_head_dim, theta, config.max_position_embeddings, rope_scaling
            )
            inv_freq = inv_freq.to(device)
        self.register_buffer("inv_freq", inv_freq)
        self.scale = yarn_softmax_scale(self.qk_head_dim ** -0.5, rope_scaling)

    def forward(self, hidden_states, positions, table_idx, ctx, batch):
        T = hidden_states.shape[0]

        if self.q_lora_rank:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        else:
            q = self.q_proj(hidden_states)
        q = q.view(T, self.num_heads, self.qk_head_dim)
        q_pass, q_rot = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # [T, kv_lora_rank + qk_rope_head_dim]
        kv_latent, k_rot = compressed_kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_latent = self.kv_a_layernorm(kv_latent)  # [T, kv_lora_rank] -- POST-norm is what gets cached

        # Interleaved-pair RoPE (real DeepSeek-V2 convention, NOT V4's
        # half-split) -- see module docstring.
        freqs = torch.outer(positions.to(torch.float32), self.inv_freq) * self.attention_factor  # [T, rope/2]
        cos, sin = freqs.cos(), freqs.sin()
        q_rot = apply_interleaved_rope(q_rot, cos[:, None, :], sin[:, None, :])
        k_rot = apply_interleaved_rope(k_rot, cos, sin)  # [T, rope]

        cache_row = torch.cat([kv_latent, k_rot], dim=-1)  # [T, kv_lora_rank + rope]
        k_for_pool = cache_row.unsqueeze(0)  # [1, T, D] -- head-major, 1 head
        v_for_pool = torch.zeros_like(k_for_pool)  # V half unused for plain MLA -- see deepseek_v4's own docstring
        ctx.kv_cache.write_kv(k_for_pool, v_for_pool, positions, self.layer_id)

        written = req_written_len(ctx, batch, table_idx)
        read_pos = torch.arange(written, device=hidden_states.device)
        cached_tok, _ = ctx.kv_cache.read_kv(table_idx, read_pos, self.layer_id)
        cached = cached_tok.squeeze(1)  # [written, kv_lora_rank + rope]
        hist_latent, hist_k_rot = cached.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        expanded = self.kv_b_proj(hist_latent).view(written, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        hist_k_nope, hist_v = expanded.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        hist_k_rot_expanded = hist_k_rot[:, None, :].expand(written, self.num_heads, self.qk_rope_head_dim)
        hist_k = torch.cat([hist_k_nope, hist_k_rot_expanded], dim=-1)  # [written, heads, qk_head_dim]

        q_full = torch.cat([q_pass, q_rot], dim=-1)  # [T, heads, qk_head_dim]
        scores = torch.einsum("thd,khd->htk", q_full, hist_k) * self.scale  # [heads, T, written]
        allowed = positions[None, :, None] >= read_pos[None, None, :]
        scores = torch.where(allowed, scores, torch.full_like(scores, float("-inf")))
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("htk,khd->thd", probs, hist_v)  # [T, heads, v_head_dim]
        return self.o_proj(out.reshape(T, -1))


def req_written_len(ctx, batch, table_idx: int) -> int:
    """Identical to `deepseek_v4`'s own helper of the same name."""
    req = next((r for r in batch.reqs if r.table_idx == table_idx), batch.reqs[0])
    is_decode = batch.phase == "decode"
    return req.device_len if is_decode else req.cached_len + req.extend_len


class _DeepseekV2LiteMLP(nn.Module):
    """Plain SwiGLU MLP -- a leading (``first_k_dense_replace``) dense
    layer's only feed-forward block. Identical shape to `deepseek_v4`'s own
    ``_DeepseekV4MLP``."""

    def __init__(self, hidden_size: int, intermediate_size: int, dtype) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _DeepseekV2LiteDecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, device, dtype, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        eps = config.attrs.get("rms_norm_eps", 1e-6)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=eps, dtype=dtype)
        self.self_attn = _DeepseekV2LiteMLA(config, device, dtype, layer_id)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=eps, dtype=dtype)
        is_dense = (not config.is_moe) or layer_id < int(config.first_k_dense_replace or 0)
        if is_dense:
            self.mlp = _DeepseekV2LiteMLP(config.hidden_size, config.intermediate_size, dtype)
        else:
            # The real MoE router is issue #218, not yet implemented -- fail
            # loud rather than silently building a numerically-wrong dense
            # MLP for a layer the real checkpoint routes through experts
            # (same "fail loud, not silently wrong" discipline as
            # `deepseek_v4`'s own offload-backend guard).
            raise NotImplementedError(
                f"DeepSeek-Coder-V2-Lite layer {layer_id} is a routed-MoE layer "
                f"(first_k_dense_replace={config.first_k_dense_replace}) -- the real "
                "MoE router is issue #218 (models-dsv2lite-moe), not yet implemented. "
                "This issue (#217) only covers dense layers / MLA attention."
            )

    def forward(self, hidden_states, positions, table_idx, ctx, batch):
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), positions, table_idx, ctx, batch)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class DeepseekV2LiteForCausalLM(nn.Module):
    """DeepSeek-Coder-V2-Lite, MLA-only scope (issue #217). Subclasses
    ``nn.Module`` directly so its parameters are real registered
    ``nn.Parameter``s -- same shape as `deepseek_v4`'s own ForCausalLM."""

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
            _DeepseekV2LiteDecoderLayer(config, device, dtype, layer_id=i) for i in range(num_layers)
        )
        eps = getattr(config, "attrs", {}).get("rms_norm_eps", 1e-6)
        self.norm = nn.RMSNorm(hidden_size, eps=eps, dtype=dtype)
        from freetoken.layers import LinearReplicated

        self.lm_head = LinearReplicated(hidden_size, vocab_size, has_bias=False, dtype=dtype)
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


__all__ = [
    "parse_config",
    "iter_weights",
    "DeepseekV2LiteForCausalLM",
    "_DeepseekV2LiteMLA",
    "apply_interleaved_rope",
    "yarn_rope_params",
    "yarn_softmax_scale",
]
