"""OpenAI gpt-oss (``GptOssForCausalLM``, model_type ``gpt_oss``) -- Intel
Arc Pro B70 port.

Upstream NVIDIA path: python/freetoken/models/gpt_oss/
Fill in: GitHub issue `models-gpt-oss` (see docs/architecture.md).

Real forward-pass math grounded directly against HF transformers' real
``modeling_gpt_oss.py`` and ``modeling_rope_utils.py`` (fetched and read
this session, not guessed) -- four features new to this port:

* **Attention sinks**: one learned scalar per head
  (``self_attn.sinks``, ``[num_attention_heads]``), concatenated as an
  extra always-attended column onto the attention logits before softmax
  (inflating the denominator only -- it never contributes a value
  vector), then dropped. Wired through the existing (previously
  unfilled) ``AttentionSpec.sinks`` field and the reference attention
  backend (``attention/triton.py``), which already declared the field
  but never read it before this PR.
* **Alternating full/sliding-window attention**: each layer's type comes
  straight from the checkpoint's own ``layer_types`` list (``"full_
  attention"`` / ``"sliding_attention"``), not a computed alternation
  rule -- ``attention/triton.py``'s ``AttentionSpec.sliding_window`` was
  already real, working machinery (built for a different model), reused
  unchanged here.
* **YaRN RoPE**: gpt-oss's real ``rope_scaling`` is
  ``{"rope_type": "yarn", "factor", "beta_fast", "beta_slow",
  "original_max_position_embeddings", "truncate"}`` -- the exact
  correction-range / linear-ramp formula from HF's own
  ``_compute_yarn_parameters`` (not this port's plain-theta RoPE, which
  every other model here uses), including the ``attention_factor``
  (``mscale``) applied to the OUTPUT cos/sin, not the inverse
  frequencies.
* **Clamped-GLU expert activation**: NOT plain SwiGLU --
  ``gate.clamp(max=swiglu_limit)``, ``up.clamp(-swiglu_limit,
  swiglu_limit)``, ``glu = gate * sigmoid(gate * 1.702)``,
  ``(up + 1) * glu`` -- confirmed against the real ``GptOssExperts``
  forward; guessing plain SiLU here would be silently, subtly wrong.

The MoE router is a plain top-k-then-softmax (``softmax`` over only the
SELECTED experts' logits, opposite order from this port's Qwen3-family
routers, which softmax over every expert first and then select) --
confirmed against the real ``GptOssTopKRouter``.

Deliberate, documented scope cuts:

* The real checkpoint's routed experts are BIASED (``gate_up_proj_bias``/
  ``down_proj_bias``) and store gate+up FUSED as one packed tensor
  (``gate_up_proj``) -- this port's generic per-expert weight streamer
  (``weight.py``'s ``stream_moe_expert_sources``) has no bias component
  or fused-gate-up convention at all yet (built for the separate-
  gate_proj/up_proj, unbiased shape every other model here uses).
  Wiring the real checkpoint's exact packed+biased tensor layout is
  separable loader-level follow-up; this port's own experts are
  UNBIASED and split gate_proj/up_proj (matching every other model's
  checkpoint convention here) so they load through the existing,
  unmodified generic path -- the CLAMPED-GLU activation math itself
  (this issue's real point of novelty) is exact and unaffected by that
  simplification. Tracked in the models-compat-matrix issue (#187).
* Only the in-VRAM (fused) forward path is built, matching every other
  freshly-ported MoE model in this session (glm4_moe, #22) -- no
  offload/CPU/hybrid backend. "120B will require offload/hybrid on
  32 GB" (this issue's own body) is real, separable follow-up.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from freetoken.attention.base import AttentionSpec
from freetoken.models.config import ModelConfig
from freetoken.models.weight import iter_safetensors
from freetoken.utils import cached_load_hf_config

# --------------------------------------------------------------------------- #
# Checkpoint side (loader contract, from `#17`)
# --------------------------------------------------------------------------- #


def parse_config(hf_config, model_path: str | None = None, **_kwargs) -> ModelConfig:
    """Build a :class:`ModelConfig` from a real HF ``GptOssConfig``.

    ``**_kwargs`` absorbs the MoE-only kwargs ``load_model`` passes to
    every architecture's ``parse_config`` when re-parsing for a resolved
    backend -- this port doesn't build those backends for gpt-oss yet
    (see this module's own docstring).
    """
    del model_path
    src = hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(hf_config)
    cfg = ModelConfig(
        architectures=["GptOssForCausalLM"],
        hidden_size=src.get("hidden_size"),
        vocab_size=src.get("vocab_size"),
        num_layers=src.get("num_hidden_layers"),
        num_experts=src.get("num_local_experts"),
        num_attention_heads=src.get("num_attention_heads"),
        num_key_value_heads=src.get("num_key_value_heads"),
        head_dim=src.get("head_dim"),
        intermediate_size=src.get("intermediate_size"),
        moe_intermediate_size=src.get("intermediate_size"),
        num_experts_per_tok=src.get("num_experts_per_tok"),
        max_position_embeddings=src.get("max_position_embeddings"),
        tie_word_embeddings=src.get("tie_word_embeddings", False),
        rope_theta=src.get("rope_theta"),
        rope_scaling=src.get("rope_scaling"),
        hidden_act=src.get("hidden_act", "silu"),
        dtype=src.get("torch_dtype") or src.get("dtype"),
    )
    cfg.is_moe = True
    cfg.attrs["attention_bias"] = bool(src.get("attention_bias", True))
    cfg.attrs["sliding_window"] = int(src.get("sliding_window") or 0)
    # Per-layer attention type, real checkpoint field -- an explicit list,
    # not a computed alternation rule (confirmed against the real config).
    # Falls back to alternating sliding/full (gpt-oss's own real pattern)
    # only when a caller hands parse_config a config without it (e.g. a
    # minimal test config.json).
    layer_types = src.get("layer_types")
    if not layer_types:
        n = int(src.get("num_hidden_layers") or 0)
        layer_types = ["sliding_attention" if i % 2 == 0 else "full_attention" for i in range(n)]
    cfg.attrs["layer_types"] = layer_types
    cfg.attrs["swiglu_limit"] = float(src.get("swiglu_limit") or 7.0)
    cfg.attrs["rms_norm_eps"] = float(src.get("rms_norm_eps") or 1e-5)
    return cfg


def iter_weights(model_path: str, device: torch.device, *, include_moe_experts: bool = True, include_non_moe: bool = True):
    """Yield the checkpoint's tensors, each on its destination device.

    Same dense/expert split as every other MoE model in this port. See
    this module's own docstring for the deliberate simplification: this
    port's own checkpoint uses separate, unbiased ``gate_proj``/
    ``up_proj``/``down_proj`` per-expert keys (not the real gpt-oss
    checkpoint's fused, biased ``gate_up_proj``), so it loads through the
    existing generic streamer unmodified.
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


def _yarn_inv_freq_and_scaling(
    head_dim: int, rope_theta: float, rope_scaling: dict, device
) -> tuple[torch.Tensor, float]:
    """The exact real YaRN formula (HF ``modeling_rope_utils.py``'s
    ``_compute_yarn_parameters``, fetched and read this session -- not
    guessed): correction-range + linear-ramp blend between the
    extrapolated (raw theta) and interpolated (theta scaled by ``factor``)
    inverse frequencies, plus an ``attention_factor`` (mscale) applied to
    the OUTPUT cos/sin, not the frequencies themselves.
    """
    dim = head_dim
    base = float(rope_theta)
    factor = float(rope_scaling["factor"])
    original_max = float(rope_scaling["original_max_position_embeddings"])
    beta_fast = float(rope_scaling.get("beta_fast") or 32)
    beta_slow = float(rope_scaling.get("beta_slow") or 1)
    truncate = bool(rope_scaling.get("truncate", True))
    attention_factor = rope_scaling.get("attention_factor")
    mscale = rope_scaling.get("mscale")
    mscale_all_dim = rope_scaling.get("mscale_all_dim")

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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Half-split rotation, confirmed against the real modeling file (NOT
    interleaved pairs) -- identical convention to qwen3_5_moe/glm4_moe's
    own partial-RoPE helpers."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class _GptOssAttention(nn.Module):
    """gpt-oss GQA attention: biased q/k/v/o projections, full-head-dim
    YaRN RoPE, per-layer attention sinks (an extra learned always-
    attended softmax column -- see this module's own docstring), and
    (on alternating layers, per the real checkpoint's own ``layer_types``)
    a sliding attention window. No QK-norm (confirmed absent from the
    real modeling file)."""

    def __init__(self, config: ModelConfig, device, dtype, layer_id: int, sliding_window: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads or config.num_attention_heads
        self.head_dim = config.head_dim or (config.hidden_size // self.num_heads)
        self.sliding_window = sliding_window
        bias = bool(config.attrs.get("attention_bias", True))

        from freetoken.layers import LinearOProj, LinearReplicated

        self.q_proj = LinearReplicated(config.hidden_size, self.num_heads * self.head_dim, has_bias=bias, dtype=dtype)
        self.k_proj = LinearReplicated(config.hidden_size, self.num_kv_heads * self.head_dim, has_bias=bias, dtype=dtype)
        self.v_proj = LinearReplicated(config.hidden_size, self.num_kv_heads * self.head_dim, has_bias=bias, dtype=dtype)
        self.o_proj = LinearOProj(self.num_heads * self.head_dim, config.hidden_size, has_bias=bias, dtype=dtype)
        self.sinks = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))

        rope_scaling = config.rope_scaling or {}
        if str(rope_scaling.get("rope_type") or rope_scaling.get("type") or "").lower() == "yarn":
            inv_freq, self.attention_scaling = _yarn_inv_freq_and_scaling(
                self.head_dim, config.rope_theta or 10000.0, rope_scaling, device
            )
        else:
            theta = config.rope_theta or 10000.0
            inv_freq = 1.0 / (
                theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim)
            )
            self.attention_scaling = 1.0
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, hidden_states, positions, table_idx, ctx, batch):
        bsz, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(hidden_states).view(bsz, self.num_kv_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(hidden_states).view(bsz, self.num_kv_heads, self.head_dim).transpose(0, 1)

        freqs = torch.outer(positions.to(torch.float32), self.inv_freq)  # [T, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, D]
        cos = (emb.cos() * self.attention_scaling)[None, :, :]
        sin = (emb.sin() * self.attention_scaling)[None, :, :]
        q_f, k_f = q.to(torch.float32), k.to(torch.float32)
        q = (q_f * cos + _rotate_half(q_f) * sin).to(q.dtype)
        k = (k_f * cos + _rotate_half(k_f) * sin).to(k.dtype)

        ctx.kv_cache.write_kv(k, v, positions, self.layer_id)
        attn_spec = AttentionSpec(sliding_window=self.sliding_window or None, sinks=self.sinks)
        out = ctx.attn_backend.forward(q, k, v, self.layer_id, batch, attn_spec=attn_spec, table_idx=table_idx)
        return self.o_proj(out.transpose(0, 1).contiguous().reshape(bsz, -1))


class _GptOssExpert(nn.Module):
    """One MoE expert: the real clamped-GLU activation (NOT plain SwiGLU
    -- see this module's own docstring), unbiased split gate_proj/up_proj
    (this port's own deliberate checkpoint-format simplification vs. the
    real fused+biased ``gate_up_proj``)."""

    def __init__(self, hidden_size: int, intermediate_size: int, swiglu_limit: float, dtype) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x).clamp(max=self.swiglu_limit)
        up = self.up_proj(x).clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        glu = gate * torch.sigmoid(gate * 1.702)
        return self.down_proj((up + 1) * glu)


class _GptOssMoE(nn.Module):
    """gpt-oss MoE block: a plain top-k-THEN-softmax router (softmax only
    over the SELECTED experts -- the opposite order from this port's
    Qwen3-family routers, confirmed against the real ``GptOssTopKRouter``)
    + N routed experts. No shared expert (gpt-oss has none)."""

    def __init__(self, config: ModelConfig, device, dtype, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.top_k = config.num_experts_per_tok
        bias = bool(config.attrs.get("attention_bias", True))
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=bias, dtype=dtype)
        swiglu_limit = config.attrs.get("swiglu_limit", 7.0)
        self.experts = nn.ModuleList(
            _GptOssExpert(config.hidden_size, config.moe_intermediate_size, swiglu_limit, dtype)
            for _ in range(config.num_experts)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        in_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, in_shape[-1])  # [n_tok, H]
        router_logits = self.gate(flat)  # [n_tok, E]
        topk_values, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_values, dim=-1, dtype=torch.float32).to(flat.dtype)

        top_idx_cpu = topk_indices.to("cpu")
        out = torch.zeros_like(flat)
        for e in range(len(self.experts)):
            for slot in range(self.top_k):
                sel_cpu = top_idx_cpu[:, slot] == e
                if not bool(sel_cpu.any()):
                    continue
                idx = sel_cpu.nonzero(as_tuple=True)[0].to(flat.device)
                w = topk_weights.index_select(0, idx)[:, slot, None]
                y = self.experts[e](flat.index_select(0, idx))
                out.index_add_(0, idx, w * y)
        return out.view(in_shape)


class _GptOssDecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, device, dtype, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        eps = config.attrs.get("rms_norm_eps", 1e-5)
        layer_type = config.attrs["layer_types"][layer_id]
        sliding_window = config.attrs.get("sliding_window", 0) if layer_type == "sliding_attention" else 0
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=eps, dtype=dtype)
        self.self_attn = _GptOssAttention(config, device, dtype, layer_id, sliding_window)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=eps, dtype=dtype)
        self.mlp = _GptOssMoE(config, device, dtype, layer_id)

    def forward(self, hidden_states, positions, table_idx, ctx, batch):
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), positions, table_idx, ctx, batch)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class GptOssForCausalLM(nn.Module):
    """The gpt-oss model: real forward pass for the Intel engine loop
    (``#14``). Subclasses ``nn.Module`` directly so its parameters are
    real registered ``nn.Parameter``s -- the loader resolves
    ``named_parameters()``/``named_buffers()`` to fill weights."""

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
            _GptOssDecoderLayer(config, device, dtype, layer_id=i) for i in range(num_layers)
        )
        eps = getattr(config, "attrs", {}).get("rms_norm_eps", 1e-5)
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


__all__ = ["parse_config", "iter_weights", "GptOssForCausalLM"]
