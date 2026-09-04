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


def _xpu_available() -> bool:
    try:
        return bool(torch.xpu.is_available())
    except Exception:
        return False


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool = True,
    include_non_moe: bool = True,
) -> "iter[tuple[str, torch.Tensor]]":
    """Yield the checkpoint's tensors, each on its destination device.

    Fused (in-VRAM) experts only for now (issue #226's own scope note: no
    offload/CPU/hybrid MoE backend for Mellum yet, mirroring how
    ``glm4_moe``/``gpt_oss`` shipped fused-only first) -- every tensor,
    including experts, is placed on ``device``. Real bug found wiring this:
    ``include_moe_experts``/``include_non_moe`` are NOT no-ops even on the
    fused path -- the loader's own ``load_moe_expert_sources`` streams this
    generator with ``include_non_moe=False`` to gather ONLY expert tensors
    into banks (fused and offload both build the banks first; fused then
    copies them into resident expert modules), and ``_place_dense`` streams
    it separately with ``include_moe_experts=False`` for the dense pass.
    Ignoring these kwargs mixes dense tensors (e.g. ``lm_head.weight``) into
    the expert-bank stream, which raises inside
    ``weight.py:stream_moe_expert_sources`` ("Unexpected expert weight
    key") -- confirmed by hitting exactly that error before this filter was
    added. Mirrors ``qwen3_moe``'s own ``iter_weights`` filtering.

    Synthesizes ``lm_head.weight`` from ``embed_tokens.weight`` for a
    ``tie_word_embeddings: true`` checkpoint that ships no separate
    ``lm_head.weight`` key -- same real failure mode ``qwen3``/``qwen3_moe``
    already guard against (an unfilled ``lm_head`` silently zeros every
    logit).
    """
    from freetoken.models.weight import iter_safetensors

    embed_tokens_weight = None
    saw_lm_head = False
    for name, tensor in iter_safetensors(model_path, device):
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


class _MellumExpert(nn.Module):
    """A single MoE expert: gate/up/down projections (SwiGLU), fused/in-VRAM only."""

    def __init__(self, config, dtype) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _MellumMoE(nn.Module):
    """Router + N fused (XPU-resident) experts, no shared expert.

    Reuses ``mellum_moe_router`` (#227's own standalone primitive) for the
    routing math, then a per-expert gather over the resident ``_MellumExpert``
    modules -- the same "route on host, gather with index_select" pattern
    ``qwen3_moe``'s fused path uses (XPU `nonzero()`/boolean-mask indexing on
    this torch/XPU build silently returns empty for a bool tensor regardless
    of content -- a real, confirmed bug this port routes around everywhere).
    """

    def __init__(self, config, device, dtype) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        from freetoken.layers import LinearReplicated

        self.gate = LinearReplicated(config.hidden_size, self.num_experts, has_bias=False, dtype=dtype)
        self.experts = nn.ModuleList(_MellumExpert(config, dtype).to(device, dtype) for _ in range(self.num_experts))

    def forward(self, hidden_states: torch.Tensor, model=None, batch=None) -> torch.Tensor:
        in_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, in_shape[-1])  # [T, hidden]
        top_w, top_idx = mellum_moe_router(flat, self.gate.weight, self.top_k)

        out = torch.zeros_like(flat)
        top_idx_host = top_idx.to("cpu")
        for e in range(self.num_experts):
            token_rows, k_slots = (top_idx_host == e).nonzero(as_tuple=True)
            if token_rows.numel() == 0:
                continue
            token_rows = token_rows.to(flat.device)
            k_slots = k_slots.to(flat.device)
            sel = flat.index_select(0, token_rows)
            expert_out = self.experts[e](sel)
            w = top_w[token_rows, k_slots].unsqueeze(-1)
            out.index_add_(0, token_rows, expert_out * w)
        return out.view(in_shape)


class _MellumDecoderLayer(nn.Module):
    def __init__(self, config, device, dtype, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        layer_types = config.attrs.get("layer_types") or []
        layer_type = layer_types[layer_id] if layer_id < len(layer_types) else "full_attention"
        eps = config.attrs.get("rms_norm_eps", 1e-6)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=eps, dtype=dtype)
        self.self_attn = MellumAttention(config, device, dtype, layer_id, layer_type)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=eps, dtype=dtype)
        self.mlp = _MellumMoE(config, device, dtype)
        self.mlp.layer_id = layer_id

    def forward(self, hidden_states, positions, table_idx, ctx, batch):
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), positions, table_idx, ctx, batch)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = residual + self.mlp(
            self.post_attention_layernorm(hidden_states), model=ctx.model, batch=batch
        )
        return hidden_states


class MellumForCausalLM(nn.Module):
    """The Mellum2 model: real forward pass for the Intel engine loop.

    Subclasses ``nn.Module`` (not the torch-free ``BaseLLMModel`` stub) so its
    parameters are real registered ``nn.Parameter``s -- the loader resolves
    ``named_parameters()``/``named_buffers()`` to fill weights, which only
    works for proper ``nn.Module`` children. Structurally identical to
    ``qwen3_moe``'s own ``Qwen3MoeForCausalLM`` (same reuse this whole
    package already leans on), fused-MoE-only (no offload path yet, see
    ``iter_weights``'s own note).
    """

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
            _MellumDecoderLayer(config, device, dtype, layer_id=i) for i in range(num_layers)
        )
        eps = config.attrs.get("rms_norm_eps", 1e-6) if getattr(config, "attrs", None) else 1e-6
        self.norm = nn.RMSNorm(hidden_size, eps=eps, dtype=dtype)
        from freetoken.layers import LinearReplicated

        self.lm_head = LinearReplicated(hidden_size, vocab_size, has_bias=False, dtype=dtype)
        # No offload/CPU/hybrid MoE backend yet (see iter_weights); always
        # fused/in-VRAM, so the engine's offload plumbing must see this model
        # as never eligible for it.
        self.moe_offload = False
        self.moe_cache = None
        self.moe_layer_id = None
        if self.device.type != "cpu":
            self.to(self.device)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, out_loc: torch.Tensor) -> torch.Tensor:
        """Run one engine step; return the **last-position** logits ``[bs, V]``."""
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
    "MellumAttention",
    "mellum_moe_router",
    "parse_config",
    "iter_weights",
    "MellumForCausalLM",
]
