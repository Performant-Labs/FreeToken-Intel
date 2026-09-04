"""Qwen1.5-MoE-A2.7B model (issue `models-qwen2moe-attn`, #221).

Upstream NVIDIA path: python/freetoken/models/qwen2_moe/

Real, registered `transformers.Qwen2MoeConfig` (`model_type=qwen2_moe`,
confirmed against the real downloaded `Qwen/Qwen1.5-MoE-A2.7B` checkpoint).
Two real differences from this port's Qwen3-family models:

* Attention is plain MHA with **bias terms** on q/k/v projections
  (`qkv_bias=True` on the real checkpoint) -- no q/k RMS-norm at all,
  unlike Qwen3's qk-norm attention. `o_proj` stays bias-free (confirmed
  against `transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeAttention`).
* The MoE router is plain softmax top-k (`Qwen2MoeTopKRouter`: softmax over
  all `num_experts` logits, top-`num_experts_per_tok`, optional renorm) with
  an **always-on, gated** shared expert -- `sigmoid(shared_expert_gate(x)) *
  shared_expert(x)`, added to the routed-expert output. This sigmoid gate is
  a real, different mechanism from `deepseek_v4`/`glm_moe_dsa`'s
  unconditional shared-expert add; confirmed against the real HF
  `Qwen2MoeSparseMoeBlock.forward`.

Issue #221 built config parsing, attention, and the router/shared-expert
combination as standalone primitives. Issue #222 (this file's decoder layer
and `Qwen2MoeForCausalLM`) wires them into the real engine loop: dense vs.
MoE layers split by `mlp_only_layers`/`decoder_sparse_step`, fused
(in-VRAM) MoE only -- offload/cpu/hybrid raise loudly rather than silently
leaving experts uninitialized (the real bug class `loader.py`'s
`_CPU_MOE_CAPABLE_ARCHS` gate exists to prevent).
"""
from __future__ import annotations

import glob
import json
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from freetoken.models.config import ModelConfig
from freetoken.models.weight import iter_safetensors
from freetoken.utils import cached_load_hf_config

# --------------------------------------------------------------------------- #
# Checkpoint side
# --------------------------------------------------------------------------- #


def _probe_head_dim(model_path: str, num_heads) -> int | None:
    """Recover the per-head dim from the checkpoint's real ``o_proj`` shape.

    Mirrors ``qwen3``/``qwen3_moe``'s own ``_probe_head_dim`` exactly.
    """
    if not num_heads or not isinstance(model_path, str) or not os.path.isdir(model_path):
        return None
    try:
        folder = model_path
        index = os.path.join(folder, "model.safetensors.index.json")
        target = None
        if os.path.isfile(index):
            with open(index, encoding="utf-8") as f:
                weight_map = json.load(f)["weight_map"]
            for name in sorted(weight_map):
                if name.endswith(".self_attn.o_proj.weight"):
                    target = (name, weight_map[name]); break
        if target is None:
            for path in sorted(glob.glob(os.path.join(folder, "*.safetensors"))):
                if path.endswith("consolidated.safetensors"):
                    continue
                with safe_open(path, framework="pt", device="cpu") as f:
                    if "model.layers.0.self_attn.o_proj.weight" in f.keys():
                        target = ("model.layers.0.self_attn.o_proj.weight", path); break
        if target is None:
            return None
        name, path = target
        with safe_open(path, framework="pt", device="cpu") as f:
            heads_times_head_dim = f.get_slice(name).get_shape()[1]
        if heads_times_head_dim % num_heads:
            return None
        return heads_times_head_dim // num_heads
    except Exception:
        return None


def parse_config(hf_config, model_path: str | None = None, **_kwargs) -> ModelConfig:
    """Build a :class:`ModelConfig` from a HF Qwen2-MoE (Qwen1.5-MoE) config.

    ``mlp_only_layers``/``decoder_sparse_step`` control the real dense-vs-MoE
    layer split (real checkpoint: ``decoder_sparse_step=1``, no
    ``mlp_only_layers`` key -- the HF class default ``[]`` -- so every layer
    is MoE; do not assume this for a different checkpoint, read it).
    ``**_kwargs`` absorbs the MoE-backend kwargs ``load_model`` passes
    uniformly across architectures.
    """
    src = hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(hf_config)
    cfg = ModelConfig(
        architectures=["Qwen2MoeForCausalLM"],
        hidden_size=src.get("hidden_size"),
        vocab_size=src.get("vocab_size"),
        num_layers=src.get("num_hidden_layers"),
        num_experts=src.get("num_experts"),
        num_attention_heads=src.get("num_attention_heads"),
        num_key_value_heads=src.get("num_key_value_heads"),
        head_dim=(
            _probe_head_dim(model_path, src.get("num_attention_heads"))
            or (int(src.get("head_dim")) if src.get("head_dim") else None)
        ),
        intermediate_size=src.get("intermediate_size"),
        moe_intermediate_size=src.get("moe_intermediate_size"),
        num_experts_per_tok=src.get("num_experts_per_tok"),
        max_position_embeddings=src.get("max_position_embeddings"),
        tie_word_embeddings=src.get("tie_word_embeddings", False),
        rope_theta=src.get("rope_theta"),
        rope_scaling=src.get("rope_scaling"),
        hidden_act=src.get("hidden_act", "silu"),
        dtype=src.get("torch_dtype"),
    )
    cfg.is_moe = True
    cfg.attrs["qkv_bias"] = bool(src.get("qkv_bias", True))
    cfg.attrs["norm_topk_prob"] = bool(src.get("norm_topk_prob", False))
    cfg.attrs["shared_expert_intermediate_size"] = int(src.get("shared_expert_intermediate_size") or 0)
    cfg.attrs["decoder_sparse_step"] = int(src.get("decoder_sparse_step") or 1)
    cfg.attrs["mlp_only_layers"] = list(src.get("mlp_only_layers") or [])
    cfg.attrs["rms_norm_eps"] = float(src.get("rms_norm_eps") or 1e-6)
    return cfg


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool = True,
    include_non_moe: bool = True,
    **_kwargs,
) -> "iter[tuple[str, torch.Tensor]]":
    """Yield the checkpoint's tensors, each on its destination device.

    MoE expert tensors (``...mlp.experts...``) stay on **host** memory (the
    loader's fused-path expert placement reads them from there); every other
    (dense) tensor is yielded on ``device``. ``include_moe_experts``/
    ``include_non_moe`` let the loader stream just one half (mirrors
    ``qwen3_moe``'s own ``iter_weights`` exactly -- same real failure mode:
    without this split, ``load_moe_expert_sources`` chokes on dense tensors
    like ``lm_head.weight`` mixed into the expert stream).

    Synthesizes ``lm_head.weight`` from ``embed_tokens.weight`` for a
    ``tie_word_embeddings: true`` checkpoint that ships no separate
    ``lm_head.weight`` key (same real failure mode ``qwen3``/``qwen3_moe``
    already guard against: an unfilled ``lm_head`` silently zeros every
    logit). The real Qwen1.5-MoE-A2.7B checkpoint does NOT tie embeddings,
    so this is a defensive no-op there, not exercised.
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
# Forward side -- standalone primitives (#221); decoder-layer/engine wiring
# and the causal-LM wrapper are #222's job.
# --------------------------------------------------------------------------- #


class Qwen2MoeAttention(nn.Module):
    """Plain bias-term MHA (no q/k norm), RoPE, KV-pool driven.

    Real difference from this port's Qwen3-family attention: q/k/v carry a
    real bias term (``qkv_bias=True`` on the checkpoint) and there is no
    per-head RMS-norm on q/k at all. ``o_proj`` stays bias-free. Confirmed
    against ``transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeAttention``.
    """

    def __init__(self, config, device, dtype, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads or config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.num_heads
        qkv_bias = bool(config.attrs.get("qkv_bias", True))
        from freetoken.layers import LinearOProj, LinearReplicated

        self.q_proj = LinearReplicated(config.hidden_size, self.num_heads * self.head_dim, has_bias=qkv_bias, dtype=dtype)
        self.k_proj = LinearReplicated(config.hidden_size, self.num_kv_heads * self.head_dim, has_bias=qkv_bias, dtype=dtype)
        self.v_proj = LinearReplicated(config.hidden_size, self.num_kv_heads * self.head_dim, has_bias=qkv_bias, dtype=dtype)
        self.o_proj = LinearOProj(self.num_heads * self.head_dim, config.hidden_size, has_bias=False, dtype=dtype)
        theta = config.rope_theta or 10000.0
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def _rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # Half-split (rotate_half) RoPE -- matches HF's real
        # ``rotate_half``/``apply_rotary_pos_emb`` convention (confirmed
        # directly against transformers' qwen2_moe modeling code; same
        # convention this port's qwen3/qwen3_moe use after their own rope
        # fix this session).
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
        q = self._rope(q, positions)
        k = self._rope(k, positions)
        # write_kv's third argument is out_loc -- PHYSICAL pool slots, not
        # logical token positions (see qwen3/__init__.py's own extensive
        # comment on the real bug this fixed this session: MHAKVCache's
        # real allocator reserves slot 0 as padding, so positions and slots
        # are never the same number for a real request).
        out_loc = ctx.page_table[table_idx, positions.long()]
        ctx.kv_cache.write_kv(k, v, out_loc, self.layer_id)
        out = ctx.attn_backend.forward(q, k, v, self.layer_id, batch, table_idx=table_idx)
        return self.o_proj(out.transpose(0, 1).contiguous().reshape(bsz, -1))


def qwen2moe_router(
    hidden_states: torch.Tensor, gate_weight: torch.Tensor, top_k: int, norm_topk_prob: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The real ``Qwen2MoeTopKRouter``: plain softmax over all experts, then
    top-``top_k``, optional renormalization. ``hidden_states`` is ``[T, H]``,
    ``gate_weight`` is ``[num_experts, H]``. Returns ``(routing_weights,
    selected_experts)``, both ``[T, top_k]``."""
    router_logits = F.linear(hidden_states, gate_weight)  # [T, num_experts]
    router_probs = torch.softmax(router_logits.float(), dim=-1)
    routing_weights, selected_experts = torch.topk(router_probs, top_k, dim=-1)
    if norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    return routing_weights.to(hidden_states.dtype), selected_experts


def qwen2moe_shared_expert_output(
    hidden_states: torch.Tensor,
    shared_gate_proj: torch.Tensor,
    shared_up_proj: torch.Tensor,
    shared_down_proj: torch.Tensor,
    shared_expert_gate_weight: torch.Tensor,
) -> torch.Tensor:
    """The real, gated shared-expert combination: ``sigmoid(shared_expert_gate(x))
    * shared_expert(x)`` -- a real, different mechanism from
    ``deepseek_v4``/``glm_moe_dsa``'s unconditional shared-expert add.
    Confirmed against the real HF ``Qwen2MoeSparseMoeBlock.forward``.
    ``hidden_states`` is ``[T, H]``; the three shared-expert projections are
    plain SwiGLU MLP weights ``[out, in]``; ``shared_expert_gate_weight`` is
    ``[1, H]`` (bias-free)."""
    shared = F.linear(
        F.silu(F.linear(hidden_states, shared_gate_proj)) * F.linear(hidden_states, shared_up_proj),
        shared_down_proj,
    )
    gate = torch.sigmoid(F.linear(hidden_states, shared_expert_gate_weight))
    return gate * shared


# --------------------------------------------------------------------------- #
# Full model wiring (#222): decoder layer + causal-LM wrapper.
# --------------------------------------------------------------------------- #


def _xpu_available() -> bool:
    try:
        return bool(torch.xpu.is_available())
    except Exception:
        return False


class _Qwen2MoeMLP(nn.Module):
    """Plain dense SwiGLU MLP -- used for any layer ``mlp_only_layers``
    marks dense (or every layer, if the checkpoint's ``decoder_sparse_step``
    somehow yields none as MoE, though the real checkpoint has every layer
    MoE)."""

    def __init__(self, hidden_size: int, intermediate_size: int, dtype) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _Qwen2MoeMoE(nn.Module):
    """Router (flat softmax top-k) + per-expert dense loop + the real
    sigmoid-gated shared expert. In-VRAM (fused) only -- offload/cpu/hybrid
    are not wired for this architecture yet, same scope cut as
    ``glm4_moe``/``deepseek_v4``/``glm_moe_dsa`` before their own follow-up
    issues; guard loudly rather than silently leaving experts uninitialized
    (the real bug class `loader.py`'s ``_CPU_MOE_CAPABLE_ARCHS`` gate exists
    to prevent)."""

    def __init__(self, config: ModelConfig, device, dtype, layer_id: int) -> None:
        super().__init__()
        if bool(getattr(config, "use_offload_moe", False)) or bool(getattr(config, "use_cpu_moe", False)) or bool(getattr(config, "use_hybrid", False)):
            raise NotImplementedError(
                "Qwen2MoeForCausalLM only supports the in-VRAM (fused) MoE backend "
                "-- offload/cpu/hybrid are not wired yet. Pass moe_backend=\"fused\" "
                "explicitly (EngineConfig's \"auto\" default does not know this "
                "architecture can't offload)."
            )
        self.layer_id = layer_id
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = bool(config.attrs.get("norm_topk_prob", False))
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False, dtype=dtype)
        self.experts = nn.ModuleList(
            _Qwen2MoeMLP(config.hidden_size, config.moe_intermediate_size, dtype) for _ in range(config.num_experts)
        )
        shared_inter = int(config.attrs.get("shared_expert_intermediate_size", 0))
        self.shared_expert = _Qwen2MoeMLP(config.hidden_size, shared_inter, dtype) if shared_inter > 0 else None
        self.shared_expert_gate = (
            nn.Linear(config.hidden_size, 1, bias=False, dtype=dtype) if shared_inter > 0 else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        in_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, in_shape[-1])
        routing_weights, selected_experts = qwen2moe_router(
            flat, self.gate.weight, self.top_k, self.norm_topk_prob
        )

        sel_cpu = selected_experts.to("cpu")
        out = torch.zeros_like(flat)
        for e in range(len(self.experts)):
            for slot in range(selected_experts.shape[1]):
                mask = sel_cpu[:, slot] == e
                if not bool(mask.any()):
                    continue
                idx = mask.nonzero(as_tuple=True)[0].to(flat.device)
                w = routing_weights.index_select(0, idx)[:, slot, None]
                y = self.experts[e](flat.index_select(0, idx))
                out.index_add_(0, idx, w * y)

        if self.shared_expert is not None:
            out = out + qwen2moe_shared_expert_output(
                flat,
                self.shared_expert.gate_proj.weight,
                self.shared_expert.up_proj.weight,
                self.shared_expert.down_proj.weight,
                self.shared_expert_gate.weight,
            )
        return out.view(in_shape)


class _Qwen2MoeDecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, device, dtype, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        eps = config.attrs.get("rms_norm_eps", 1e-6)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=eps, dtype=dtype)
        self.self_attn = Qwen2MoeAttention(config, device, dtype, layer_id)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=eps, dtype=dtype)

        mlp_only_layers = config.attrs.get("mlp_only_layers", [])
        sparse_step = int(config.attrs.get("decoder_sparse_step", 1)) or 1
        # Real checkpoint's own dense-vs-MoE rule (HF Qwen2MoeDecoderLayer):
        # a layer is dense iff it's in mlp_only_layers OR sparse_step does
        # not divide (layer_id + 1). Confirmed: real checkpoint has
        # mlp_only_layers=[] and decoder_sparse_step=1, so every layer is
        # MoE there -- but don't assume that for any other checkpoint.
        is_dense = layer_id in mlp_only_layers or (layer_id + 1) % sparse_step != 0
        self.mlp = (
            _Qwen2MoeMLP(config.hidden_size, config.intermediate_size, dtype)
            if is_dense
            else _Qwen2MoeMoE(config, device, dtype, layer_id)
        )

    def forward(self, hidden_states, positions, table_idx, ctx, batch):
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), positions, table_idx, ctx, batch)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class Qwen2MoeForCausalLM(nn.Module):
    """Qwen1.5-MoE-A2.7B: real forward pass for the Intel engine loop (``#14``)."""

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
            _Qwen2MoeDecoderLayer(config, device, dtype, layer_id=i) for i in range(num_layers)
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

        hidden = self.embed_tokens(input_ids)
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
    "Qwen2MoeAttention",
    "qwen2moe_router",
    "qwen2moe_shared_expert_output",
    "Qwen2MoeForCausalLM",
]
