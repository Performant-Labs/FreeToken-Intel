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

This issue (#221) covers config parsing, attention, and the router/shared-
expert combination as standalone, independently-testable primitives (same
shape as `qwen4_exp`'s own `#206`/`#208` primitives) -- NOT yet wired into a
decoder layer, the engine's KV cache, or a `ForCausalLM` class; that is
issue #222's job.
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
    return cfg


def iter_weights(model_path: str, device: torch.device, **_kwargs) -> "iter[tuple[str, torch.Tensor]]":
    """Yield the checkpoint's tensors, each placed on ``device``.

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
        placed = tensor.to(device)
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


__all__ = [
    "parse_config",
    "iter_weights",
    "Qwen2MoeAttention",
    "qwen2moe_router",
    "qwen2moe_shared_expert_output",
]
