"""Qwen3 (dense) model -- first-class Intel Arc Pro B70 port.

Upstream NVIDIA path: python/freetoken/models/qwen3/
Fill in: GitHub issue `models-dense` (see docs/architecture.md).

The dense sibling of ``qwen3_moe``: identical attention (grouped-query,
RoPE, per-head q/k RMS-norm) and decoder-layer/causal-LM shape, but a
single plain SwiGLU MLP per layer instead of a router + N experts -- no
MoE offload/CPU/hybrid backend, no host expert banks, every weight lives
on the accelerator. Parameter names match the HF checkpoint exactly
(``model.embed_tokens`` / ``model.layers.<l>.self_attn.*`` /
``model.layers.<l>.mlp.{gate,up,down}_proj.weight`` / ``model.norm`` /
``lm_head``) so the loader (``#17``) fills every weight.

Deliberately self-contained (not importing qwen3_moe's private classes):
every model package in this port stands alone, matching how each of
``qwen3_moe``/``qwen3_5_moe``/etc. is independently portable -- the
attention block below is adapted from ``qwen3_moe``'s own (proven,
tested) ``_Qwen3Attention``, not shared by reference.
"""
from __future__ import annotations

import glob
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from freetoken.models.config import ModelConfig
from freetoken.models.weight import iter_safetensors
from freetoken.utils import cached_load_hf_config

# --------------------------------------------------------------------------- #
# Checkpoint side (loader contract, from `#17`)
# --------------------------------------------------------------------------- #


def _probe_head_dim(model_path: str, num_heads) -> int | None:
    """Recover the per-head dim from the checkpoint's first ``o_proj`` shape.

    Mirrors ``qwen3_moe``'s own ``_probe_head_dim`` exactly: ``o_proj`` is
    ``[hidden, heads*head_dim]``, so its second dim divided by the head
    count is the true per-head dim -- the one source that is always right
    even when the config's own ``head_dim`` is absent or (on an extended-
    head checkpoint) disagrees with ``hidden // heads``.
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
    """Build a :class:`ModelConfig` from a HF Qwen3 (dense) config.

    ``**_kwargs`` absorbs the MoE-only kwargs (``use_offload_moe`` etc.)
    ``load_model`` passes to every architecture's ``parse_config`` when
    re-parsing for a resolved MoE backend -- ``loader.py``'s own
    ``_CPU_MOE_CAPABLE_ARCHS`` gate keeps that re-parse from ever firing
    for a dense model in practice, but accepting (and ignoring) them here
    means a stray call never crashes construction. ``model_path`` lets
    ``_probe_head_dim`` recover the real per-head dim from the checkpoint.
    """
    src = hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(hf_config)
    cfg = ModelConfig(
        architectures=["Qwen3ForCausalLM"],
        hidden_size=src.get("hidden_size"),
        vocab_size=src.get("vocab_size"),
        num_layers=src.get("num_hidden_layers"),
        num_attention_heads=src.get("num_attention_heads"),
        num_key_value_heads=src.get("num_key_value_heads"),
        head_dim=(
            _probe_head_dim(model_path, src.get("num_attention_heads"))
            or (int(src.get("head_dim")) if src.get("head_dim") else None)
        ),
        intermediate_size=src.get("intermediate_size"),
        max_position_embeddings=src.get("max_position_embeddings"),
        tie_word_embeddings=src.get("tie_word_embeddings", False),
        rope_theta=src.get("rope_theta"),
        rope_scaling=src.get("rope_scaling"),
        hidden_act=src.get("hidden_act", "silu"),
        dtype=src.get("torch_dtype"),
    )
    # A dense model: ModelConfig.is_moe defaults False, so the loader's own
    # MoE branches (offload cache, expert banks, moe_backend re-parse) never
    # fire for this architecture -- see loader.py's own `is_moe` gate.
    return cfg


def iter_weights(model_path: str, device: torch.device, **_kwargs) -> "iter[tuple[str, torch.Tensor]]":
    """Yield the checkpoint's tensors, each placed on ``device``.

    No expert routing at all (unlike ``qwen3_moe``'s own ``iter_weights``):
    every tensor in a dense checkpoint is a dense weight. ``**_kwargs``
    absorbs ``include_moe_experts``/``include_non_moe`` the loader may pass
    uniformly across architectures; both are no-ops here since there is
    nothing to filter.

    Synthesizes ``lm_head.weight`` from ``embed_tokens.weight`` for a
    ``tie_word_embeddings: true`` checkpoint that ships no separate
    ``lm_head.weight`` key -- mirrors ``qwen3_moe``'s own fix for the same
    real failure mode (an unfilled ``lm_head`` silently zeros every logit).
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
# Forward side (the real model the engine runs, `#14`)
# --------------------------------------------------------------------------- #


def _xpu_available() -> bool:
    try:
        return bool(torch.xpu.is_available())
    except Exception:
        return False


class _Qwen3Attention(nn.Module):
    """Qwen3 grouped-query attention (RoPE + q/k RMS-norm), KV-pool driven.

    Adapted from ``qwen3_moe``'s own ``_Qwen3Attention`` -- identical math
    and KV-pool contract (see that module's own extensive comments on the
    head-major/token-major layout, which real prefill/decode bugs this
    exact shape fixed, and why); duplicated rather than imported (see this
    module's own docstring on why each model package stands alone).
    """

    def __init__(self, config, device, dtype, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads or config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.num_heads
        from freetoken.layers import LinearOProj, LinearReplicated

        self.q_proj = LinearReplicated(config.hidden_size, self.num_heads * self.head_dim, has_bias=False, dtype=dtype)
        self.k_proj = LinearReplicated(config.hidden_size, self.num_kv_heads * self.head_dim, has_bias=False, dtype=dtype)
        self.v_proj = LinearReplicated(config.hidden_size, self.num_kv_heads * self.head_dim, has_bias=False, dtype=dtype)
        self.o_proj = LinearOProj(self.num_heads * self.head_dim, config.hidden_size, has_bias=False, dtype=dtype)
        self.q_norm = nn.RMSNorm(self.head_dim, dtype=dtype)
        self.k_norm = nn.RMSNorm(self.head_dim, dtype=dtype)
        theta = config.rope_theta or 10000.0
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def _rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        freqs = torch.outer(pos.to(torch.float32), self.inv_freq)  # [N, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [N, D]
        cos = emb.cos()[None, :, :]
        sin = emb.sin()[None, :, :]
        x_f = x.to(torch.float32)
        x1, x2 = x_f[..., ::2], x_f[..., 1::2]
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x_f * cos + rotated * sin).to(x.dtype)

    def forward(self, hidden_states, positions, table_idx, ctx, batch):
        bsz, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(hidden_states).view(bsz, self.num_kv_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(hidden_states).view(bsz, self.num_kv_heads, self.head_dim).transpose(0, 1)
        q = self._rope(self.q_norm(q), positions)
        k = self._rope(self.k_norm(k), positions)
        ctx.kv_cache.write_kv(k, v, positions, self.layer_id)
        out = ctx.attn_backend.forward(q, k, v, self.layer_id, batch, table_idx=table_idx)
        return self.o_proj(out.transpose(0, 1).contiguous().reshape(bsz, -1))


class _Qwen3MLP(nn.Module):
    """The dense SwiGLU MLP (gate/up/down projections) -- every layer's
    only feed-forward block (no router, no experts)."""

    def __init__(self, config, dtype) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, device, dtype, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = nn.RMSNorm(config.hidden_size, dtype=dtype)
        self.self_attn = _Qwen3Attention(config, device, dtype, layer_id)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, dtype=dtype)
        self.mlp = _Qwen3MLP(config, dtype)

    def forward(self, hidden_states, positions, table_idx, ctx, batch):
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), positions, table_idx, ctx, batch)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """The dense Qwen3 model: real forward pass for the Intel engine loop
    (``#14``). Subclasses ``nn.Module`` (not the torch-free ``BaseLLMModel``
    stub) so its parameters are real registered ``nn.Parameter``s -- the
    loader resolves ``named_parameters()``/``named_buffers()`` to fill
    weights, which only works for proper ``nn.Module`` children."""

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
            _Qwen3DecoderLayer(config, device, dtype, layer_id=i) for i in range(num_layers)
        )
        self.norm = nn.RMSNorm(hidden_size, dtype=dtype)
        from freetoken.layers import LinearReplicated

        self.lm_head = LinearReplicated(hidden_size, vocab_size, has_bias=False, dtype=dtype)
        # No MoE offload machinery at all -- a dense model's weights are
        # never non-device-resident, so the engine's moe_offload / moe_cache
        # attribute reads (getattr(model, "moe_offload", False), guarded)
        # simply see the defaults they already treat as "nothing to offload".
        self.moe_offload = False
        self.moe_cache = None
        if self.device.type != "cpu":
            self.to(self.device)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, out_loc: torch.Tensor) -> torch.Tensor:
        """Run one engine step; return the **last-position** logits ``[bs, V]``.

        Same per-request slicing contract as ``qwen3_moe``'s own forward
        (see its own extensive comments on why a decode batch skips the
        device->host extend_lens sync and a prefill batch does not)."""
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


__all__ = ["parse_config", "iter_weights", "Qwen3ForCausalLM"]
