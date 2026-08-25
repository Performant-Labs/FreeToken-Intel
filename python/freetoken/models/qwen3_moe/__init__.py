"""Qwen3-MoE model (first-class Intel Arc Pro B70 port).

Upstream NVIDIA path: python/freetoken/models/qwen3_moe/
Fill in: GitHub issue `models-qwen3-moe` (see docs/architecture.md).

This is the real, pure-torch Qwen3-MoE that the engine loop runs. Parameter
names match the HF checkpoint exactly (``model.embed_tokens`` /
``model.layers.<l>.self_attn.*`` / ``model.layers.<l>.mlp.experts.<e>.*`` /
``model.norm`` / ``lm_head``) so the loader (``#17``) fills every weight.

Design:

* **Dense weights** (embeddings, attention, norms, MoE router, lm_head) live
  on the accelerator. The attention block writes each token's K/V into the
  paged KV pool and the reference attention backend (``#14``) reads the full
  history back -- so attention is correct across prefill and decode.
* **MoE experts** (128 per layer) live on **host** RAM (the loader routes
  ``...experts...`` tensors to CPU) and are gathered into the accelerator on
  demand for each token's routed experts. The production fused-MoE /
  host-offload kernel is a later issue; this path is a correct,
  dependency-free reference (per-expert gather, a grouped-GEMM kernel is a
  later optimization).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from freetoken.models.config import ModelConfig
from freetoken.models.weight import iter_safetensors

# --------------------------------------------------------------------------- #
# Checkpoint side (loader contract, from `#17`)
# --------------------------------------------------------------------------- #


def parse_config(hf_config) -> ModelConfig:
    """Build a :class:`ModelConfig` from a HF Qwen3-MoE config.

    ``hf_config`` is the lru-cached object shared across callers, so it is
    copied (``to_dict``) before the parsed fields are derived -- never mutated.
    """
    src = hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(hf_config)
    # transformers' Qwen3MoeConfig stores the expert count under
    # ``num_local_experts`` (its ``num_experts`` attribute is the public alias and
    # is dropped by to_dict); accept either spelling.
    cfg = ModelConfig(
        architectures=["Qwen3MoeForCausalLM"],
        hidden_size=src.get("hidden_size"),
        vocab_size=src.get("vocab_size"),
        num_layers=src.get("num_hidden_layers"),
        num_experts=src.get("num_local_experts") or src.get("num_experts"),
        num_attention_heads=src.get("num_attention_heads"),
        num_key_value_heads=src.get("num_key_value_heads"),
        intermediate_size=src.get("intermediate_size"),
        moe_intermediate_size=src.get("moe_intermediate_size"),
        num_experts_per_tok=src.get("num_experts_per_tok"),
        first_k_dense_replace=src.get("first_k_dense_replace") or 0,
        max_position_embeddings=src.get("max_position_embeddings"),
        tie_word_embeddings=src.get("tie_word_embeddings", False),
        rope_theta=src.get("rope_theta"),
        rope_scaling=src.get("rope_scaling"),
        hidden_act=src.get("hidden_act", "silu"),
        # Record the checkpoint's dtype so the model can build its modules in
        # the same dtype the loader streams weights in (avoids a bf16-module /
        # fp32-weight mismatch when the engine pins a dtype).
        dtype=src.get("torch_dtype"),
    )
    # FreeToken's MoE plumbing keys off config.is_moe; expose it. (num_moe_layers
    # is derived in ModelConfig.__post_init__.)
    cfg.is_moe = True
    return cfg


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool = True,
    include_non_moe: bool = True,
) -> "iter[tuple[str, torch.Tensor]]":
    """Yield the checkpoint's tensors, each on its destination device.

    MoE expert tensors (``...mlp.experts...``) stay on **host** memory -- the XPU
    holds dense weights and serves experts from host offload banks on demand. Every other
    (dense) tensor is yielded on ``device`` (the XPU).
    """
    for name, tensor in iter_safetensors(model_path, device):
        is_expert = ".experts." in name
        if is_expert and not include_moe_experts:
            continue
        if not is_expert and not include_non_moe:
            continue
        # Dense -> destination device; experts -> host offload banks.
        dest = torch.device("cpu") if is_expert else device
        yield name, tensor.to(dest)


# --------------------------------------------------------------------------- #
# Forward side (the real model the engine runs, `#14`)
# --------------------------------------------------------------------------- #


def _xpu_available() -> bool:
    try:
        return bool(torch.xpu.is_available())
    except Exception:
        return False


class _Qwen3Attention(nn.Module):
    """Qwen3 grouped-query attention (RoPE + q/k RMS-norm), KV-pool driven."""

    def __init__(self, config, device, dtype, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads or config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        # Precompute the RoPE inverse frequencies (theta = rope_theta).
        theta = config.rope_theta or 10000.0
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def _rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # x: [H, N, D] (head-major); pos: [N] absolute positions (rotate_half RoPE).
        # The token dim is the *middle* one here, so the per-token cos/sin must index dim 1.
        freqs = torch.outer(pos.to(torch.float32), self.inv_freq)  # [N, D/2]
        # Expand to the full head dim [N, D] (interleaved (x, y) pairs) and
        # place it on the token (middle) dim so it broadcasts over [H, N, D].
        emb = torch.cat((freqs, freqs), dim=-1)  # [N, D]
        cos = emb.cos()[None, :, :]  # [1, N, D]
        sin = emb.sin()[None, :, :]  # [1, N, D]
        x_f = x.to(torch.float32)
        x1, x2 = x_f[..., ::2], x_f[..., 1::2]
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x_f * cos + rotated * sin).to(x.dtype)

    def forward(self, hidden_states, positions, table_idx, ctx, batch):
        bsz, _ = hidden_states.shape
        # Lay the projections out head-major [heads, tokens, head_dim]: the
        # attention backend expects q/k/v in that order (so the per-request
        # token slice is the *middle* dim) and it returns the output the same
        # way, letting us fold the heads back into the hidden dim with an
        # identity transpose(1, 2).
        q = self.q_proj(hidden_states).view(self.num_heads, bsz, self.head_dim)
        k = self.k_proj(hidden_states).view(self.num_kv_heads, bsz, self.head_dim)
        v = self.v_proj(hidden_states).view(self.num_kv_heads, bsz, self.head_dim)
        q = self._rope(self.q_norm(q), positions)
        k = self._rope(self.k_norm(k), positions)
        # Append this step's K/V to the paged pool (token-ordered, so the
        # out_loc gather aligns), then attend over the full KV history.
        ctx.kv_cache.write_kv(k, v, batch.out_loc)
        out = ctx.attn_backend.forward(q, k, v, self.layer_id, batch)
        return self.o_proj(out.transpose(1, 2).reshape(bsz, -1))


class _Qwen3Expert(nn.Module):
    """A single MoE expert: gate/up/down projections (SwiGLU)."""

    def __init__(self, config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _Qwen3MoE(nn.Module):
    """Mixture-of-experts block: router + N experts.

    Weights live on host (loader-routed). For each token the router picks the
    top-k experts; this reference implementation gathers each token's expert
    inputs and runs the selected experts, accumulating the weighted output.
    """

    def __init__(self, config, device, dtype) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(_Qwen3Expert(config).to(device, dtype) for _ in range(self.num_experts))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The engine feeds a *token-major* 2-D slice [num_tokens, hidden]
        # (one request at a time), so we must not assume a [bsz, seq, hidden]
        # batch dim. Flatten to [T, hidden] and restore the same shape on the way out.
        in_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, in_shape[-1])  # [T, hidden]
        routing = self.gate(flat)  # [T, num_experts]
        gate_log = F.softmax(routing, dim=-1)
        top_w, top_idx = torch.topk(gate_log, self.top_k, dim=-1)  # [T, k]
        top_w = (top_w / top_w.sum(dim=-1, keepdim=True)).to(flat.dtype)

        out = torch.zeros_like(flat)
        # Per-expert gather: route each expert's tokens in one matmul each.
        for e in range(self.num_experts):
            for slot in range(self.top_k):
                sel = (top_idx[:, slot] == e)
                if not sel.any():
                    continue
                out[sel] += top_w[sel, slot, None] * self.experts[e](flat[sel])
        return out.view(in_shape)


class _Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, device, dtype, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = nn.RMSNorm(config.hidden_size)
        self.self_attn = _Qwen3Attention(config, device, dtype, layer_id)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size)
        self.mlp = _Qwen3MoE(config, device, dtype)

    def forward(self, hidden_states, positions, table_idx, ctx, batch):
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), positions, table_idx, ctx, batch)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class Qwen3MoeForCausalLM(nn.Module):
    """The Qwen3-MoE model: real forward pass for the Intel engine loop (`#14`).

    Subclasses ``nn.Module`` (not the torch-free ``BaseLLMModel`` stub) so its
    parameters are real registered nn.Parameters: the loader resolves
    ``model.named_parameters()`` / ``named_buffers()`` to fill weights, which
    only works when the submodules are proper nn.Module children.
    """

    def __init__(self, config, device=None) -> None:
        super().__init__()
        self.config = config
        # An explicit device always wins (the loader / engine pass the XPU or
        # CPU). Only when none is given do we default to the XPU (when
        # present) so a bare get_model_class lands parameters on the accelerator.
        if device is None:
            device = torch.device("xpu") if _xpu_available() else torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        dtype = getattr(config, "dtype", None) or torch.bfloat16
        # Defensive defaults: ``get_model_class`` may hand us a minimal config
        # (e.g. the serve-spine's _StubConfig, which only carries the
        # architecture string) when it just wants the *class* to exist. Reading
        # dims via getattr keeps construction from crashing on such configs; a
        # real forward pass still needs the full parsed ModelConfig.
        vocab_size = getattr(config, "vocab_size", 256)
        hidden_size = getattr(config, "hidden_size", 256)
        num_layers = getattr(config, "num_layers", 0)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            _Qwen3DecoderLayer(config, device, dtype, layer_id=i) for i in range(num_layers)
        )
        self.norm = nn.RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, out_loc: torch.Tensor) -> torch.Tensor:
        """Run one engine step; return the **last-position** logits ``[bs, V]``.

        ``input_ids`` / ``positions`` / ``out_loc`` are ``[num_tokens]`` device
        tensors (set by the engine on the global ``Batch``). For decode
        ``num_tokens == bs`` so the last row of each request is its next-token
        logits. Returns ``[bs, vocab_size]``.
        """
        from freetoken.core import get_global_ctx

        ctx = get_global_ctx()
        batch = ctx.batch
        reqs = batch.reqs

        hidden = self.embed_tokens(input_ids)  # [num_tokens, hidden]
        out = torch.empty((batch.size, self.config.hidden_size), device=hidden.device, dtype=hidden.dtype)

        offset = 0
        for i, req in enumerate(reqs):
            ext = req.extend_len if batch.is_prefill else 1
            token_slice = slice(offset, offset + ext)
            h = hidden[token_slice]
            for layer in self.layers:
                h = layer(h, positions[token_slice], req.table_idx, ctx, batch)
            # Keep only the last position of this request (next-token logits).
            out[i] = self.norm(h)[-1]
            offset += ext

        return self.lm_head(out)


__all__ = ["parse_config", "iter_weights", "Qwen3MoeForCausalLM"]
