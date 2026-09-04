"""OLMoE (allenai/OLMoE-1B-7B) model -- issue #224 (`models-olmoe-attn`), part of
epic #223.

The smallest real MoE checkpoint found this session: ~6.9B total params,
~1.3B active. Confirmed from the real downloaded checkpoint's own
``config.json`` (not the ``transformers.OlmoeConfig`` class default, which
would have been wrong here): plain MHA (``num_key_value_heads == 16 ==
num_attention_heads``, NOT the MQA an earlier unverified external claim
suggested), 64 routed experts, flat top-8, ``norm_topk_prob: false``, no
shared expert -- the simplest router shape of any MoE model in this port.

QK-norm is real (confirmed via real checkpoint weight names:
``self_attn.q_norm.weight`` / ``self_attn.k_norm.weight`` both present) but
NOT per-head like ``qwen3``/``qwen3_moe`` -- confirmed against the real HF
``modeling_olmoe.py`` (``transformers`` v5.15.1, installed): OLMoE applies
RMSNorm over the FULL flat projected q/k vector (``q_norm =
OlmoeRMSNorm(hidden_size)``, applied to ``q_proj(hidden_states)`` BEFORE
the head reshape), not a per-head-sized norm applied after reshaping into
heads. Getting this wrong (reusing qwen3's per-head norm) would silently
run a differently-shaped, wrong normalization.

Router forward: fused (in-VRAM) only, no offload/CPU/hybrid backend yet --
same documented scope cut as ``glm4_moe``/``gpt_oss``/``deepseek_v4`` (a
later issue, not this one).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from freetoken.models.config import ModelConfig
from freetoken.models.weight import iter_safetensors
from freetoken.utils import cached_load_hf_config

if TYPE_CHECKING:
    pass

# --------------------------------------------------------------------------- #
# Checkpoint side
# --------------------------------------------------------------------------- #


def parse_config(hf_config, model_path: str | None = None, **_kwargs) -> ModelConfig:
    """Build a :class:`ModelConfig` from a real OLMoE HF config.

    ``**_kwargs`` absorbs the MoE-backend kwargs (``use_offload_moe`` etc.)
    ``load_model`` passes uniformly to every architecture's ``parse_config``
    -- this architecture doesn't support those backends yet (fused-only, see
    module docstring), so they are accepted and ignored rather than crashing
    construction.
    """
    src = hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(hf_config)
    num_heads = src.get("num_attention_heads")
    # OLMoE's real HF class default is head_dim = hidden_size // num_attention_heads
    # (no separate ``head_dim`` config field at all) -- confirmed against
    # ``modeling_olmoe.py``'s own ``getattr(config, "head_dim", hidden//heads)``.
    head_dim = None
    if src.get("head_dim"):
        head_dim = int(src["head_dim"])
    elif num_heads and src.get("hidden_size"):
        head_dim = src["hidden_size"] // num_heads

    cfg = ModelConfig(
        architectures=["OlmoeForCausalLM"],
        hidden_size=src.get("hidden_size"),
        vocab_size=src.get("vocab_size"),
        num_layers=src.get("num_hidden_layers"),
        num_experts=src.get("num_experts"),
        num_attention_heads=num_heads,
        # Real checkpoint confirmed num_key_value_heads == num_attention_heads
        # (plain MHA) -- read verbatim from the checkpoint's own config.json,
        # never assume either the class default or an unverified MQA claim.
        num_key_value_heads=src.get("num_key_value_heads") or num_heads,
        head_dim=head_dim,
        intermediate_size=src.get("intermediate_size"),
        moe_intermediate_size=src.get("intermediate_size"),  # OLMoE: one field, no separate MoE size
        num_experts_per_tok=src.get("num_experts_per_tok"),
        max_position_embeddings=src.get("max_position_embeddings"),
        tie_word_embeddings=src.get("tie_word_embeddings", False),
        rope_theta=src.get("rope_theta"),
        rope_scaling=src.get("rope_scaling"),
        hidden_act=src.get("hidden_act", "silu"),
        dtype=src.get("torch_dtype"),
    )
    cfg.is_moe = True
    cfg.attrs["norm_topk_prob"] = bool(src.get("norm_topk_prob", False))
    cfg.attrs["clip_qkv"] = src.get("clip_qkv")
    cfg.attrs["rms_norm_eps"] = src.get("rms_norm_eps", 1e-5)
    return cfg


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool = True,
    include_non_moe: bool = True,
    **_kwargs,
) -> "iter[tuple[str, torch.Tensor]]":
    """Yield the checkpoint's tensors on their destination device.

    Fused-only (no offload backend yet): every tensor, including experts,
    lands on ``device``. Synthesizes ``lm_head.weight`` from
    ``model.embed_tokens.weight`` for a ``tie_word_embeddings: true``
    checkpoint that ships no separate ``lm_head.weight`` key (same real
    failure mode this port's other model packages already guard against --
    an unfilled ``lm_head`` silently zeros every logit).
    """
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
        hf_config = cached_load_hf_config(model_path)
        if bool(getattr(hf_config, "tie_word_embeddings", False)):
            yield "lm_head.weight", embed_tokens_weight


# --------------------------------------------------------------------------- #
# Forward side
# --------------------------------------------------------------------------- #


def _xpu_available() -> bool:
    try:
        return bool(torch.xpu.is_available())
    except Exception:
        return False


class _OlmoeAttention(nn.Module):
    """OLMoE attention: plain MHA/GQA + RoPE + FLAT (non-per-head) q/k RMS-norm.

    Real HF ``OlmoeAttention.forward`` (``modeling_olmoe.py``, confirmed):
    ``q_norm``/``k_norm`` are RMSNorm over the FULL projected q/k vector
    (sizes ``hidden_size`` and ``head_dim * num_key_value_heads``
    respectively), applied BEFORE the reshape into per-head slices --
    unlike ``qwen3``/``qwen3_moe``'s per-head-sized q_norm/k_norm applied
    AFTER the reshape. Getting the order/shape wrong here silently runs a
    different normalization than the checkpoint was trained with.
    """

    def __init__(self, config, device, dtype, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads or config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.num_heads
        self.clip_qkv = config.attrs.get("clip_qkv")
        eps = config.attrs.get("rms_norm_eps", 1e-5)
        from freetoken.layers import LinearOProj, LinearReplicated

        self.q_proj = LinearReplicated(config.hidden_size, self.num_heads * self.head_dim, has_bias=False, dtype=dtype)
        self.k_proj = LinearReplicated(config.hidden_size, self.num_kv_heads * self.head_dim, has_bias=False, dtype=dtype)
        self.v_proj = LinearReplicated(config.hidden_size, self.num_kv_heads * self.head_dim, has_bias=False, dtype=dtype)
        self.o_proj = LinearOProj(self.num_heads * self.head_dim, config.hidden_size, has_bias=False, dtype=dtype)
        # FLAT norm sizes -- NOT self.head_dim (see class docstring).
        self.q_norm = nn.RMSNorm(self.num_heads * self.head_dim, eps=eps, dtype=dtype)
        self.k_norm = nn.RMSNorm(self.num_kv_heads * self.head_dim, eps=eps, dtype=dtype)
        theta = config.rope_theta or 10000.0
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def _rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # Half-split (rotate_half) RoPE -- matches transformers' real OLMoE
        # ``rotate_half``/``apply_rotary_pos_emb`` (confirmed against
        # ``modeling_olmoe.py``: identical to the now-fixed qwen3 convention).
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
        # Flat q/k-norm applied on the UN-reshaped [T, heads*head_dim] output,
        # matching real OLMoE order (q_norm(q_proj(x)) then reshape), before
        # splitting into per-head slices for RoPE/attention/KV-cache write.
        q_flat = self.q_norm(self.q_proj(hidden_states))
        k_flat = self.k_norm(self.k_proj(hidden_states))
        v_flat = self.v_proj(hidden_states)
        if self.clip_qkv is not None:
            q_flat = q_flat.clamp(min=-self.clip_qkv, max=self.clip_qkv)
            k_flat = k_flat.clamp(min=-self.clip_qkv, max=self.clip_qkv)
            v_flat = v_flat.clamp(min=-self.clip_qkv, max=self.clip_qkv)
        q = q_flat.view(bsz, self.num_heads, self.head_dim).transpose(0, 1)
        k = k_flat.view(bsz, self.num_kv_heads, self.head_dim).transpose(0, 1)
        v = v_flat.view(bsz, self.num_kv_heads, self.head_dim).transpose(0, 1)
        q = self._rope(q, positions)
        k = self._rope(k, positions)
        # write_kv's third argument is out_loc -- PHYSICAL pool slots, not
        # logical token positions (see PR #234: passing raw positions here
        # silently corrupted every real request's KV cache on real hardware,
        # since MHAKVCache's real allocator is not an identity map -- slot 0
        # is reserved padding). Resolve through the page table, matching the
        # already-fixed qwen3/qwen3_moe pattern.
        out_loc = ctx.page_table[table_idx, positions.long()]
        ctx.kv_cache.write_kv(k, v, out_loc, self.layer_id)
        out = ctx.attn_backend.forward(q, k, v, self.layer_id, batch, table_idx=table_idx)
        return self.o_proj(out.transpose(0, 1).contiguous().reshape(bsz, -1))


class _OlmoeExpert(nn.Module):
    """A single MoE expert: gate/up/down projections (SwiGLU)."""

    def __init__(self, config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _OlmoeMoE(nn.Module):
    """Flat top-8/64 router, NO shared expert, NO renormalization
    (``norm_topk_prob: false`` on the real checkpoint -- the raw softmax
    top-k weights are used as-is, unlike every grouped-topk MoE in this
    port so far, which always renormalizes). Fused (in-VRAM) only; no
    offload/CPU/hybrid backend yet (later issue, same scope cut as
    ``glm4_moe``/``gpt_oss``/``deepseek_v4``)."""

    def __init__(self, config, device, dtype) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = bool(config.attrs.get("norm_topk_prob", False))
        from freetoken.layers import LinearReplicated

        self.gate = LinearReplicated(config.hidden_size, self.num_experts, has_bias=False, dtype=dtype)
        self.experts = nn.ModuleList(_OlmoeExpert(config).to(device, dtype) for _ in range(self.num_experts))

    def forward(self, hidden_states: torch.Tensor, model=None, batch=None) -> torch.Tensor:
        in_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, in_shape[-1])
        routing = self.gate(flat)
        gate_log = F.softmax(routing, dim=-1, dtype=torch.float32)
        top_w, top_idx = torch.topk(gate_log, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_w = top_w / top_w.sum(dim=-1, keepdim=True)
        top_w = top_w.to(flat.dtype)

        out = torch.zeros_like(flat)
        top_idx_cpu = top_idx.to("cpu")
        for e in range(self.num_experts):
            for slot in range(self.top_k):
                sel_cpu = top_idx_cpu[:, slot] == e
                if not bool(sel_cpu.any()):
                    continue
                idx = sel_cpu.nonzero(as_tuple=True)[0].to(flat.device)
                w = top_w.index_select(0, idx)[:, slot, None]
                y = self.experts[e](flat.index_select(0, idx))
                out.index_add_(0, idx, w * y)
        return out.view(in_shape)


class _OlmoeDecoderLayer(nn.Module):
    def __init__(self, config, device, dtype, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        eps = config.attrs.get("rms_norm_eps", 1e-5)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=eps, dtype=dtype)
        self.self_attn = _OlmoeAttention(config, device, dtype, layer_id)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=eps, dtype=dtype)
        self.mlp = _OlmoeMoE(config, device, dtype)

    def forward(self, hidden_states, positions, table_idx, ctx, batch):
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), positions, table_idx, ctx, batch)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states), model=ctx.model, batch=batch)
        return hidden_states


class OlmoeForCausalLM(nn.Module):
    """The OLMoE model: real forward pass for the Intel engine loop."""

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
        eps = config.attrs.get("rms_norm_eps", 1e-5) if hasattr(config, "attrs") else 1e-5
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            _OlmoeDecoderLayer(config, device, dtype, layer_id=i) for i in range(num_layers)
        )
        self.norm = nn.RMSNorm(hidden_size, eps=eps, dtype=dtype)
        from freetoken.layers import LinearReplicated

        self.lm_head = LinearReplicated(hidden_size, vocab_size, has_bias=False, dtype=dtype)
        # Fused-only: no offload backend yet (see module docstring).
        self.moe_offload = False
        self.moe_cache = None
        self.moe_layer_id = None
        if self.device.type != "cpu":
            self.to(self.device)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, out_loc: torch.Tensor) -> torch.Tensor:
        from freetoken.core import get_global_ctx

        ctx = get_global_ctx()
        batch = ctx.batch
        reqs = batch.reqs
        hidden = self.embed_tokens(input_ids)
        out = torch.empty((batch.size, self.config.hidden_size), device=hidden.device, dtype=hidden.dtype)

        extend_lens = batch.extend_lens
        if extend_lens is None:
            prefill = batch.is_prefill or (input_ids.shape[0] > batch.size)
            extend_lens = [req.extend_len if prefill else 1 for req in reqs]
        is_decode_batch = batch.phase == "decode"
        offset = 0
        for i, req in enumerate(reqs):
            ext = 1 if is_decode_batch else int(extend_lens[i])
            token_slice = slice(offset, offset + ext)
            h = hidden[token_slice]
            for layer in self.layers:
                h = layer(h, positions[token_slice], req.table_idx, ctx, batch)
            out[i] = self.norm(h)[-1]
            offset += ext

        return self.lm_head(out)


__all__ = ["parse_config", "iter_weights", "OlmoeForCausalLM"]
