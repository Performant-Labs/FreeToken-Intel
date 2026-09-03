"""GLM-5.2 (``GlmMoeDsaForCausalLM``, model_type ``glm_moe_dsa``) -- Intel
Arc Pro B70 port.

Upstream NVIDIA path: python/freetoken/models/glm_moe_dsa/
Fill in: GitHub issue `models-glm` (see docs/architecture.md).

The GLM-5.2 sibling of ``deepseek_v4`` (MLA #190 + DSA #191 + MoE #192,
all merged): same MLA/DSA/router mechanism, but GLM-5.2 diverges from
DeepSeek-V3.2 in two real, confirmed ways -- grounded directly against the
real, already-installed ``transformers`` package's own
``models/glm_moe_dsa/modeling_glm_moe_dsa.py`` (read directly this
session, not guessed -- a real ``GlmMoeDsaConfig``/model already exists in
the installed transformers, the same discovery that grounded #190):

* **Interleaved RoPE, not half-split**: both the main MLA attention AND
  the DSA indexer use ``apply_rotary_pos_emb_interleave`` (even/odd-pair
  rotation: ``q1,q2 = q[...,0::2], q[...,1::2]``, output
  ``cat([q1*cos-q2*sin, q2*cos+q1*sin])``) -- confirmed by an explicit
  real docstring on ``GlmMoeDsaIndexer.forward`` itself: "Same as
  ``DeepseekV32Indexer.forward``, but the indexer applies **interleaved**
  RoPE rather than the non-interleaved half-split RoPE used by
  DeepSeek-V3.2." ``deepseek_v4`` (#190/#191) uses half-split for both;
  this is a real, deliberate architectural difference, not an
  inconsistency to reconcile.
* **Cross-layer top-k sharing** (the real answer to the open question
  #191 flagged): ``config.indexer_types[layer_idx]`` is ``"full"`` (this
  layer runs its own indexer) or ``"shared"`` (this layer reuses the
  PREVIOUS layer's top-k selection, no indexer weights at all for that
  layer -- confirmed: ``self.indexer = None if skip_topk else
  GlmMoeDsaIndexer(...)``). The selection is threaded layer-to-layer as
  ``prev_topk_indices`` / a returned ``topk_indices`` (real code:
  ``topk_indices = None`` before the layer loop, each layer receives the
  previous one's output and returns its own -- reused verbatim if
  ``"shared"``). ``indexer_types``, when absent from the checkpoint, is
  DERIVED via the real formula (``configuration_glm_moe_dsa.py``'s own
  ``__post_init__``): ``"full" if (max(i - offset + 1, 0) % freq) == 0
  else "shared"`` from ``index_topk_freq``/``index_skip_topk_offset`` --
  the real, authoritative resolution of the four config fields #191 had
  explicitly flagged as unresolved (``indexer_types``/``index_topk_freq``/
  ``index_skip_topk_offset``/``index_share_for_mtp_iteration``; the last
  one is genuinely MTP-only, confirmed absent from this forward path,
  same conclusion #171/#172's own MTP research reached for a different
  field).

Router (grouped/sigmoid/bias-corrected top-k) and MoE block shape are
otherwise identical to ``deepseek_v4``'s own (confirmed against the real
``GlmMoeDsaTopkRouter`` -- byte-for-byte the same class as
``DeepseekV3TopkRouter``/``Glm4MoeTopkRouter``), so duplicated verbatim
from there (every model package in this port stands alone).

Same deliberate scope cuts ``deepseek_v4`` already documented: in-VRAM
(fused) MoE only (no offload/CPU/hybrid backend -- and now enforced at the
loader level too, see ``loader.py``'s own real bug-fix from #192), and
experts split as separate unbiased ``gate_proj``/``up_proj``/``down_proj``
per-expert modules (this port's own established per-expert checkpoint
convention) rather than the real checkpoint's packed 3-D ``gate_up_proj``
tensor.
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


def _derive_indexer_types(num_layers: int, index_topk_freq, index_skip_topk_offset, index_topk_pattern) -> list[str]:
    """The real derivation from ``configuration_glm_moe_dsa.py``'s own
    ``__post_init__`` (read directly, not guessed): a pattern string (e.g.
    ``"FSSF..."``) overrides the freq/offset schedule when given; otherwise
    ``"full" if (max(i - offset + 1, 0) % freq) == 0 else "shared"``."""
    if index_topk_pattern is not None:
        if isinstance(index_topk_pattern, str):
            return [{"F": "full", "S": "shared"}[c] for c in index_topk_pattern]
        return list(index_topk_pattern)
    freq = max(int(index_topk_freq) if index_topk_freq else 1, 1)
    offset = int(index_skip_topk_offset) if index_skip_topk_offset is not None else 2
    return ["full" if (max(i - offset + 1, 0) % freq) == 0 else "shared" for i in range(num_layers)]


def parse_config(hf_config, model_path: str | None = None, **_kwargs) -> ModelConfig:
    """Build a :class:`ModelConfig` from a real HF ``GlmMoeDsaConfig``.

    Mirrors ``deepseek_v4``'s own ``parse_config`` (same "some fields leak
    a class default / don't match to_dict()" caution, resolved the same
    way -- reading the checkpoint's actual config.json file for the
    "was this explicitly set" gating decisions), plus deriving
    ``indexer_types`` when the checkpoint doesn't ship it explicitly.
    """
    raw = hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(hf_config)
    file_raw = _raw_checkpoint_json(model_path) if model_path else raw

    def field(name, default=None):
        val = getattr(hf_config, name, None)
        return val if val is not None else raw.get(name, default)

    src = {k: field(k) for k in (
        "hidden_size", "vocab_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "q_lora_rank", "kv_lora_rank", "qk_rope_head_dim",
        "qk_nope_head_dim", "v_head_dim", "attention_bias", "intermediate_size",
        "max_position_embeddings", "tie_word_embeddings", "rope_parameters",
        "hidden_act", "torch_dtype", "dtype", "rms_norm_eps",
        "index_topk", "index_head_dim", "index_n_heads",
        "index_topk_freq", "index_skip_topk_offset", "index_topk_pattern",
        "indexer_types",
        "n_routed_experts", "moe_intermediate_size", "num_experts_per_tok",
        "first_k_dense_replace", "n_group", "topk_group",
        "routed_scaling_factor", "n_shared_experts", "norm_topk_prob",
    )}
    kv_lora_rank = int(src.get("kv_lora_rank") or 0)
    qk_rope_head_dim = int(src.get("qk_rope_head_dim") or 0)
    # is_moe / moe_intermediate_size: file_raw, not src/field() -- confirmed
    # directly (mirrors deepseek_v4's own #191/#192 findings) that a real
    # installed HF config class's non-None defaults for these leak through
    # both getattr AND to_dict() even when the checkpoint's own config.json
    # never mentions them.
    is_moe = bool(file_raw.get("n_routed_experts"))
    rope_parameters = src.get("rope_parameters") or {}
    rope_theta = rope_parameters.get("rope_theta") or 10000.0

    cfg = ModelConfig(
        architectures=["GlmMoeDsaForCausalLM"],
        hidden_size=src.get("hidden_size"),
        vocab_size=src.get("vocab_size"),
        num_layers=src.get("num_hidden_layers"),
        num_attention_heads=src.get("num_attention_heads"),
        num_key_value_heads=1,  # MLA: one shared compressed latent, not per-head K/V
        head_dim=kv_lora_rank + qk_rope_head_dim,  # the pool's real per-token row width
        intermediate_size=src.get("intermediate_size"),
        moe_intermediate_size=file_raw.get("moe_intermediate_size") if is_moe else None,
        num_experts=int(src["n_routed_experts"]) if is_moe else None,
        num_experts_per_tok=int(src.get("num_experts_per_tok") or 0) if is_moe else None,
        first_k_dense_replace=int(src.get("first_k_dense_replace") or 0),
        is_moe=is_moe,
        max_position_embeddings=src.get("max_position_embeddings"),
        tie_word_embeddings=src.get("tie_word_embeddings", False),
        kv_lora_rank=kv_lora_rank or None,
        q_lora_rank=int(src.get("q_lora_rank")) if src.get("q_lora_rank") else None,
        rope_theta=rope_theta,
        rope_scaling=rope_parameters,
        hidden_act=src.get("hidden_act", "silu"),
        dtype=src.get("torch_dtype") or src.get("dtype"),
    )
    cfg.attrs["qk_rope_head_dim"] = qk_rope_head_dim
    cfg.attrs["qk_nope_head_dim"] = int(src.get("qk_nope_head_dim") or 0)
    cfg.attrs["v_head_dim"] = int(src.get("v_head_dim") or 0)
    cfg.attrs["attention_bias"] = bool(src.get("attention_bias", False))
    cfg.attrs["rms_norm_eps"] = float(src.get("rms_norm_eps") or 1e-5)
    if is_moe:
        cfg.attrs["n_group"] = int(src.get("n_group") or 1)
        cfg.attrs["topk_group"] = int(src.get("topk_group") or 1)
        cfg.attrs["routed_scaling_factor"] = float(src.get("routed_scaling_factor") or 1.0)
        cfg.attrs["n_shared_experts"] = int(src.get("n_shared_experts") or 0)
        cfg.attrs["norm_topk_prob"] = bool(src.get("norm_topk_prob", True))
    # DSA (#191) + cross-layer sharing (#22's own real answer, see module
    # docstring): only set when the checkpoint's own config.json FILE
    # explicitly declares index_topk.
    if file_raw.get("index_topk"):
        cfg.attrs["index_topk"] = int(file_raw["index_topk"])
        cfg.attrs["index_head_dim"] = int(file_raw.get("index_head_dim") or 0)
        cfg.attrs["index_n_heads"] = int(file_raw.get("index_n_heads") or 1)
        num_layers = int(src.get("num_hidden_layers") or 0)
        indexer_types = file_raw.get("indexer_types")
        if indexer_types is None:
            indexer_types = _derive_indexer_types(
                num_layers,
                file_raw.get("index_topk_freq"),
                file_raw.get("index_skip_topk_offset"),
                file_raw.get("index_topk_pattern"),
            )
        cfg.attrs["indexer_types"] = indexer_types
    return cfg


def iter_weights(model_path: str, device: torch.device, *, include_moe_experts: bool = True, include_non_moe: bool = True):
    """Yields the checkpoint's tensors, each on its destination device.
    Identical dense/expert split to ``deepseek_v4``'s own ``iter_weights``
    (issue #192) -- see its own docstring."""
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


def _apply_rotary_pos_emb_interleave(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Interleaved-pair RoPE (GLM-5.2's own real convention, confirmed
    against the real ``apply_rotary_pos_emb_interleave`` -- NOT the
    half-split ``rotate_half`` ``deepseek_v4`` uses). ``q``/``k`` are
    ``[T, heads_or_1, rope_dim]``; ``cos``/``sin`` are ``[T, rope_dim]``
    (``cat(freqs, freqs)`` -- only the first half is the real per-pair
    angle, matching the real code's own ``cos[..., :D//2]`` slice)."""
    half = cos.shape[-1] // 2
    cos = cos[..., :half].unsqueeze(1)
    sin = sin[..., :half].unsqueeze(1)
    q1, q2 = q[..., 0::2], q[..., 1::2]
    k1, k2 = k[..., 0::2], k[..., 1::2]
    q_embed = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k_embed = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def req_written_len(ctx, batch, table_idx: int) -> int:
    """Identical to ``deepseek_v4``'s own helper of the same name."""
    req = next((r for r in batch.reqs if r.table_idx == table_idx), batch.reqs[0])
    is_decode = batch.phase == "decode"
    return req.device_len if is_decode else req.cached_len + req.extend_len


class _GlmMoeDsaMLA(nn.Module):
    """MLA + DSA with cross-layer top-k sharing (see module docstring).
    Structurally identical to ``deepseek_v4``'s own ``_DeepseekV4MLA``
    (same compression/decompression/cache-reuse design -- see that
    module's own docstring for the shared rationale), except: (1)
    interleaved RoPE throughout, (2) a ``"shared"`` layer has NO indexer
    at all and instead consumes ``prev_topk_indices`` from the previous
    layer's forward call, returning it unchanged for the next.
    """

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
        eps = config.attrs.get("rms_norm_eps", 1e-5)
        hidden = config.hidden_size

        if self.q_lora_rank:
            self.q_a_proj = nn.Linear(hidden, self.q_lora_rank, bias=bias, dtype=dtype)
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=eps, dtype=dtype)
            self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False, dtype=dtype)
        else:
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
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, self.qk_rope_head_dim, 2, dtype=torch.float32, device=device) / self.qk_rope_head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        self.scale = self.qk_head_dim ** -0.5

        # DSA + cross-layer sharing (see module docstring). indexer_types
        # is only in config.attrs when the checkpoint declares index_topk
        # at all (see parse_config); every plain-MLA (no DSA) checkpoint
        # is completely unaffected.
        self.index_topk = config.attrs.get("index_topk")
        self.skip_topk = False
        if self.index_topk:
            indexer_types = config.attrs["indexer_types"]
            self.skip_topk = indexer_types[layer_id] == "shared"
            if not self.skip_topk:
                self.index_head_dim = config.attrs["index_head_dim"]
                self.index_n_heads = config.attrs["index_n_heads"]
                pool_row_width = self.kv_lora_rank + self.qk_rope_head_dim
                if self.index_head_dim > pool_row_width:
                    raise ValueError(
                        f"index_head_dim ({self.index_head_dim}) exceeds this pool's row "
                        f"width ({pool_row_width}) -- the V-buffer reuse this port's DSA "
                        "relies on (see deepseek_v4's own module docstring) needs the "
                        "indexer key to fit inside MLA's own (otherwise-unused) V slot."
                    )
                q_resid_width = self.q_lora_rank if self.q_lora_rank else hidden
                self.wq_b = nn.Linear(q_resid_width, self.num_heads * self.index_head_dim, bias=False, dtype=dtype)
                self.wk = nn.Linear(hidden, self.index_head_dim, bias=False, dtype=dtype)
                self.indexer_k_norm = nn.LayerNorm(self.index_head_dim, eps=eps, dtype=dtype)
                self.weights_proj = nn.Linear(hidden, self.num_heads, bias=False, dtype=dtype)
                self.index_scale = self.index_head_dim ** -0.5

    def forward(self, hidden_states, positions, table_idx, ctx, batch, prev_topk_indices=None):
        T = hidden_states.shape[0]

        q_resid = None
        if self.q_lora_rank:
            q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
            q = self.q_b_proj(q_resid)
        else:
            q = self.q_proj(hidden_states)
        q = q.view(T, self.num_heads, self.qk_head_dim)
        q_pass, q_rot = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # [T, kv_lora_rank + qk_rope_head_dim]
        kv_latent, k_rot = compressed_kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_latent = self.kv_a_layernorm(kv_latent)  # [T, kv_lora_rank] -- POST-norm is what gets cached

        freqs = torch.outer(positions.to(torch.float32), self.inv_freq)  # [T, rope/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, rope]
        q_rot, k_rot_2d = _apply_rotary_pos_emb_interleave(q_rot, k_rot.unsqueeze(1), emb.cos(), emb.sin())
        k_rot = k_rot_2d.squeeze(1)  # [T, rope]

        pool_row_width = self.kv_lora_rank + self.qk_rope_head_dim
        cache_row = torch.cat([kv_latent, k_rot], dim=-1)  # [T, kv_lora_rank + rope]
        k_for_pool = cache_row.unsqueeze(0)  # [1, T, D] -- head-major, 1 head

        topk_indices = prev_topk_indices
        if self.index_topk and not self.skip_topk:
            k_idx = self.indexer_k_norm(self.wk(hidden_states))  # [T, index_head_dim]
            k_idx_rot, k_idx_pass = k_idx.split([self.qk_rope_head_dim, self.index_head_dim - self.qk_rope_head_dim], dim=-1)
            q_idx_full = self.wq_b(q_resid if q_resid is not None else hidden_states).view(T, self.num_heads, self.index_head_dim)
            q_idx_rot, q_idx_pass = q_idx_full.split([self.qk_rope_head_dim, self.index_head_dim - self.qk_rope_head_dim], dim=-1)
            q_idx_rot, k_idx_rot_2d = _apply_rotary_pos_emb_interleave(q_idx_rot, k_idx_rot.unsqueeze(1), emb.cos(), emb.sin())
            k_idx_rot = k_idx_rot_2d.squeeze(1)
            k_idx = torch.cat([k_idx_rot, k_idx_pass], dim=-1)
            v_for_pool = F.pad(k_idx, (0, pool_row_width - self.index_head_dim)).unsqueeze(0)
        else:
            v_for_pool = torch.zeros_like(k_for_pool)
        ctx.kv_cache.write_kv(k_for_pool, v_for_pool, positions, self.layer_id)

        written = req_written_len(ctx, batch, table_idx)
        read_pos = torch.arange(written, device=hidden_states.device)
        cached_tok, cached_v_tok = ctx.kv_cache.read_kv(table_idx, read_pos, self.layer_id)
        cached = cached_tok.squeeze(1)
        hist_latent, hist_k_rot = cached.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        index_mask = None
        if self.index_topk:
            if not self.skip_topk:
                q_idx = torch.cat([q_idx_rot, q_idx_pass], dim=-1)  # [T, heads, index_head_dim]
                hist_k_idx = cached_v_tok.squeeze(1)[:, : self.index_head_dim]  # [written, index_head_dim]
                raw = F.relu(
                    torch.einsum("thd,kd->htk", q_idx.float(), hist_k_idx.float()) * self.index_scale
                )
                w = self.weights_proj(hidden_states).float() * (self.num_heads ** -0.5)
                index_scores = torch.einsum("th,htk->tk", w, raw)
                causal_idx = positions[:, None] >= read_pos[None, :]
                index_scores = index_scores.masked_fill(~causal_idx, float("-inf"))
                topk = min(self.index_topk, written)
                _, topk_indices = torch.topk(index_scores, topk, dim=-1)
            elif topk_indices is None:
                raise ValueError(
                    f"Layer {self.layer_id} is a DSA \"shared\" indexer layer but received no "
                    "prev_topk_indices from a previous \"full\" layer -- indexer_types is malformed."
                )
            index_mask = torch.zeros(T, written, dtype=torch.bool, device=hidden_states.device)
            index_mask.scatter_(1, topk_indices, True)

        expanded = self.kv_b_proj(hist_latent).view(written, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        hist_k_nope, hist_v = expanded.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        hist_k_rot_expanded = hist_k_rot[:, None, :].expand(written, self.num_heads, self.qk_rope_head_dim)
        hist_k = torch.cat([hist_k_nope, hist_k_rot_expanded], dim=-1)

        q_full = torch.cat([q_pass, q_rot], dim=-1)
        scores = torch.einsum("thd,khd->htk", q_full, hist_k) * self.scale
        allowed = positions[None, :, None] >= read_pos[None, None, :]
        if index_mask is not None:
            allowed = allowed & index_mask[None, :, :]
        scores = torch.where(allowed, scores, torch.full_like(scores, float("-inf")))
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("htk,khd->thd", probs, hist_v)
        attn_out = self.o_proj(out.reshape(T, -1))
        return attn_out, topk_indices


class _GlmMoeDsaMLP(nn.Module):
    """Plain SwiGLU MLP -- identical shape to every other model here."""

    def __init__(self, hidden_size: int, intermediate_size: int, dtype) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _GlmMoeDsaTopkRouter(nn.Module):
    """Identical math to ``deepseek_v4``'s own ``_DeepseekV4TopkRouter``
    (confirmed byte-for-byte against the real ``GlmMoeDsaTopkRouter``)."""

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
        router_logits = F.linear(hidden_states.float(), self.weight.float())
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        if self.n_group > 1:
            T = scores_for_choice.shape[0]
            group_scores = (
                scores_for_choice.view(T, self.n_group, self.num_experts // self.n_group)
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1)[1]
            group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1.0)
            expert_mask = (
                group_mask.unsqueeze(-1)
                .expand(T, self.n_group, self.num_experts // self.n_group)
                .reshape(T, self.num_experts)
            )
            scores_for_choice = scores_for_choice.masked_fill(expert_mask == 0, float("-inf"))
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1)[1]
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


class _GlmMoeDsaMoE(nn.Module):
    """Identical shape to ``deepseek_v4``'s own ``_DeepseekV4MoE`` -- see
    that module's own docstring for the shared rationale (in-VRAM only,
    the loader-level offload-eligibility fix from #192 applies here too
    since ``GlmMoeDsaForCausalLM`` is likewise not in
    ``loader.py``'s ``_CPU_MOE_CAPABLE_ARCHS``)."""

    def __init__(self, config: ModelConfig, device, dtype, layer_id: int) -> None:
        super().__init__()
        if bool(getattr(config, "use_offload_moe", False)) or bool(getattr(config, "use_cpu_moe", False)) or bool(getattr(config, "use_hybrid", False)):
            raise NotImplementedError(
                "GlmMoeDsaForCausalLM only supports the in-VRAM (fused) MoE backend "
                "-- offload/cpu/hybrid are not wired yet (see the module's own "
                "docstring). Pass moe_backend=\"fused\" explicitly (EngineConfig's "
                "\"auto\" default does not know this architecture can't offload)."
            )
        self.layer_id = layer_id
        self.gate = _GlmMoeDsaTopkRouter(config, dtype)
        self.experts = nn.ModuleList(
            _GlmMoeDsaMLP(config.hidden_size, config.moe_intermediate_size, dtype) for _ in range(config.num_experts)
        )
        n_shared = config.attrs.get("n_shared_experts", 0)
        self.shared_experts = (
            _GlmMoeDsaMLP(config.hidden_size, config.moe_intermediate_size * n_shared, dtype) if n_shared > 0 else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        in_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, in_shape[-1])
        topk_indices, topk_weights = self.gate(flat)
        topk_weights = topk_weights.to(flat.dtype)

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


class _GlmMoeDsaDecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, device, dtype, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        eps = config.attrs.get("rms_norm_eps", 1e-5)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=eps, dtype=dtype)
        self.self_attn = _GlmMoeDsaMLA(config, device, dtype, layer_id)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=eps, dtype=dtype)
        is_dense = (not config.is_moe) or layer_id < int(config.first_k_dense_replace or 0)
        self.mlp = (
            _GlmMoeDsaMLP(config.hidden_size, config.intermediate_size, dtype)
            if is_dense
            else _GlmMoeDsaMoE(config, device, dtype, layer_id)
        )

    def forward(self, hidden_states, positions, table_idx, ctx, batch, prev_topk_indices=None):
        residual = hidden_states
        attn_out, topk_indices = self.self_attn(
            self.input_layernorm(hidden_states), positions, table_idx, ctx, batch, prev_topk_indices=prev_topk_indices
        )
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, topk_indices


class GlmMoeDsaForCausalLM(nn.Module):
    """GLM-5.2: real forward pass for the Intel engine loop (``#14``)."""

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
            _GlmMoeDsaDecoderLayer(config, device, dtype, layer_id=i) for i in range(num_layers)
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
            topk_indices = None  # cross-layer DSA top-k chain, per request
            for layer in self.layers:
                h, topk_indices = layer(h, positions[token_slice], req.table_idx, ctx, batch, prev_topk_indices=topk_indices)
            out[i] = self.norm(h)[-1]
            offset += ext

        return self.lm_head(out)


__all__ = ["parse_config", "iter_weights", "GlmMoeDsaForCausalLM"]
