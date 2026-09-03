"""DeepSeek-V4 -- Intel Arc Pro B70 port.

Upstream NVIDIA path: python/freetoken/models/deepseek_v4/
Fill in: GitHub issue `models-dsv4` (see docs/architecture.md).

**Current scope: Multi-head Latent Attention (MLA) only** (issue
`models-mla`, #190, child of the DeepSeek-V4 epic #21). DeepSeek Sparse
Attention (the trained top-k indexer sitting on top of MLA, #191) and the
real MoE router / full model wiring (#192) are deliberate, separate
follow-up issues -- every layer here runs a plain dense MLP so this
module's own accept bar (a real MLA forward, end to end, numerically
sound) is provable in isolation, without DSA's or the router's own
correctness questions in the same PR.

Real forward-pass math grounded directly against HF transformers' real
``modeling_deepseek_v3.py`` (fetched and read this session, not guessed):

* **Compression**: the query goes through its own low-rank compression
  (``q_a_proj`` -> ``q_a_layernorm`` -> ``q_b_proj``) when ``q_lora_rank``
  is set (the real V3/V4 case); K/V share ONE projection
  (``kv_a_proj_with_mqa``, output size ``kv_lora_rank + qk_rope_head_dim``)
  that produces the compressed KV latent AND the shared RoPE key in one
  shot -- the RoPE key slice is NOT layernormed and is NEVER passed through
  ``kv_b_proj``; only the ``kv_lora_rank`` slice is (``kv_a_layernorm``).
* **What gets cached**: exactly the POST-layernorm compressed KV latent
  (``kv_lora_rank`` elements) concatenated with the un-rotated RoPE key
  slice (``qk_rope_head_dim`` elements) -- ONE shared "head" cached once,
  not per-attention-head. This port's existing paged KV pool
  (``kvcache/base.py``) already stores exactly ``[num_kv_heads, head_dim]``
  per token with NO other assumption about what those dims mean, so this
  is configured as ``num_key_value_heads=1``,
  ``head_dim=kv_lora_rank+qk_rope_head_dim`` -- reusing the existing pool
  unmodified rather than building a new cache structure, a real, direct
  memory-shape win (576 vs. the ~32K elements/token a real V3-shaped
  config's fully-materialized per-head K/V would need) even though this
  port's V storage half goes unused (documented below).
* **Decompression** (the eager/explicit path, NOT the "absorption" trick
  some inference stacks use to avoid ever materializing full per-head K/V
  -- deliberately the simpler, more obviously-correct version for a first
  implementation): every forward call, the cached compressed latent is
  read back and ``kv_b_proj``-expanded into real per-head ``k_nope``/
  ``value``; the cached RoPE key is broadcast (not projected) across every
  head and concatenated with ``k_nope``.
* **Softmax scale**: ``1/sqrt(qk_nope_head_dim + qk_rope_head_dim)`` (NOT
  ``head_dim`` in the GQA sense -- V has a different width than QK here).

Deliberate scope cut on the KV pool: ``write_kv``/``read_kv`` require K and
V to share one shape (``kvcache/base.py``'s own ``[L, S, H, D]`` buffer
pair) -- MLA has nothing meaningful to put in the V slot (the compressed
latent IS the cached content; V is derived from it at decompress time),
so this stores a zeroed dummy in the V buffer and only ever reads the K
buffer back. A dedicated MLA-shaped cache (storing the compressed latent
once instead of paying for an unused, same-sized V buffer) is real,
separable follow-up, not blocking this issue's own accept bar (a real,
numerically-correct MLA forward).

This module bypasses the shared ``attention/`` backend machinery entirely
(``ctx.attn_backend.forward()``) -- MLA's decompress-then-attend shape
does not fit that contract (which assumes the stored K/V IS what gets
dot-producted against Q) -- and calls ``kv_cache.read_kv``/``write_kv``
directly, mirroring how ``qwen3_5_moe``'s ``_GatedDeltaNet`` (linear
attention) already bypasses the same machinery for the same reason.

**DeepSeek Sparse Attention (DSA)** (issue `models-dsa`, #191, on top of
MLA above): grounded directly against HF transformers' real
``modeling_deepseek_v32.py`` (DeepSeek-V3.2-Exp -- fetched and read this
session, not guessed). Active only when the checkpoint's config declares
``index_topk`` (real DeepSeek-V3.2/V4 field); every #190-only checkpoint
with no such field is completely unaffected (dense causal MLA, unchanged).

* **The indexer**: a small, separately-trained module per layer --
  ``wq_b`` (reads the SAME compressed query residual MLA's own
  ``q_a_layernorm(q_a_proj(hidden_states))`` already computes, not a
  separate query path), ``wk`` + ``k_norm`` (one shared "head", MQA-style,
  mirroring MLA's own K compression shape), ``weights_proj`` (combines
  ``index_n_heads`` indexer heads into one score). Real math:
  ``scores = relu((q_idx @ k_idx^T) * head_dim**-0.5)``,
  ``index_scores = (weights_proj(hidden) * n_heads**-0.5) @ scores``.
* **Indexer RoPE**: non-interleaved (half-split ``rotate_half``, the exact
  convention this port's own MLA/Qwen3.5/GLM4 RoPE already uses) over the
  same ``qk_rope_head_dim`` width as MLA's own rotary portion -- the real
  modeling file explicitly notes this differs from ITS OWN main-attention
  RoPE (which uses an interleaved-pairs convention this port has never
  needed, so nothing to port there).
* **Indexer cache -- a deliberate, novel-to-this-port reuse, not what the
  real reference does**: the real HF reference caches the indexer key in
  its own dedicated buffer (and, per an explicit ``# TODO`` in that same
  real file, currently gives up on MLA's compressed-cache optimization
  entirely once DSA is active). This port does neither: MLA's own pool
  slot already carries an unused, same-shaped, all-zero V buffer (see
  above) -- the indexer key (``index_head_dim`` <= the pool's
  ``kv_lora_rank + qk_rope_head_dim`` row width for every real DeepSeek-V3/
  V3.2 config) is stored there instead of thrown away, zero new kvcache
  plumbing, MLA's own compressed-cache win fully preserved. A real
  engineering choice, not a reference-matching one -- documented here so
  it is never mistaken for the real upstream cache design.
* **Top-k masking**: ``topk = min(index_topk, written)`` (naturally a
  no-op once the real sequence is shorter than ``index_topk``); the
  resulting sparse boolean mask is ANDed with the existing causal mask
  (layered ON TOP, never a replacement).
* **Open question, explicitly flagged by the research this was grounded
  against, not resolved here**: GLM-5.2's own real config additionally
  carries ``indexer_types``/``index_topk_freq``/``index_skip_topk_offset``
  fields with NO corresponding logic anywhere in the real DeepSeek
  reference this was grounded against -- they appear to be a GLM-5.2-
  specific "shared indexer across layers" extension. This module
  implements DeepSeek's own real, authoritative "full indexer every
  layer" behavior only; GLM-5.2's own alternation semantics are a
  separate, not-yet-researched follow-up for whoever wires
  ``glm_moe_dsa`` (#22).
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


def parse_config(hf_config, model_path: str | None = None, **_kwargs) -> ModelConfig:
    """Build a :class:`ModelConfig` from a real HF ``DeepseekV3Config``-shaped
    config (DeepSeek-V4 is assumed to keep the same field names -- V3 is the
    real, documented source this was grounded against).

    ``config.num_key_value_heads``/``config.head_dim`` are deliberately set
    to the KV-POOL's storage shape (``1`` / ``kv_lora_rank +
    qk_rope_head_dim``), NOT the real attention head count -- see this
    module's own docstring. The real head count lives in
    ``config.num_attention_heads`` (read directly by the attention class,
    which never asks the pool about it) and is unaffected.
    """
    # Prefer direct attribute access over to_dict(): the installed
    # transformers' real DeepseekV4Config.to_dict() silently drops
    # intermediate_size (confirmed directly -- getattr has it, to_dict()
    # doesn't), so to_dict() alone is not a reliable source of truth here,
    # unlike every other model package in this port.
    raw = hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(hf_config)

    def field(name, default=None):
        val = getattr(hf_config, name, None)
        return val if val is not None else raw.get(name, default)

    src = {k: field(k) for k in (
        "hidden_size", "vocab_size", "num_hidden_layers", "num_attention_heads",
        "q_lora_rank", "kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim",
        "v_head_dim", "attention_bias", "intermediate_size",
        "max_position_embeddings", "tie_word_embeddings", "rope_theta",
        "rope_scaling", "hidden_act", "torch_dtype", "dtype", "rms_norm_eps",
        "index_topk", "index_head_dim", "index_n_heads",
    )}
    kv_lora_rank = int(src.get("kv_lora_rank") or 0)
    qk_rope_head_dim = int(src.get("qk_rope_head_dim") or 0)
    cfg = ModelConfig(
        architectures=["DeepseekV4ForCausalLM"],
        hidden_size=src.get("hidden_size"),
        vocab_size=src.get("vocab_size"),
        num_layers=src.get("num_hidden_layers"),
        num_attention_heads=src.get("num_attention_heads"),
        num_key_value_heads=1,  # MLA: one shared compressed latent, not per-head K/V
        head_dim=kv_lora_rank + qk_rope_head_dim,  # the pool's real per-token row width
        intermediate_size=src.get("intermediate_size"),
        max_position_embeddings=src.get("max_position_embeddings"),
        tie_word_embeddings=src.get("tie_word_embeddings", False),
        kv_lora_rank=kv_lora_rank or None,
        q_lora_rank=int(src.get("q_lora_rank")) if src.get("q_lora_rank") else None,
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
    # DSA (issue #191): only set when the checkpoint's own config.json FILE
    # explicitly declares index_topk. Neither getattr(hf_config, ...) NOR
    # hf_config.to_dict() are reliable "was it explicit" signals here:
    # confirmed directly, the installed transformers' real DeepseekV4Config
    # class defaults index_topk to a real non-None value (512), and
    # to_dict() serializes that default too -- both would silently turn
    # DSA "on" for a plain-MLA checkpoint whose config.json never mentions
    # it at all, breaking the "every #190-only checkpoint is unaffected"
    # guarantee this gate exists for. Read the actual JSON file instead
    # (mirrors qwen3_moe's own _probe_head_dim, which reads raw checkpoint
    # bytes for the same class-of-reason: a parsed config object's
    # attribute access isn't trustworthy for this specific question).
    # Without a model_path (a mock/unit-test hf_config with no backing
    # file -- every model package's own test suite uses this shape) fall
    # back to the plain to_dict() output, which is NOT polluted for a
    # hand-built mock object (only a real HF config CLASS has non-None
    # defaults to leak).
    file_raw = _raw_checkpoint_json(model_path) if model_path else raw
    if file_raw.get("index_topk"):
        cfg.attrs["index_topk"] = int(file_raw["index_topk"])
        cfg.attrs["index_head_dim"] = int(file_raw.get("index_head_dim") or raw.get("index_head_dim") or 0)
        cfg.attrs["index_n_heads"] = int(file_raw.get("index_n_heads") or raw.get("index_n_heads") or 1)
    return cfg


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


def iter_weights(model_path: str, device: torch.device, **_kwargs):
    """Yield the checkpoint's tensors, each placed on ``device``.

    Issue #190's own scope (MLA only, dense MLP every layer, see this
    module's own docstring): no MoE expert routing at all yet -- every
    tensor in this issue's checkpoints is dense. #192 (the real model
    wiring issue) adds the expert-routing split back in when the real MoE
    router lands.
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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class _DeepseekV4MLA(nn.Module):
    """Multi-head Latent Attention (issue `models-mla`, #190). See this
    module's own docstring for the full real-math grounding."""

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

        # DSA (issue #191): built only when the checkpoint's real config
        # sets index_topk -- see this module's own docstring for the full
        # real-math grounding and the deliberate V-buffer cache reuse.
        self.index_topk = config.attrs.get("index_topk")
        if self.index_topk:
            self.index_head_dim = config.attrs["index_head_dim"]
            self.index_n_heads = config.attrs["index_n_heads"]
            pool_row_width = self.kv_lora_rank + self.qk_rope_head_dim
            if self.index_head_dim > pool_row_width:
                raise ValueError(
                    f"index_head_dim ({self.index_head_dim}) exceeds this pool's row "
                    f"width ({pool_row_width}) -- the V-buffer reuse this port's DSA "
                    "relies on (see module docstring) needs the indexer key to fit "
                    "inside MLA's own (otherwise-unused) V slot."
                )
            q_resid_width = self.q_lora_rank if self.q_lora_rank else hidden
            self.wq_b = nn.Linear(q_resid_width, self.num_heads * self.index_head_dim, bias=False, dtype=dtype)
            self.wk = nn.Linear(hidden, self.index_head_dim, bias=False, dtype=dtype)
            self.indexer_k_norm = nn.LayerNorm(self.index_head_dim, eps=eps, dtype=dtype)
            self.weights_proj = nn.Linear(hidden, self.num_heads, bias=False, dtype=dtype)
            self.index_scale = self.index_head_dim ** -0.5

    def forward(self, hidden_states, positions, table_idx, ctx, batch):
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
        cos, sin = emb.cos(), emb.sin()
        q_rot_f, k_rot_f = q_rot.to(torch.float32), k_rot.to(torch.float32).unsqueeze(1)  # k_rot: [T, 1, rope]
        q_rot = (q_rot_f * cos[:, None, :] + _rotate_half(q_rot_f) * sin[:, None, :]).to(q.dtype)
        k_rot = (k_rot_f * cos[:, None, :] + _rotate_half(k_rot_f) * sin[:, None, :]).to(q.dtype).squeeze(1)  # [T, rope]

        # Cache the compressed representation exactly as computed (post-norm
        # latent + rotated rope key), one shared "head" -- see this module's
        # own docstring for why num_kv_heads=1 here reuses the existing pool
        # unmodified. write_kv wants head-major [num_kv_heads, T, head_dim].
        cache_row = torch.cat([kv_latent, k_rot], dim=-1)  # [T, kv_lora_rank + rope]
        k_for_pool = cache_row.unsqueeze(0)  # [1, T, D] -- head-major, 1 head
        pool_row_width = k_for_pool.shape[-1]

        if self.index_topk:
            # DSA indexer key (issue #191): its own small wk projection +
            # LayerNorm, then the SAME rope (same qk_rope_head_dim slice,
            # same cos/sin already computed above for MLA) applied to just
            # the indexer key's own rope portion. Stored in MLA's own
            # otherwise-unused V buffer slot (padded with zeros past
            # index_head_dim) -- see module docstring.
            k_idx = self.indexer_k_norm(self.wk(hidden_states))  # [T, index_head_dim]
            k_idx_rot, k_idx_pass = k_idx.split([self.qk_rope_head_dim, self.index_head_dim - self.qk_rope_head_dim], dim=-1)
            k_idx_rot_f = k_idx_rot.to(torch.float32)
            k_idx_rot = (k_idx_rot_f * cos + _rotate_half(k_idx_rot_f) * sin).to(k_idx.dtype)
            k_idx = torch.cat([k_idx_rot, k_idx_pass], dim=-1)  # [T, index_head_dim]
            v_for_pool = F.pad(k_idx, (0, pool_row_width - self.index_head_dim)).unsqueeze(0)
        else:
            v_for_pool = torch.zeros_like(k_for_pool)  # V half is unused for plain MLA -- see docstring
        ctx.kv_cache.write_kv(k_for_pool, v_for_pool, positions, self.layer_id)

        written = req_written_len(ctx, batch, table_idx)
        read_pos = torch.arange(written, device=hidden_states.device)
        cached_tok, cached_v_tok = ctx.kv_cache.read_kv(table_idx, read_pos, self.layer_id)  # [written, 1, D] each
        cached = cached_tok.squeeze(1)  # [written, kv_lora_rank + rope]
        hist_latent, hist_k_rot = cached.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        index_mask = None
        if self.index_topk:
            hist_k_idx = cached_v_tok.squeeze(1)[:, : self.index_head_dim]  # [written, index_head_dim]
            q_idx = self.wq_b(q_resid if q_resid is not None else hidden_states).view(T, self.num_heads, self.index_head_dim)
            q_idx_rot, q_idx_pass = q_idx.split([self.qk_rope_head_dim, self.index_head_dim - self.qk_rope_head_dim], dim=-1)
            q_idx_rot_f = q_idx_rot.to(torch.float32)
            q_idx_rot = (q_idx_rot_f * cos[:, None, :] + _rotate_half(q_idx_rot_f) * sin[:, None, :]).to(q_idx.dtype)
            q_idx = torch.cat([q_idx_rot, q_idx_pass], dim=-1)  # [T, heads, index_head_dim]

            # scores = relu((q_idx . k_idx) * scale) per (query, head, key);
            # weights_proj combines the index_n_heads-worth of per-head
            # scores into one score per (query, key) -- real math, see
            # module docstring.
            raw = torch.einsum("thd,kd->htk", q_idx.float(), hist_k_idx.float()) * self.index_scale
            raw = F.relu(raw)  # [heads, T, written]
            w = self.weights_proj(hidden_states).float() * (self.num_heads ** -0.5)  # [T, heads]
            index_scores = torch.einsum("th,htk->tk", w, raw)  # [T, written]
            # Causal-mask BEFORE top-k: a query must never select a future
            # key (an un-masked pick that the later AND-with-causal-mask
            # step would exclude anyway) -- without this, a query whose
            # top-index_topk scores happen to all be future positions ends
            # up with an all-excluded (all -inf softmax) row, a real NaN
            # bug, not just a suboptimal selection. Guarantees every
            # query's own position is always a selectable (score, not
            # necessarily chosen) candidate.
            causal_idx = positions[:, None] >= read_pos[None, :]
            index_scores = index_scores.masked_fill(~causal_idx, float("-inf"))

            topk = min(self.index_topk, written)
            _, topk_idx = torch.topk(index_scores, topk, dim=-1)  # [T, topk]
            index_mask = torch.zeros(T, written, dtype=torch.bool, device=hidden_states.device)
            index_mask.scatter_(1, topk_idx, True)

        # Explicit (non-absorbed) decompression: expand the shared latent
        # into real per-head k_nope/value, then broadcast the (already-
        # rotated, shared) rope key across every head.
        expanded = self.kv_b_proj(hist_latent).view(written, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        hist_k_nope, hist_v = expanded.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        hist_k_rot_expanded = hist_k_rot[:, None, :].expand(written, self.num_heads, self.qk_rope_head_dim)
        hist_k = torch.cat([hist_k_nope, hist_k_rot_expanded], dim=-1)  # [written, heads, qk_head_dim]

        q_full = torch.cat([q_pass, q_rot], dim=-1)  # [T, heads, qk_head_dim]
        scores = torch.einsum("thd,khd->htk", q_full, hist_k) * self.scale  # [heads, T, written]
        q_pos = positions
        key_pos = read_pos
        allowed = q_pos[None, :, None] >= key_pos[None, None, :]
        if index_mask is not None:
            # DSA: the sparse top-k mask is ANDed onto the causal mask
            # (layered on top, never a replacement) -- real math, see
            # module docstring.
            allowed = allowed & index_mask[None, :, :]
        scores = torch.where(allowed, scores, torch.full_like(scores, float("-inf")))
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("htk,khd->thd", probs, hist_v)  # [T, heads, v_head_dim]
        return self.o_proj(out.reshape(T, -1))


def req_written_len(ctx, batch, table_idx: int) -> int:
    """How much of this request's history is resident in the pool by the
    time this forward call needs to read it back -- mirrors
    ``attention/triton.py``'s own ``_attend_one`` logic exactly: a decode
    step reads the full history; a prefill step reads its already-resident
    prefix plus the chunk it is extending by."""
    req = next((r for r in batch.reqs if r.table_idx == table_idx), batch.reqs[0])
    is_decode = batch.phase == "decode"
    return req.device_len if is_decode else req.cached_len + req.extend_len


class _DeepseekV4MLP(nn.Module):
    """Plain SwiGLU MLP -- every layer's feed-forward block for this
    issue's own scope (no MoE router yet, see this module's own
    docstring)."""

    def __init__(self, hidden_size: int, intermediate_size: int, dtype) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _DeepseekV4DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, device, dtype, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        eps = config.attrs.get("rms_norm_eps", 1e-6)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=eps, dtype=dtype)
        self.self_attn = _DeepseekV4MLA(config, device, dtype, layer_id)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=eps, dtype=dtype)
        self.mlp = _DeepseekV4MLP(config.hidden_size, config.intermediate_size, dtype)

    def forward(self, hidden_states, positions, table_idx, ctx, batch):
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), positions, table_idx, ctx, batch)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class DeepseekV4ForCausalLM(nn.Module):
    """DeepSeek-V4, MLA-only scope (issue #190). Subclasses ``nn.Module``
    directly so its parameters are real registered ``nn.Parameter``s."""

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
            _DeepseekV4DecoderLayer(config, device, dtype, layer_id=i) for i in range(num_layers)
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


__all__ = ["parse_config", "iter_weights", "DeepseekV4ForCausalLM"]
