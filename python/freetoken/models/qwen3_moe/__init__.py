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


def parse_config(hf_config, use_offload_moe: bool = False) -> ModelConfig:
    """Build a :class:`ModelConfig` from a HF Qwen3-MoE config.

    ``hf_config`` is the lru-cached object shared across callers, so it is
    copied (``to_dict``) before the parsed fields are derived -- never mutated.

    ``use_offload_moe`` (ADR 0002) is *not* a checkpoint field: the loader /
    engine set it from the ``moe_backend`` choice. When True the MoE experts are
    never XPU-resident and are streamed from host RAM through the LRU slot pool
    during the forward pass.
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
    # ADR 0002: flag the host-offload MoE path (off by default = in-VRAM experts).
    cfg.use_offload_moe = use_offload_moe
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
        # The q/k/v/o projections are the pure-torch tensor-parallel Linear port
        # (issue-24 WP6), not ``nn.Linear``. On the B70 (TP=1) the TP-aware classes
        # reduce to a plain ``x @ w.T`` matmul -- the identical math ``nn.Linear``
        # runs -- but as ``nn.Parameter``-backed ``BaseOP``s they stay drop-in for
        # the checkpoint loader (``named_parameters()`` + ``.to(device)``) and drop
        # the CUDA JIT / NCCL dependencies upstream's Linear carries. Shapes match
        # ``nn.Linear`` exactly: q is ``[heads*head_dim, hidden]``, k/v
        # ``[kv*head_dim, hidden]`` (``[out, in]``), o ``[heads*head_dim, hidden]``.
        from freetoken.layers import LinearOProj, LinearReplicated

        self.q_proj = LinearReplicated(config.hidden_size, self.num_heads * self.head_dim, has_bias=False)
        self.k_proj = LinearReplicated(config.hidden_size, self.num_kv_heads * self.head_dim, has_bias=False)
        self.v_proj = LinearReplicated(config.hidden_size, self.num_kv_heads * self.head_dim, has_bias=False)
        self.o_proj = LinearOProj(self.num_heads * self.head_dim, config.hidden_size, has_bias=False)
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
        # ``hidden_states`` is *this request's* hidden slice -- the decoder layer
        # runs each request's layers on its own token rows (hidden[token_slice]),
        # so the slice length is this request's new-token count, NOT the
        # whole batch's token count. Project that slice (bsz = its length), not
        # the whole batch, and write *only those* K/V rows to the pool: a
        # whole-batch projection would write every request's K/V into this
        # request's out_loc (a cross-request KV corruption). This is why the
        # prefill batch is correct on XPU -- the pool row count matches the
        # number of out_loc entries this request actually writes.
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
        # Append this request's K/V to the pool. ``positions`` here is this
        # request's token positions (the decoder layer passed
        # positions[token_slice]); the new token's out_loc slot equals its
        # absolute position under the identity page table, so index out_loc by
        # this request's positions rather than the whole-batch out_loc.
        ctx.kv_cache.write_kv(k, v, positions)
        # ``table_idx`` identifies *this* request (the decoder layer runs each
        # request's layers on its own hidden slice, so q/k/v hold only this
        # request's new tokens, not the whole batch's). The backend uses it to
        # read this request's KV history from the pool and to interpret the
        # per-request q/k/v; without it a backend cannot tell which request is
        # 'current' when a step mixes multiple requests.
        out = ctx.attn_backend.forward(q, k, v, self.layer_id, batch, table_idx=table_idx)
        return self.o_proj(out.transpose(1, 2).reshape(bsz, -1))


def _expert_compute(gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Run one expert on a [t, H] input using *detached* projection weights.

    The expert weights live in the host-offload banks (ADR 0002) and are handed
    in as plain tensors (views of the XPU slot pool), so this is a hand-rolled
    SwiGLU -- not an ``nn.Linear`` -- over the gathered per-expert input:
    ``down(silu(gate(x)) * up(x))``. The bank rows are stored in *weight*
    orientation (``[out, in]``, matching ``nn.Linear.weight``): gate/up are
    ``[I, H]`` and down is ``[H, I]``. The projection is therefore
    ``x @ w.t()`` -- the same ``F.linear`` form the in-VRAM ``_Qwen3Expert``
    uses -- so the math is identical to the resident path, which the reference
    test compares against.
    """
    # gate_w [I, H], up_w [I, H], down_w [H, I]; x [t, H].
    # Every projection is ``x @ w.t()`` (F.linear form). The down step is
    # ``h @ down_w.t()`` (NOT ``down_w.t() @ h``): in this torch XPU build the
    # ``@`` operator requires ``left.cols == right.rows``, so the leading
    # ``[t, *]`` operand must be on the left. ``h @ down_w.t()`` is exactly
    # ``F.linear(h, down_w)`` -- the in-VRAM expert's down projection.
    inter = gate_w.shape[0]
    return (F.silu(x @ gate_w.t()) * (x @ up_w.t())) @ down_w.t()


class _Qwen3Expert(nn.Module):
    """A single MoE expert: gate/up/down projections (SwiGLU).

    Used by the in-VRAM path only (``use_offload_moe=False``). The offload path
    (ADR 0002) does not build these at all -- the experts live in host RAM and
    are read through the LRU slot pool instead.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _Qwen3MoE(nn.Module):
    """Mixture-of-experts block: router + N experts.

    Two paths (ADR 0002):

    * ``use_offload_moe=False`` (default): the N experts are XPU-resident
      ``_Qwen3Expert`` modules; the forward gathers each token's routed experts
      from them.
    * ``use_offload_moe=True``: the experts are *never* XPU-resident. Only the
      router is built; at each step the routed expert ids are routed through the
      engine's ``OffloadMoeCache`` (host banks -> small XPU LRU slot pool), the
      missed experts are streamed from host, and each routed token runs the
      selected expert through the slot weights.
    """

    def __init__(self, config, device, dtype) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.use_offload = bool(getattr(config, "use_offload_moe", False))
        # Router: pure-torch replicated Linear (issue-24 WP6). ``num_experts`` is
        # small and fully replicated, so ``LinearReplicated`` is the right class.
        from freetoken.layers import LinearReplicated

        self.gate = LinearReplicated(config.hidden_size, self.num_experts, has_bias=False)
        if self.use_offload:
            # ADR 0002: no XPU-resident expert params. The routed experts are
            # read from the LRU slot pool the loader attaches to the model
            # (``model.moe_cache`` + ``model.moe_layer_id``); the host banks are
            # the source of truth. (``self.experts`` is left unset -- the loader
            # never copies into expert modules on this path.)
            self.experts = None
        else:
            self.experts = nn.ModuleList(_Qwen3Expert(config).to(device, dtype) for _ in range(self.num_experts))

    def forward(self, hidden_states: torch.Tensor, model=None, batch=None) -> torch.Tensor:
        # The engine feeds a *token-major* 2-D slice [num_tokens, hidden]
        # (one request at a time), so we must not assume a [bsz, seq, hidden]
        # batch dim. Flatten to [T, hidden] and restore the same shape on the way out.
        in_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, in_shape[-1])  # [T, hidden]
        routing = self.gate(flat)  # [T, num_experts]
        gate_log = F.softmax(routing, dim=-1)
        top_w, top_idx = torch.topk(gate_log, self.top_k, dim=-1)  # [T, k]
        top_w = (top_w / top_w.sum(dim=-1, keepdim=True)).to(flat.dtype)

        if self.use_offload:
            return self._forward_offload(flat, top_idx, top_w, model, batch).view(in_shape)

        out = torch.zeros_like(flat)
        # Per-expert gather: route each expert's tokens in one matmul each.
        for e in range(self.num_experts):
            for slot in range(self.top_k):
                sel = (top_idx[:, slot] == e)
                if not sel.any():
                    continue
                out[sel] += top_w[sel, slot, None] * self.experts[e](flat[sel])
        return out.view(in_shape)

    def _forward_offload(self, flat, top_idx, top_w, model, batch) -> torch.Tensor:
        """Serve the routed experts through the host-offload LRU slot pool.

        The pool is a single global timestamp LRU shared by every MoE layer
        (ADR 0002): a prefill materializes the *whole* layer into the pool
        (evicting the LRU-resident experts of other layers), and a decode step
        streams in only the *missed* routed experts, each into an evicted slot.
        After :meth:`ensure_experts` / :meth:`materialize_layer` the ``top_idx``
        tensor holds *slot* ids, so each routed token runs the expert the slot
        currently holds -- which, after ``copy_missing``, is exactly the routed
        expert (the forward indexes the slot cache, not the layer bank).
        """
        layer_id = model.moe_layer_id[self.layer_id]
        cache = model.moe_cache
        # A prefill must materialize the *whole* MoE layer into the LRU pool
        # (evicting other layers' resident experts); a decode streams in only the
        # missed routed experts, one at a time. The engine tags the prompt step
        # ``phase="prefill"`` and later steps ``phase="decode"`` (engine.step),
        # so ``batch.is_prefill`` is the phase signal; the ``flat.shape[0] > 1``
        # check is a phase-independent fallback (prefill >1 token, decode == 1).
        is_prefill = (batch is not None and batch.is_prefill) or flat.shape[0] > 1
        is_xpu = bool(getattr(cache, "is_xpu", False))
        if is_prefill:
            # materialize_layer rewrites top_idx to slot ids in place (same
            # contract as ensure_experts), so prefill and decode index the
            # slot pool identically.
            cache.materialize_layer(layer_id, top_idx)
        else:
            cache.ensure_experts(layer_id, top_idx)
        cache.copy_missing()

        # bank_views() returns a tuple indexed by bank registration order
        # ("gate_up" first, "down" second); index the pool (S slots) and pick
        # the per-slot row by slot id.
        gu, dn = cache.bank_views()  # ([S, 2I, H], [S, H, I])
        S = gu.shape[0]
        intermediate = int(model.config.moe_intermediate_size)

        out = torch.zeros_like(flat)
        k = top_idx.shape[1]
        for slot_pos in range(k):
            routed = top_idx[:, slot_pos]  # [B] slot ids (after ensure_experts)
            # Mask out-of-range slot ids too (>= S): a stale/recycled id from
            # ensure_experts/copy_missing would index gu/dn out of bounds and
            # abort the (shared) XPU kernel, so clamp it out of the gather.
            valid = (routed >= 0) & (routed < S)
            if not valid.any():
                continue
            if is_xpu:
                if (routed >= S).any() and (routed >= 0).any():
                    raise IndexError(
                        f"layer {self.layer_id}: offload routed slot id "
                        f"{int(routed.max())} >= cache_size {S}: stale slot id "
                        "(ensure_experts/copy_missing desync)"
                    )
            s = routed[valid]  # the slot each token's expert landed in
            # Dedup on the host (the routed set per step is tiny): the XPU's
            # torch.unique runs an async sort-scatter that aborts (Indexing.h
            # "index out of bounds") when a slot id races the pool; a CPU
            # unique + CPU->XPU gather is deterministic and identical in math.
            for s_i in torch.unique(s.cpu()).tolist():
                sub = valid & (routed == s_i)
                out[sub] += top_w[sub, slot_pos, None] * _expert_compute(
                    gu[s_i, 0:intermediate],
                    gu[s_i, intermediate : 2 * intermediate],
                    dn[s_i],
                    flat[sub],
                )
        return out


class _Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, device, dtype, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = nn.RMSNorm(config.hidden_size)
        self.self_attn = _Qwen3Attention(config, device, dtype, layer_id)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size)
        self.mlp = _Qwen3MoE(config, device, dtype)
        # The offload path must know *which* MoE layer this block serves (the
        # slot pool is indexed by layer id, and the loader maps layer id ->
        # MoE-layer index). Dense layers have no mlp cache to index.
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
        # Pure-torch replicated Linear (issue-24 WP6); weight ``[vocab, hidden]``,
        # the same shape the checkpoint's ``lm_head.weight`` carries.
        from freetoken.layers import LinearReplicated

        self.lm_head = LinearReplicated(hidden_size, vocab_size, has_bias=False)

        # ADR 0002: when the engine picks the host-offload MoE backend the
        # experts are never XPU-resident. The MoE blocks then hold *only* the
        # router (no expert ``nn.Linear`` params); the routed experts are read
        # from the LRU slot pool the loader attaches to ``self`` before the
        # engine runs (self.moe_cache / self.moe_layer_id / self.ctx).
        if bool(getattr(config, "use_offload_moe", False)) and bool(getattr(config, "is_moe", False)):
            self.moe_offload = True
        else:
            self.moe_offload = False
        self.moe_cache = None
        self.moe_layer_id = None
        # The modules above (pure-torch Linear / nn.RMSNorm) are registered on
        # the CPU, and the loader fills them in place without moving them -- so
        # without
        # this the dense weights (norms, projections, lm_head) would stay on the
        # CPU while the engine feeds XPU activations, and the first forward would
        # hit a device mismatch. Move the whole module to its target device here
        # (a no-op for the CPU reference path). The host-offload path is
        # unaffected: parse_config never sets use_offload_moe, so this branch
        # only moves in-VRAM models, whose expert params are not built here.
        if self.device.type != "cpu":
            self.to(self.device)

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
        num_tokens = input_ids.shape[0]

        hidden = self.embed_tokens(input_ids)  # [num_tokens, hidden]
        out = torch.empty((batch.size, self.config.hidden_size), device=hidden.device, dtype=hidden.dtype)

        offset = 0
        # Per-request token counts, in request order. The engine sets these in
        # step() (extend_len for a request still prefilling its prompt, 1 once
        # it has entered decode) so a step that mixes phases sizes each
        # request's slice by its own count -- a single batch-level phase flag
        # would over/under-slice in a mixed batch. When a caller doesn't
        # populate them (it never does via the engine), fall back to the old
        # global-phase heuristic.
        extend_lens = batch.extend_lens
        if extend_lens is None:
            prefill = batch.is_prefill or (num_tokens > batch.size)
            extend_lens = [req.extend_len if prefill else 1 for req in reqs]
        for i, req in enumerate(reqs):
            ext = int(extend_lens[i])
            token_slice = slice(offset, offset + ext)
            h = hidden[token_slice]
            for layer in self.layers:
                h = layer(h, positions[token_slice], req.table_idx, ctx, batch)
            # Keep only the last position of this request (next-token logits).
            out[i] = self.norm(h)[-1]
            offset += ext

        return self.lm_head(out)


__all__ = ["parse_config", "iter_weights", "Qwen3MoeForCausalLM"]
