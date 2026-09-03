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

    The attention projections store ``q_proj`` as ``[heads*head_dim, hidden]``
    and ``o_proj`` as ``[hidden, heads*head_dim]`` (both ``[out, in]``), so
    ``o_proj``'s *second* dim is ``heads*head_dim`` -- divided by the head count
    that is the true per-head dim. (Reading ``o_proj``'s first dim would give
    ``hidden``, which is ``heads*head_dim`` only when head_dim == hidden/heads.)
    This is the one source that is always right for extended-head MoEs (Qwen3.6/
    3.8), whose ``head_dim`` != ``hidden // heads`` -- and whose config may or
    may not set ``head_dim`` explicitly. Returns ``None`` when it can't resolve a
    shape (the caller then falls back to the config field, then to deriving).
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
            # o_proj is [hidden, heads*head_dim]; the head-agg axis is dim 1.
            heads_times_head_dim = f.get_slice(name).get_shape()[1]
        if heads_times_head_dim % num_heads:
            return None
        return heads_times_head_dim // num_heads
    except Exception:
        return None


def parse_config(
    hf_config,
    use_offload_moe: bool = False,
    use_cpu_moe: bool = False,
    moe_cpu_layers: str | None = None,
    use_hybrid: bool = False,
    model_path: str | None = None,
) -> ModelConfig:
    """Build a :class:`ModelConfig` from a HF Qwen3-MoE config.

    ``hf_config`` is the lru-cached object shared across callers, so it is
    copied (``to_dict``) before the parsed fields are derived -- never mutated.

    ``use_offload_moe`` (ADR 0002) is *not* a checkpoint field: the loader /
    engine set it from the ``moe_backend`` choice. When True the MoE experts are
    never XPU-resident and are streamed from host RAM through the LRU slot pool
    during the forward pass. ``use_cpu_moe`` (issue #8) likewise flags the
    backend (but True means the expert GEMM runs *on* the host, not streamed).
    ``use_hybrid`` (issue #9) flags the hybrid backend: each decode step splits
    its routed-expert misses between PCIe-fetch (XPU GEMM) and host-CPU GEMM by
    the ``ft bench bw`` profile's per-format fetch fraction. ``moe_cpu_layers``
    (issue #8) is the ``--moe-cpu-layers`` spec, stored verbatim on the config
    and resolved to concrete layer indices at build time. ``model_path`` is the
    local checkpoint dir (or ``None``); it lets the probe above recover the real
    per-head dim from the checkpoint's ``o_proj`` shape.
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
        # Per-head dim. Recovered from the checkpoint's real o_proj shape (the
        # only trustworthy source -- see _probe_head_dim), else the config's
        # explicit ``head_dim``, else ``None`` ("derive": hidden // heads, which
        # is correct for the standard Qwen3-30B family). Extended-head MoEs
        # (Qwen3.6/3.8) carry head_dim != hidden//heads, so the derive alone
        # mis-sizes q/o_proj.
        head_dim=(
            _probe_head_dim(model_path, src.get("num_attention_heads"))
            or (int(src.get("head_dim")) if src.get("head_dim") else None)
        ),
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
    # Issue #8: flag the CPU MoE path (expert GEMM runs on the host), plus the
    # --moe-cpu-layers spec (stored verbatim; resolved at build time).
    cfg.use_cpu_moe = use_cpu_moe
    # Issue #9: flag the hybrid backend (per-step miss split, ADR 0002 + the
    # ``ft bench bw`` profile's q* fetch fraction).
    cfg.use_hybrid = use_hybrid
    cfg.moe_cpu_layers = moe_cpu_layers
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

    A ``tie_word_embeddings: true`` checkpoint (common on smaller / merged
    Qwen3-MoE models) ships no separate ``lm_head.weight`` key -- HF ties it to
    ``model.embed_tokens.weight`` at load time. Without replicating that tie
    here, ``lm_head.weight`` never receives a placed tensor and is left at its
    constructor-time ``torch.empty`` value (observed as all-zero in practice),
    which makes every logit zero and every decode step argmax to token 0 (a
    silent, checkpoint-shaped repeat-the-same-token failure, not a crash). When
    ``include_non_moe`` is requested and the stream never yields a
    ``lm_head.weight``, synthesize one from ``embed_tokens.weight`` for a
    checkpoint that declares the tie.
    """
    embed_tokens_weight = None
    saw_lm_head = False
    for name, tensor in iter_safetensors(model_path, device):
        is_expert = ".experts." in name
        if is_expert and not include_moe_experts:
            continue
        if not is_expert and not include_non_moe:
            continue
        # Dense -> destination device; experts -> host offload banks.
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


class _Qwen3Attention(nn.Module):
    """Qwen3 grouped-query attention (RoPE + q/k RMS-norm), KV-pool driven."""

    def __init__(self, config, device, dtype, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads or config.num_attention_heads
        # Per-head dim: the checkpoint's explicit ``head_dim`` when it is one
        # (extended-head MoEs like Qwen3.6, where head_dim != hidden//heads),
        # else derive it. Both the projection shapes and the RoPE inv_freq key
        # off this, so it must match the checkpoint's real per-head size.
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.num_heads
        # The q/k/v/o projections are the pure-torch tensor-parallel Linear port
        # (issue-24 WP6), not ``nn.Linear``. On the B70 (TP=1) the TP-aware classes
        # reduce to a plain ``x @ w.T`` matmul -- the identical math ``nn.Linear``
        # runs -- but as ``nn.Parameter``-backed ``BaseOP``s they stay drop-in for
        # the checkpoint loader (``named_parameters()`` + ``.to(device)``) and drop
        # the CUDA JIT / NCCL dependencies upstream's Linear carries. Shapes match
        # ``nn.Linear`` exactly: q is ``[heads*head_dim, hidden]``, k/v
        # ``[kv*head_dim, hidden]`` (``[out, in]``), o ``[heads*head_dim, hidden]``.
        from freetoken.layers import LinearOProj, LinearReplicated

        self.q_proj = LinearReplicated(config.hidden_size, self.num_heads * self.head_dim, has_bias=False, dtype=dtype)
        self.k_proj = LinearReplicated(config.hidden_size, self.num_kv_heads * self.head_dim, has_bias=False, dtype=dtype)
        self.v_proj = LinearReplicated(config.hidden_size, self.num_kv_heads * self.head_dim, has_bias=False, dtype=dtype)
        self.o_proj = LinearOProj(self.num_heads * self.head_dim, config.hidden_size, has_bias=False, dtype=dtype)
        self.q_norm = nn.RMSNorm(self.head_dim, dtype=dtype)
        self.k_norm = nn.RMSNorm(self.head_dim, dtype=dtype)
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
        # The projection output is token-major [bsz, heads*head_dim] (each row
        # is one token whose columns are the per-head slices concatenated). A
        # flat ``.view(heads, bsz, head_dim)`` reinterprets memory linearly and
        # therefore scrambles which token lands in which head -- it is
        # *bsz-dependent* (only accidentally correct when bsz == num_heads),
        # which is exactly what made chunked-prefill (bsz>1) diverge from
        # decode (bsz=1). Lay it out token-major first, then transpose to the
        # head-major [heads, bsz, head_dim] the rest of the pipeline (RoPE,
        # write_kv's token-major round-trip) expects.
        q = self.q_proj(hidden_states).view(bsz, self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(hidden_states).view(bsz, self.num_kv_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(hidden_states).view(bsz, self.num_kv_heads, self.head_dim).transpose(0, 1)
        q = self._rope(self.q_norm(q), positions)
        k = self._rope(self.k_norm(k), positions)
        # Append this request's K/V to the pool. ``positions`` here is this
        # request's token positions (the decoder layer passed
        # positions[token_slice]); the new token's out_loc slot equals its
        # absolute position under the identity page table, so index out_loc by
        # this request's positions rather than the whole-batch out_loc.
        ctx.kv_cache.write_kv(k, v, positions, self.layer_id)
        # ``table_idx`` identifies *this* request (the decoder layer runs each
        # request's layers on its own hidden slice, so q/k/v hold only this
        # request's new tokens, not the whole batch's). The backend uses it to
        # read this request's KV history from the pool and to interpret the
        # per-request q/k/v; without it a backend cannot tell which request is
        # 'current' when a step mixes multiple requests.
        out = ctx.attn_backend.forward(q, k, v, self.layer_id, batch, table_idx=table_idx)
        # ``out`` is head-major [heads, bsz, head_dim]: dim 0 is the head axis,
        # dim 1 is this request's new-token axis, dim 2 the head_dim. To feed
        # o_proj we need the *token-major* [bsz, heads, head_dim] token vector
        # (row t = concatenation over heads of head h's head_dim slice at token
        # t). That is achieved by moving the token axis (1) to the front, i.e.
        # swapping axes 0 and 1: transpose(0, 1) -> [bsz, heads, head_dim].
        # (The old transpose(1, 2) moved the *head_dim* axis to the front,
        # yielding [heads, head_dim, bsz]; on a non-contiguous view .reshape
        # then copied in raw head-first memory order, so for bsz > 1 each row
        # interleaved the wrong heads' tokens -- the o_proj input was silently
        # scrambled. bsz == 1 hid it (single token row, memory order coincides
        # with the token vector), which is exactly why chunked prefill (bsz==1
        # per step) matched while a whole-prompt prefill (bsz>1) diverged.)
        # .contiguous() after the swap materializes the [bsz, heads, head_dim]
        # layout so the flatten to [bsz, heads*head_dim] is the intended vector.
        return self.o_proj(out.transpose(0, 1).contiguous().reshape(bsz, -1))


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
        # Issue #9: hybrid backend -- the per-step miss split (PCIe-fetch vs CPU).
        # The forward reads the model's moe_backend flag (the loader records the
        # resolved backend there); a test harness that sets neither falls back to
        # use_offload, so a CPU test of the split still engages the hybrid path.
        self.use_hybrid = bool(getattr(config, "use_hybrid", False))
        # Router: pure-torch replicated Linear (issue-24 WP6). ``num_experts`` is
        # small and fully replicated, so ``LinearReplicated`` is the right class.
        from freetoken.layers import LinearReplicated

        self.gate = LinearReplicated(config.hidden_size, self.num_experts, has_bias=False, dtype=dtype)
        if self.use_offload or self.use_hybrid:
            # ADR 0002: no XPU-resident expert params. The routed experts are
            # read from the LRU slot pool the loader attaches to the model
            # (``model.moe_cache`` + ``model.moe_layer_id``); the host banks are
            # the source of truth. (``self.experts`` is left unset -- the loader
            # never copies into expert modules on this path.)
            self.experts = None
        else:
            self.experts = nn.ModuleList(_Qwen3Expert(config).to(device, dtype) for _ in range(self.num_experts))

    def _is_cpu_layer(self, model) -> bool:
        """Whether this MoE layer's experts compute on the CPU (issue #8).

        The CPU path is engaged only when the resolved MoE backend is a CPU
        variant (``cpu`` / ``hybrid``) AND the model's ``moe_cpu_moe_layers``
        partition names this layer. The partition (resolved from
        ``--moe-cpu-layers`` by the loader) is a concrete list of MoE-layer
        indices: ``[]`` means no layer is steered to the CPU (the serve default
        ``--moe-backend auto`` -> ``offload``, so this block stays on the XPU
        slot pool), and a full ``range(total)`` list means every MoE layer on
        the CPU (the ``--moe-backend cpu`` / explicit ``"auto"`` spec).
        ``self.layer_id`` is the block's decoder-layer index (set by
        ``_Qwen3DecoderLayer``); the model's ``moe_layer_id`` map turns it into
        the MoE-layer index the partition is keyed by. When the loader hasn't
        attached the map yet (a test harness), fall back to the block-local
        ``use_offload`` flag.
        """
        moe_backend = getattr(model, "moe_backend", None)
        if moe_backend == "cpu":
            # Pure CPU backend: every MoE layer not named as an offload layer
            # computes its routed experts on the host.
            cpu_layers = getattr(model, "moe_cpu_moe_layers", None)
            if cpu_layers is None:
                # Defensive: the loader always sets moe_cpu_moe_layers for the
                # cpu backend; None here means "all on CPU" (legacy harness).
                return True
            moe_layer_id = getattr(model, "moe_layer_id", None)
            if moe_layer_id is None:
                return bool(cpu_layers)
            return moe_layer_id[self.layer_id] in cpu_layers
        if moe_backend == "hybrid":
            # The fine-grained miss split is per-step, not per-layer: every layer
            # splits its routed-expert misses by the profile's fetch fraction
            # (the q* policy, ADR 0002). Only a layer explicitly carved out by
            # ``--moe-cpu-layers`` skips the split and computes fully on the CPU
            # (the coarse whole-layer split the q* policy does not model).
            if self.use_hybrid:
                return False
            cpu_layers = getattr(model, "moe_cpu_moe_layers", None)
            if cpu_layers is None or not cpu_layers:
                return False
            moe_layer_id = getattr(model, "moe_layer_id", None)
            if moe_layer_id is None:
                return True
            return moe_layer_id[self.layer_id] in cpu_layers
        # In-VRAM (fused/None) or pure offload: no CPU layers.
        return False

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

        # CPU backend (issue #8): this layer's routed-expert GEMM runs on the
        # host from the pinned banks, not streamed to the device. Decided per
        # layer off the model's ``moe_cpu_moe_layers`` partition (the loader
        # resolves ``--moe-cpu-layers`` to it; None == every MoE layer on the
        # CPU, the --moe-backend=cpu default). With --moe-cpu-layers 0 (or a
        # spec that excludes this layer) the model stays offload-only: self.use_offload
        # is True and this branch is skipped, so this layer streams through the
        # LRU slot pool below. (When use_offload is False -- the in-VRAM path --
        # moe_cpu_moe_layers is None but moe_backend is "fused"/None, so this
        # branch is also skipped and the resident ``self.experts`` below runs.)
        if self._is_cpu_layer(model):
            return self._forward_cpu(flat, top_idx, top_w, model, batch).view(in_shape)
        if self.use_hybrid:
            return self._forward_hybrid(flat, top_idx, top_w, model, batch).view(in_shape)
        if self.use_offload:
            return self._forward_offload(flat, top_idx, top_w, model, batch).view(in_shape)

        out = torch.zeros_like(flat)
        # Per-expert gather: route each expert's tokens in one matmul each.
        #
        # The routing mask is built and resolved to row indices on the HOST,
        # not with device-side boolean-mask indexing (`flat[sel]`) or
        # `torch.nonzero` on an XPU tensor: on this torch/XPU build,
        # `nonzero()` (which boolean-mask indexing calls internally) silently
        # returns an EMPTY result for an XPU bool tensor regardless of its
        # actual content (`sel.sum()` / `sel.tolist()` are correct; `sel
        # .nonzero()` / `flat[sel]` are not) -- a real correctness bug, not
        # just the "implicit D2H sync" performance concern the offload /
        # hybrid / cpu backends already route around this same way. Building
        # the indices on the host and gathering with `index_select` /
        # `index_add_` sidesteps it entirely (and keeps this path
        # graph-capture-friendly: no data-dependent output shape).
        #
        # EXCEPT: the CPU round-trip (`top_idx.to("cpu")`) is itself a
        # device->host sync, which is a hard error while a graph capture is
        # in flight (issue moe-fused-graph-capture, #123 -- found completing
        # #15's XpuGraphRunner work: #118/#119/#122 fixed every other decode
        # sync, this is the last one). While capturing, route with a dense
        # mask instead: compute every expert for every token and weight-sum
        # with a per-(token, expert) mask built from `==`/`torch.where`
        # (plain elementwise ops -- no `nonzero()`, no host sync, so neither
        # the broken-on-XPU-nonzero() bug nor the capture restriction
        # applies). Strictly more compute than the gather above (every
        # expert runs on every token, not just its routed rows), so this is
        # opt-in via `_capturing` only -- the same trade-compute-for-
        # capturability shape #118's fixed-KV-buffer attention already uses.
        if getattr(model, "_capturing", False):
            for e in range(self.num_experts):
                w = torch.zeros(flat.shape[0], device=flat.device, dtype=flat.dtype)
                for slot in range(self.top_k):
                    w = torch.where(top_idx[:, slot] == e, top_w[:, slot], w)
                out = out + w[:, None] * self.experts[e](flat)
            return out.view(in_shape)

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

    def _forward_offload(self, flat, top_idx, top_w, model, batch, *, exclude: set | None = None) -> torch.Tensor:
        """Serve the routed experts through the host-offload LRU slot pool.

        The pool is a single global timestamp LRU shared by every MoE layer
        (ADR 0002): a prefill materializes the *whole* layer into the pool
        (evicting the LRU-resident experts of other layers), and a decode step
        streams in only the *missed* routed experts, each into an evicted slot.

        Routing is done **on the host** from a snapshot of the routed *expert*
        ids (``top_idx``), mapped through the cache's ``slot_for_id``. The cache
        does NOT rewrite ``top_idx`` in place: it stays the ``torch.topk``
        expert-id tensor across steps, so a repeat routing never leaves a stale
        slot id for the next step to misread (issue #7). Nothing downstream
        reads ``top_idx`` as slot ids after this method returns.
        """
        layer_id = model.moe_layer_id[self.layer_id]
        cache = model.moe_cache
        # The phase flag (not token count) decides prefill vs decode: in a mixed
        # step (decode reqs with 1 token each, or a decode req alongside a
        # prefill req) flat.shape[0] > 1 even though every token is a 1-token
        # decode. Deriving the phase from the token count then calls
        # materialize_layer (whole layer) instead of ensure_experts (routed
        # experts only): the LRU pool is shared across layers, so materializing
        # a layer evicts the other layer's just-resident decode experts and the
        # next decode step routes them through the wrong slot (or a slot the
        # pool has re-used) -- a silent logit divergence from the in-VRAM path.
        is_prefill = bool(batch is not None and batch.is_prefill)
        is_xpu = bool(getattr(cache, "is_xpu", False))
        B, k = top_idx.shape
        # 1) Snapshot the routed *expert* ids on the host before the LRU call.
        expert_ids = top_idx.to("cpu")
        # 2) Let the LRU pool decide residency and stage the host->device copies.
        if is_prefill:
            cache.materialize_layer(layer_id)
        else:
            cache.ensure_experts(layer_id, expert_ids)
        cache.copy_missing()

        # 3) Map expert -> slot on the host from the cache's own (Python) map.
        #    An expert the pool evicted maps to -1 -> clamped to 0, valid=False.
        slots = cache.slot_for_id[layer_id].to("cpu").tolist()
        S = cache.cache_size
        intermediate = int(model.config.moe_intermediate_size)
        # SlotWeightAccessor abstracts gu[s_i, ...]/dn[s_i] bf16 indexing over a
        # quantized bank format too (issue moe-quant-banks-compute, #137): for
        # "bf16" this is the exact same plain-tensor indexing as before (zero
        # behavior change); for "gptq_int4" it dequantizes each distinct
        # resident slot at most once per step, from the packed banks, never
        # the whole checkpoint (the RAM-saving point of the whole epic, #134).
        from freetoken.moe.offload_cache import SlotWeightAccessor

        slot_weights = SlotWeightAccessor(cache, intermediate, flat.dtype)
        dev = flat.device
        out = torch.zeros_like(flat)
        routed_cpu = torch.empty(B, k, dtype=torch.int64)
        valid_cpu = torch.zeros(B, k, dtype=torch.bool)
        for i in range(B):
            for j in range(k):
                slot = slots[int(expert_ids[i, j])]
                ok = 0 <= slot < S
                routed_cpu[i, j] = slot if ok else 0
                valid_cpu[i, j] = ok
        # Belt-and-suspenders tripwire: an out-of-range slot means a stale map
        # entry; fail loudly in Python rather than aborting the (shared) XPU.
        if is_xpu and (~valid_cpu).any():
            stale = int(routed_cpu[~valid_cpu].max()) if (~valid_cpu).any() else -1
            if stale >= S:
                raise IndexError(
                    f"layer {self.layer_id}: offload routed slot id {stale} >= "
                    f"cache_size {S}: stale slot map (ensure_experts desync)"
                )
        # 4) Host-side (expert, column j) -> row indices, built in EXPERT-MAJOR
        #    order (for each expert e ascending, then each column j ascending) so
        #    the float32 accumulation order into out[i] matches the in-VRAM
        #    _forward_inram loop (for e in range(num_experts): for slot in
        #    range(top_k)). The offload transport is a byte-identical weight copy
        #    (ADR 0002); the only divergence source was the accumulation order
        #    (slot-major here vs expert-major in-VRAM) -> ~1e-7 ULP drift
        #    amplified by the recurrent attention over decode steps (issue #18).
        #    Built from the CPU tensors only: the gather below never asks the
        #    device "which rows matched?" (a boolean-mask index would call
        #    nonzero(), whose data-dependent shape forces an implicit D2H sync
        #    mid-loop).
        num_experts = model.config.num_experts
        expert_slots = [int(slots[e]) for e in range(num_experts)]
        expert_to_col: dict[int, list[tuple[int, int]]] = {}
        for j in range(k):
            for i in range(B):
                if bool(valid_cpu[i, j]):
                    e_id = int(expert_ids[i, j])
                    expert_to_col.setdefault(e_id, []).append((j, i))
        groups: list[tuple[int, int, list[int]]] = []
        for e in range(num_experts):
            # Issue #9 hybrid: an expert the host-CPU half serves this step is
            # excluded from the XPU gather (belt-and-suspenders guard).
            if exclude and e in exclude:
                continue
            s_i = expert_slots[e]
            if not (0 <= s_i < S):
                continue
            for j, i in expert_to_col.get(e, []):
                groups.append((j, s_i, [i]))
        # 5) Gather per (expert, slot, row) on the device using host-built
        #    INTEGER indices (static shapes, no nonzero(), no implicit D2H in
        #    the loop).
        for j, s_i, rows in groups:
            idx = torch.tensor(rows, dtype=torch.long, device=dev)
            gate_w, up_w, down_w = slot_weights.get(s_i)
            y = top_w.index_select(0, idx)[:, j, None] * _expert_compute(
                gate_w,
                up_w,
                down_w,
                flat.index_select(0, idx),
            )
            out.index_add_(0, idx, y)
        return out

    def _forward_hybrid(self, flat, top_idx, top_w, model, batch) -> torch.Tensor:
        """Serve the routed experts by splitting each step's misses (issue #9).

        The host-offload and host-CPU halves share the same pinned expert banks
        (ADR 0002), so a decode step's routed-expert misses can be *partitioned*:
        a fraction f PCIe-fetched into the XPU LRU slot pool and computed there,
        the rest (1 - f) computed on the host CPU from the same host banks. f is
        the ``ft bench bw`` profile's fetch fraction for this expert format
        (``q*``: pcie/(pcie+cpu) -- of the two halves' combined bandwidth, the
        share carried by PCIe), balancing the two halves' completion times.

        Correctness: the two halves are *disjoint* expert sets (never overlap,
        together cover exactly the routed experts), and each half uses the *same*
        math + accumulation order as the pure backend it mirrors -- so hybrid's
        output is numerically identical to offload's (the q* split changes
        *which* experts ride each transport, not the arithmetic). The XPU side
        gathers only the fetched experts (``exclude`` = the CPU-computed experts);
        the CPU side computes only the rest. The two contributions are summed
        per-row.

        Prefill (a whole layer, or a layer the partition steers to the CPU) has
        no miss-split -- every routed expert is made resident -- so it degrades
        cleanly to the offload path there. The split applies to the routed-expert
        decode step, where the q* balance lives.
        """
        # Issue moe-quant-banks-compute (#137): the CPU half's math
        # (_cpu_subset_math) reads model.moe_cache.bank_sources["gate_up"] /
        # ["down"] directly -- the "bf16" schema's bank names -- and runs
        # plain-float matmuls on them. A "gptq_int4" cache's bank_sources use
        # different names entirely (qweight_gate_up, ...), so that lookup
        # would KeyError. Rather than teach the CPU half to dequantize too
        # (real, separable follow-up work -- the CPU path has none of the
        # slot-cache/copy_missing machinery SlotWeightAccessor hooks into, so
        # it would need its own dequant-and-cache logic), this format is
        # excluded from the hybrid split for now: force fetch_frac to 1.0 so
        # every miss rides PCIe through the (already gptq_int4-aware)
        # offload path below. A documented, deliberate tradeoff (this
        # issue's own accept criteria explicitly allows it as a first cut),
        # not a silent gap -- gptq_int4 previously would have crashed loudly
        # here (KeyError) rather than produced wrong numbers, and now simply
        # never reaches that code path. "fp8_block" (issue
        # moe-quant-banks-fp8, #152) has the exact same gap -- its
        # bank_sources are named weight_gate_up/scale_gate_up/... -- and gets
        # the same forced-fetch_frac=1.0 treatment for the same reason.
        cache_quant_format = getattr(getattr(model, "moe_cache", None), "quant_format", "bf16")
        fetch_frac = float(getattr(model, "moe_hybrid_fetch_fraction", 0.0) or 0.0)
        if cache_quant_format in ("gptq_int4", "fp8_block"):
            fetch_frac = 1.0
        if fetch_frac <= 0.0:
            # No usable profile -> every miss rides PCIe (pure offload).
            return self._forward_offload(flat, top_idx, top_w, model, batch)
        if fetch_frac >= 1.0:
            # 100% fetch -> no CPU misses (pure offload); avoid a degenerate CPU
            # call with an empty expert set.
            return self._forward_offload(flat, top_idx, top_w, model, batch)
        # Split the routed-expert *ids* into the two disjoint halves. The XPU half
        # (fetched) gets the top round(n*f) ids; the CPU half the rest. The split
        # is over the *unique* routed ids of this step (a miss is per-expert, not
        # per-token-column), computed host-side from the topk snapshot so it is
        # deterministic and never triggers a device->host sync.
        expert_ids_cpu = top_idx.to("cpu")
        seen: list[int] = []
        for eid in expert_ids_cpu.reshape(-1).tolist():
            if eid not in seen:
                seen.append(eid)
        n = len(seen)
        n_fetch = int(round(n * fetch_frac))
        # --moe-hybrid-max-fetch (issue #9): a non-negative int caps the per-step
        # PCIe-fetched expert count -- the operator's override of the profile's
        # q* fraction (a hard ceiling, not a ratio). -1 / unset = fully
        # profile-driven (no cap). The cap only ever *shrinks* the XPU half (and
        # thus grows the CPU half): a floor on the CPU share, so the disjoint-cover
        # invariant (XPU set + CPU set == routed set) still holds and the output
        # stays numerically identical to pure offload.
        max_fetch = int(getattr(model, "moe_hybrid_max_fetch", -1) or -1)
        if 0 <= max_fetch < n_fetch:
            n_fetch = max_fetch
        # Clamp the split to a sane range (never empty a half unless f is 0 / 1,
        # handled above).
        n_fetch = max(1, min(n - 1, n_fetch))
        seen_sorted = sorted(seen)
        cpu_experts = set(seen_sorted[: n - n_fetch])  # (1 - f) share -> CPU
        # The two halves run CONCURRENTLY (issue moe-hybrid-overlap): the
        # host-CPU half's pure-CPU matmuls (no XPU tensor touched) run on a
        # persistent single-worker background thread while the XPU half's
        # PCIe fetch + gather runs on *this* (main) thread -- the same
        # regime benchbw._bench_overlap measures. A decode step then costs
        # max(cpu_half, pcie_half), not their sum, matching the q* fetch
        # fraction's bandwidth-matched assumption. The pool is reused across
        # every layer / every step (cached on the model) rather than
        # spawning a fresh thread per call: this model has ~28 MoE layers per
        # decode step, and thread-creation overhead alone was large enough
        # relative to this tiny model's per-expert matmul cost to erase most
        # of the overlap's benefit when measured with a fresh Thread each time.
        #
        # The device<->host transfers (flat -> CPU in, the CPU result -> device
        # out) must stay on this thread: the XPU runtime faults that sync when
        # issued off the thread that built the engine (see
        # test_serve_live_engine_xpu.py's docstring for the same constraint).
        # So the CPU half's *input* is prepared here before the submit, and
        # its *output* is moved back to the device here after the result is
        # collected -- the background worker itself never touches an XPU
        # tensor.
        x_cpu = flat.to("cpu", non_blocking=True).float()
        top_idx_cpu = top_idx.to("cpu")
        top_w_cpu = top_w.to("cpu")

        future = self._hybrid_cpu_pool(model).submit(
            self._cpu_subset_math, x_cpu, top_idx_cpu, top_w_cpu, model, cpu_experts
        )

        # The XPU half fetches and computes every routed expert except the CPU
        # set; its per-(expert, column) accumulation is expert-major, matching the
        # pure offload path (and the in-VRAM reference), so the rows it serves are
        # byte-identical to what offload would produce.
        out = self._forward_offload(flat, top_idx, top_w, model, batch, exclude=cpu_experts)

        cpu_out = future.result()
        # The host-CPU half's (disjoint) share, also in expert-major order, so
        # the per-row sum matches offload exactly regardless of which half
        # happened to finish first.
        out += cpu_out.to(flat.device, non_blocking=True).to(flat.dtype)
        return out

    @staticmethod
    def _hybrid_cpu_pool(model) -> "ThreadPoolExecutor":
        """A single-worker thread pool for the hybrid CPU half, cached on the
        model so it survives across decode steps / layers instead of paying
        thread-creation cost on every call (see ``_forward_hybrid``)."""
        pool = getattr(model, "_moe_hybrid_cpu_pool", None)
        if pool is None:
            from concurrent.futures import ThreadPoolExecutor

            pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="moe-hybrid-cpu")
            model._moe_hybrid_cpu_pool = pool
        return pool

    def _cpu_subset_math(self, x_cpu, top_idx_cpu, top_w_cpu, model, cpu_experts) -> torch.Tensor:
        """Pure-CPU math for the hybrid split's host-CPU half.

        Computes *only* the routed experts in ``cpu_experts`` (from the pinned
        host banks) and returns their per-row contribution, on the CPU, in
        ``x_cpu``'s dtype/device -- so this is safe to run on a background
        thread (no XPU tensor is read or written anywhere in this method; the
        device<->host transfers around it are the caller's job, on the main
        thread). A row that routes to an expert not in ``cpu_experts``
        contributes nothing (that row is served by the XPU half).

        Accumulation is expert-major then top-k-column (matching
        ``_forward_cpu`` / the in-VRAM reference), so the per-row result the
        hybrid path sums is numerically identical to what the pure CPU backend
        would produce for that subset.
        """
        if not cpu_experts:
            return torch.zeros_like(x_cpu)
        # Host banks for this MoE layer (the pinned loader-built banks; the same
        # source the XPU slot pool streams from -- ADR 0002). No PCIe round-trip:
        # the source of truth is already on the host.
        moe_idx = model.moe_layer_id[self.layer_id]
        sources = model.moe_cache.bank_sources
        gate_up = sources["gate_up"][moe_idx]
        down = sources["down"][moe_idx]
        out = torch.zeros(x_cpu.shape, dtype=x_cpu.dtype, device="cpu")
        for e in range(int(model.config.num_experts)):
            if e not in cpu_experts:
                continue
            for j in range(int(self.top_k)):
                sel = top_idx_cpu[:, j] == e
                if not bool(sel.any()):
                    continue
                rows = sel.nonzero(as_tuple=True)[0]
                w_sel = top_w_cpu[sel, j]
                x_sel = x_cpu.index_select(0, rows)
                I = gate_up.shape[1] // 2
                # The host banks carry the model's dtype (bf16 for the hero); the
                # CPU math runs in float32 (the pure-CPU executor's convention), so
                # upcast the expert slices here -- otherwise a bf16 bank meets the
                # float32 x_sel and the matmul raises a dtype mismatch.
                gu_e = gate_up[e, 0:2 * I].float()
                gate = x_sel @ gu_e[:I].t()
                up = x_sel @ gu_e[I : 2 * I].t()
                y = (F.silu(gate) * up) @ down[e].float().t()
                out.index_add_(0, rows, y * w_sel[:, None])
        return out

    def _forward_cpu(self, flat, top_idx, top_w, model, batch) -> torch.Tensor:
        """Run the routed experts on the host (issue #8, ADR 0002).

        Where ``_forward_offload`` streams activated experts over PCIe to the XPU
        and runs the GEMM there, this path runs the whole expert GEMM on the CPU
        straight out of the pinned host banks (the loader's ``cpu_sources``) and
        ships only the resulting activations back. The math is identical to the
        in-VRAM reference (the executor reuses the same SwiGLU), so the only
        difference is *where the GEMM runs*.

        The per-(expert, column) accumulation is done in the same
        expert-major-then-top-k-column order the in-VRAM path uses, so the
        float32 accumulation order -- and therefore the greedy tokens -- match
        the resident reference exactly.
        """
        # The host banks for this MoE layer, read straight from the pinned
        # loader-built banks (no device round-trip -- the source of truth is
        # already on the host).
        moe_idx = model.moe_layer_id[self.layer_id]
        # The host banks are the source of truth (ADR 0002) and live in the moe
        # cache the loader attached (``set_bank_sources`` keeps the raw per-layer
        # [E, ...] host tensors, already unwrapped from any _PlainBank). Read
        # this layer's gate_up / down straight off the host -- no device
        # round-trip, no PCIe stream, no LRU slot juggling.
        sources = model.moe_cache.bank_sources
        gate_up = sources["gate_up"][moe_idx]
        down = sources["down"][moe_idx]
        executor = getattr(model, "_moe_cpu_executor", None)
        if executor is None:
            from freetoken.moe.cpu_executor import CpuMoeExecutor

            threads = int(getattr(model.config, "moe_cpu_threads", 0) or 0)
            executor = CpuMoeExecutor(
                num_experts=int(model.config.num_experts),
                intermediate=int(model.config.moe_intermediate_size),
                threads=threads,
            )
            model._moe_cpu_executor = executor
        return executor.forward(flat, top_idx, top_w, gate_up, down)


class _Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, device, dtype, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = nn.RMSNorm(config.hidden_size, dtype=dtype)
        self.self_attn = _Qwen3Attention(config, device, dtype, layer_id)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, dtype=dtype)
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
        # The model builds every module in the loader's effective dtype: the
        # loader stamps ``config.dtype`` with the dtype the engine runs in (the
        # EngineConfig.dtype, defaulting to bfloat16 when unpinned -- the same
        # default the engine uses for its own tensors). The old fallback read
        # ``config.dtype`` and, when it was None (a config.json with no
        # torch_dtype), defaulted to bfloat16 while the engine ran float32 -- a
        # bf16-module / fp32-activation mismatch at F.linear (issue #9 surfaced it).
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
        self.norm = nn.RMSNorm(hidden_size, dtype=dtype)
        # Pure-torch replicated Linear (issue-24 WP6); weight ``[vocab, hidden]``,
        # the same shape the checkpoint's ``lm_head.weight`` carries.
        from freetoken.layers import LinearReplicated

        self.lm_head = LinearReplicated(hidden_size, vocab_size, has_bias=False, dtype=dtype)

        # ADR 0002 / issue #8: when the engine picks a host-side MoE backend
        # (offload or cpu) the experts are never XPU-resident. The MoE blocks
        # then hold *only* the router (no expert ``nn.Linear`` params); the
        # routed experts are read from the host banks (offload: the LRU slot
        # pool the loader attaches; cpu: the pinned banks, straight on the host).
        # Both are flagged ``moe_offload`` because the block's dispatch keys on
        # it to decide "no resident experts -> don't build the expert params".
        if bool(getattr(config, "is_moe", False)) and (
            bool(getattr(config, "use_offload_moe", False))
            or bool(getattr(config, "use_cpu_moe", False))
            or bool(getattr(config, "use_hybrid", False))
        ):
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
        # A decode batch is uniform: the scheduler never mixes phases within
        # one batch (confirmed while fixing issue #116's sycl.py phase check
        # -- every backend in this codebase already trusts batch.phase for
        # exactly this), so every request contributes exactly one new token.
        # Skipping the extend_lens[i] tensor read for that case avoids a
        # device->host sync per request inside this loop -- the sync that
        # blocked capturing a whole decode-step model.forward() in a
        # torch.xpu.graph() (found building issue #15's XpuGraphRunner,
        # #117-#121). Prefill keeps reading the real per-request value: chunk
        # sizes genuinely vary there, and prefill is not a graph-capture
        # target (its shape already varies step to step).
        is_decode_batch = batch.phase == "decode"
        for i, req in enumerate(reqs):
            ext = 1 if is_decode_batch else int(extend_lens[i])
            token_slice = slice(offset, offset + ext)
            h = hidden[token_slice]
            for layer in self.layers:
                h = layer(h, positions[token_slice], req.table_idx, ctx, batch)
            # Keep only the last position of this request (next-token logits).
            out[i] = self.norm(h)[-1]
            offset += ext

        return self.lm_head(out)


__all__ = ["parse_config", "iter_weights", "Qwen3MoeForCausalLM"]
