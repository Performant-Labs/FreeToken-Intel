"""Fused MoE (router top-k + SwiGLU expert GEMM) for the Intel Arc Pro B70 (XPU).

Upstream NVIDIA path: python/freetoken/kernel/triton/fused_moe.py
Fill in: GitHub issue ``moe-fused`` (see docs/architecture.md).

Like the reference attention backend (``freetoken/attention/triton.py``), this is a
*dependency-free, pure-torch* implementation: it runs identically on the XPU and on
a CPU (which is what makes the per-PR CPU test able to build a reference), and it
needs no triton-intel / sgl-kernel CUDA. The "fused" in the name is the router
top-k + the per-expert SwiGLU GEMM + the weighted combine done as a *grouped* GEMM:
each expert's weights are read once and applied to exactly the tokens routed to that
expert (a small per-expert matmul), then the per-token contributions are combined.
Grouping by expert -- rather than expanding the weight banks to one copy per
token-expert pair -- is what keeps VRAM flat: the expanded form
(``[T*topk, 2I, H]``) duplicates every routed expert's weights per token and OOMs on
real serving sizes (the B70 has 32 GB).

Combine: each pair's weighted contribution is written into a ``[T, topk]``-sized
buffer at its (token, slot) position and the buffer is reduced with a sum over the
``topk`` slots. The scatter is a fixed-stride ``buffer[t, s] = contrib[t*topk + s]``
(resize of a flat ``[T*topk, H]``), NOT an indirect ``index_add_`` -- this torch XPU
build miscompiles / deadlocks ``index_add_`` on a *masked, non-contiguous* index
tensor at serving scale (in-bounds indices still trip the ``Indexing.h`` assert), so
the fixed-stride scatter + sum is used instead. Both are VRAM-flat (``[T, topk, H]``
activations only; the weight banks are read once per expert, never expanded).

XPU reliability notes (all verified against a CPU reference on the B70):
  * Pair grouping by expert uses ``argsort`` + ``searchsorted`` for the group
    boundaries. ``torch.bincount`` (mis-buckets the ids) and ``torch.diff``+``nonzero``
    (drops a boundary) both report wrong boundaries on this XPU build, which shifts
    every per-expert slice and silently mis-routes the contributions. ``argsort`` and
    ``searchsorted`` are bit-identical on XPU and CPU, so the grouping is stable.
  * The per-expert token gather (``hidden_states[rows]``) and the router ``topk`` are
    plain ``argsort``-driven direct-indexing, which is XPU-safe at serving scale.

Weight convention (matches the loader's MoE banks, see models/loader.py):
  * ``w1`` (the "gate_up" bank) is ``[E, 2*I, H]`` -- for expert ``e`` the first
    ``I`` rows are the gate projection ``[I, H]`` and the next ``I`` rows are the up
    projection ``[I, H]``.
  * ``w2`` (the "down" bank) is ``[E, H, I]`` -- expert ``e``'s down projection.
  * ``gating`` is ``[T, E]`` router logits; ``hidden_states`` is ``[T, H]``.

Per-expert SwiGLU: ``down( silu(x @ gate^T) * (x @ up^T) )``. On this torch XPU
build the ``[tokens, *]`` operand must be on the LEFT of the matmul (``x @ w^T``),
so the GEMMs below are written that way (NOT ``w @ x^T``).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _act(activation: str, x: torch.Tensor) -> torch.Tensor:
    """The activation between the gate and up projections."""
    if activation in ("silu", "swiglu", "swish"):
        return F.silu(x)
    if activation == "gelu":
        return F.gelu(x)
    if activation == "relu":
        return F.relu(x)
    raise ValueError(f"unsupported MoE activation: {activation!r}")


def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating: torch.Tensor,
    topk: int,
    renormalize: bool,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    """Router top-k + grouped SwiGLU expert GEMM + weighted combine.

    ``hidden_states`` ``[T, H]`` -> ``gating`` ``[T, E]`` -> top-k route -> per-expert
    SwiGLU GEMM over ``w1`` ``[E, 2I, H]`` / ``w2`` ``[E, H, I]`` -> ``[T, H]`` output
    (same shape as ``hidden_states``). Each token's contribution is the sum, over its
    ``topk`` routed experts, of ``router_weight * expert(x)`` (or the expert applied to
    ``router_weight * x`` when ``apply_router_weight_on_input`` is set -- vLLM's option
    to fold the router weight into the input so the two GEMMs share the scaling).
    """
    if gating.shape[0] != hidden_states.shape[0]:
        raise ValueError(
            f"gating rows ({gating.shape[0]}) != hidden_states rows ({hidden_states.shape[0]})"
        )
    hidden_states = hidden_states.contiguous()
    T, H = hidden_states.shape
    inter = w1.shape[1] // 2  # I; w1 is [E, 2I, H]

    # -- 1. Router: softmax -> top-k (weights + expert ids), optional renorm. --
    gate_log = F.softmax(gating, dim=-1)
    top_w, top_idx = torch.topk(gate_log, topk, dim=-1)  # both [T, topk]
    if renormalize:
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)
    top_w = top_w.to(hidden_states.dtype)
    top_idx = top_idx.to(torch.long)

    # 2. Group the token-expert pairs by expert and run a *grouped* GEMM: one matmul
    #    per expert over the tokens routed to it. This reads each expert's weights
    #    ONCE (w1[e], w2[e]) instead of expanding them to [T*topk, ...] (which
    #    duplicates every routed expert's weights per token and OOMs on real serving
    #    sizes -- the B70 has 32 GB). Within an expert the tokens are processed in
    #    their flat pair order (t*topk + s), so the [n_e, H] result re-resizes
    #    back onto the [T, topk] slot buffer with a fixed stride (no indirect scatter).
    E = w1.shape[0]
    flat_tok = top_idx.reshape(-1)  # [T*topk] expert ids
    flat_w = top_w.reshape(-1)  # [T*topk] router weights
    contribs = torch.empty(T * topk, H, device=hidden_states.device, dtype=hidden_states.dtype)

    # Sort the pairs by expert (stable) so each expert's pairs form one contiguous
    # run in ``order``. Group boundaries are found with searchsorted on the sorted
    # expert-id sequence: expert e's run spans [first index >= e, first index >= e+1).
    # We deliberately AVOID torch.bincount and torch.diff+nonzero here -- on this
    # torch XPU build both mis-report group boundaries at serving scale (bincount
    # mis-buckets the ids; diff+nonzero drops a boundary), which shifts every
    # per-expert slice and silently mis-routes the contributions. argsort and
    # searchsorted are both bit-identical on XPU and CPU, so the grouping is stable.
    order = torch.argsort(flat_tok, stable=True)  # [T*topk] pair ids, sorted by expert
    exp_sorted = flat_tok[order]  # [T*topk] expert id of each sorted position (ascending)
    dev = flat_tok.device
    n_pairs = exp_sorted.shape[0]
    starts = torch.empty(E, dtype=torch.long, device=dev)
    ends = torch.empty(E, dtype=torch.long, device=dev)
    for e in range(E):
        starts[e] = torch.searchsorted(exp_sorted, torch.tensor(e, device=dev))
        ends[e] = torch.searchsorted(exp_sorted, torch.tensor(e + 1, device=dev))

    for e in range(E):
        s = int(starts[e].item())
        end = int(ends[e].item())
        if end == s:
            continue  # most experts serve zero tokens at a given step
        # this expert's pairs (the sorted slice is contiguous -> a fixed-stride view)
        pair_ids = order[s:end]
        weights = flat_w[pair_ids]  # [n_e]
        rows = pair_ids // topk  # [n_e] token rows (pair p belongs to token p // topk)
        x = hidden_states[rows]  # [n_e, H]
        if apply_router_weight_on_input:
            x = x * weights[:, None]
        # One grouped SwiGLU GEMM for expert e: its weights are read once.
        gate_w = w1[e, 0:inter, :]  # [I, H]
        up_w = w1[e, inter : 2 * inter, :]  # [I, H]
        down_w = w2[e]  # [H, I]
        h = _act(activation, x @ gate_w.t()) * (x @ up_w.t())  # [n_e, I]
        contrib = h @ down_w.t()  # [n_e, H]
        if not apply_router_weight_on_input:
            contrib = contrib * weights[:, None]
        # Scatter each pair's contribution back to its (token, slot) position.
        # contrib is in flat pair order, so position (t, s) = (pair // topk, pair %
        # topk) -- a fixed stride (== the original flat pair id, since pair = t*topk+s).
        # A plain advanced-index assignment (NOT index_add_) is XPU-safe at serving
        # scale, where index_add_ on a *masked, non-contiguous* index tensor deadlocks
        # (in-bounds indices still trip the Indexing.h assert). A token's slots are
        # distinct (topk is a set -- no slot is written twice), so assign is exact; the
        # final combine is the sum over slots below.
        contribs[rows * topk + (pair_ids % topk)] = contrib

    # 3. Combine: sum each token's topk routed contributions over the slot axis.
    buf = contribs.reshape(T, topk, H)
    return buf.sum(dim=1)
