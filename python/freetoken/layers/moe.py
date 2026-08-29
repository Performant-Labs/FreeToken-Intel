"""Reusable Mixture-of-Experts layer: router + a pluggable routed-expert backend.

Upstream NVIDIA path: python/freetoken/layers/moe.py
Fill in: GitHub issue ``moe-fused`` (see docs/architecture.md).

This is the layer the models use to run a MoE block: it owns the *router*
(``gate``: hidden -> num_experts logits, softmax -> top-k) and the stacked expert
weight banks, and delegates the routed-expert compute (top-k + SwiGLU GEMM +
router-weighted combine) to a :class:`~freetoken.moe.base.BaseMoeBackend` (the
``fused`` XPU backend by default). Keeping the router here and the expert GEMM in
the backend is what lets the same layer run with a ``fused`` (in-VRAM) or an
offload/CPU backend depending on the engine's choice.

Weight-bank convention (matches the loader, see models/loader.py):
  * ``w1`` (gate_up) is ``[num_experts, 2*intermediate, hidden]`` -- per expert the
    first ``intermediate`` rows are the gate projection and the next are the up.
  * ``w2`` (down) is ``[num_experts, hidden, intermediate]``.
``hidden_states`` is token-major ``[num_tokens, hidden]`` (the engine feeds one
request's slice at a time), and the output restores that shape.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from freetoken.moe import create_moe_backend


class Moe(nn.Module):
    """Router (gate) + stacked expert banks + a pluggable routed-expert backend."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        intermediate_size: int,
        num_experts_per_tok: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        moe_backend: str = "fused",
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(device, str):
            device = torch.device(device)
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.intermediate_size = int(intermediate_size)
        self.top_k = int(num_experts_per_tok)
        self.renormalize = bool(renormalize)
        self.activation = activation
        self.apply_router_weight_on_input = bool(apply_router_weight_on_input)

        # Router: hidden -> num_experts logits (no bias; the top-k renormalizes).
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False, device=device, dtype=dtype)
        # Stacked expert banks: gate_up [E, 2I, H] (gate rows then up rows) and
        # down [E, H, I]. Registered as parameters so the loader can fill them
        # in-place (the same mechanism it uses for every other nn.Linear).
        self.w1 = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_size, self.hidden_size, device=device, dtype=dtype)
        )
        self.w2 = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, self.intermediate_size, device=device, dtype=dtype)
        )
        self.backend = create_moe_backend(moe_backend)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        in_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, in_shape[-1])  # [T, hidden]
        gating = self.gate(flat)  # [T, num_experts]
        out = self.backend.forward(
            flat,
            self.w1,
            self.w2,
            gating,
            self.top_k,
            self.renormalize,
            self.activation,
            self.apply_router_weight_on_input,
        )
        return out.view(in_shape)
