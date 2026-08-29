"""Fused MoE on XPU (experts resident in VRAM). Upstream: CUDA fused MoE.

Upstream NVIDIA path: python/freetoken/moe/fused.py
Fill in: GitHub issue ``moe-fused`` (see docs/architecture.md).

The router (``nn.Linear`` -> softmax -> top-k) is owned by the model's MoE block;
this backend takes the router's ``gating`` logits and runs the routed-expert compute
-- top-k selection, the per-expert SwiGLU GEMM over the stacked ``w1``/``w2`` banks,
and the router-weighted combine -- as a fused grouped op on the XPU.
"""
from __future__ import annotations

import torch

from freetoken.kernel.moe_impl import fused_moe
from freetoken.moe.base import BaseMoeBackend


class FusedMoe(BaseMoeBackend):
    """Fused (XPU-VRAM-resident) MoE backend: router top-k + SwiGLU expert GEMM."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
    ) -> torch.Tensor:
        """Run the routed experts.

        ``hidden_states`` ``[T, H]``, ``w1`` ``[E, 2I, H]`` (gate||up), ``w2`` ``[E, H, I]``
        (down), ``gating_output`` ``[T, E]`` router logits -> ``[T, H]``.
        """
        return fused_moe(
            hidden_states,
            w1,
            w2,
            gating_output,
            topk,
            renormalize,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
