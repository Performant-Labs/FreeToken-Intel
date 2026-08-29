"""MoE kernel dispatch (XPU fused vs CPU vs hybrid).

Upstream NVIDIA path: python/freetoken/kernel/moe_impl.py
Fill in: GitHub issue ``moe-fused`` (see docs/architecture.md).

The ``fused`` MoE backend runs the routed experts on the XPU with the grouped
row-paired SwiGLU GEMM in :mod:`freetoken.kernel.triton.fused_moe`. That module is
dependency-free pure torch (no triton-intel / sgl-kernel CUDA) and runs identically
on XPU and CPU, so it is the single source of truth for the ``fused`` backend here.
(There is no separate CUDA kernel on this port: the "fused" path is the torch
grouped-GEMM implementation, and the CPU/offload paths live in ``freetoken.moe``.)
"""
from __future__ import annotations

import torch

from freetoken.kernel.triton.fused_moe import fused_moe as _fused_moe_torch


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

    See :func:`freetoken.kernel.triton.fused_moe.fused_moe` for the shapes and the
    weight-bank convention. This dispatcher exists so the ``fused`` backend and any
    future kernel-level callers share one entry point (and a place to later branch on
    XPU-vs-CPU or to swap in a triton-intel kernel without touching the backend).
    """
    return _fused_moe_torch(
        hidden_states,
        w1,
        w2,
        gating,
        topk,
        renormalize,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
