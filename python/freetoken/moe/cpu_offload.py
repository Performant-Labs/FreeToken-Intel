"""CPU and hybrid CPU+XPU MoE backends (q-star overlap policy).

Upstream NVIDIA path: python/freetoken/moe/cpu_offload.py
Fill in: GitHub issue `moe-hybrid` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented
from freetoken.moe.base import BaseMoeBackend


class CpuOffloadMoeBackend(BaseMoeBackend):
    """Registry/engine hook for the ``cpu`` MoE backend (issue #8, ADR 0002).

    Selecting ``moe_backend="cpu"`` makes the engine run the MoE expert GEMM on
    the host CPU (host RAM bandwidth) instead of streaming activated experts over
    PCIe to the XPU. The per-block forward does *not* route through this
    backend's ``forward``: ``_Qwen3MoE.forward`` (and the Qwen3.5 variant) call
    :class:`freetoken.moe.cpu_executor.CpuMoeExecutor` directly with the block's
    router ``top_idx``/``top_w`` and the layer's pinned host banks. This class
    only signals to the engine *which* backend the experts run on; ``forward``
    exists to satisfy :class:`BaseMoeBackend` and raises, because a caller that
    actually routed the Qwen CPU experts through it (with the dense
    ``w1``/``w2``/``gating_output`` contract) would be a bug.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def forward(
        self,
        hidden_states,
        w1,
        w2,
        gating_output,
        topk: int,
        renormalize: bool,
        activation: str,
        apply_router_weight_on_input: bool,
    ):
        raise NotImplementedError(
            "CpuOffloadMoeBackend.forward is not the Qwen CPU entry point: the "
            "block calls CpuMoeExecutor directly with the router top_idx/top_w "
            "and the layer's host banks (issue #8, ADR 0002)."
        )


class HybridMoeBackend(BaseMoeBackend):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(self, *args, **kwargs):
        unimplemented("HybridMoeBackend.forward", "moe-hybrid")
