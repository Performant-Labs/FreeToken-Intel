"""CPU expert executor: run the routed MoE experts on the host.

Upstream NVIDIA path: python/freetoken/moe/cpu_executor.py (issue ``moe-cpu``).

Where the ``offload`` backend streams *activated* experts over PCIe to the XPU
and runs the GEMM there, the CPU backend runs the whole expert GEMM on the host:
it reads the expert weights straight out of the pinned host banks the loader
builds (ADR 0002 -- the same banks #7 streams and #9 partitions), computes the
routed experts on the CPU's high RAM-bandwidth, and ships only the resulting
activations back. That makes "some MoE layers run on the CPU" real on the B70.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from freetoken.kernel._toolchain import ToolchainError
from freetoken.kernel.cpu_moe import cpu_moe, cpu_moe_forward_fast


class CpuMoeExecutor:
    """Host-side expert GEMM over the pinned host banks.

    The executor owns no model state: it is constructed once per engine and, per
    MoE layer, is handed the layer's routed ``top_idx`` / ``top_w`` plus that
    layer's host expert banks. It returns the routed contribution
    ``sum_j top_w[:, j] * expert_j(flat)`` shaped ``[T, H]`` -- exactly what the
    model's in-VRAM (and offload) paths produce, so the CPU path is numerically
    identical to the resident reference (it differs only in *where* the GEMM
    runs, never in the math -- ADR 0002).
    """

    def __init__(self, num_experts: int, intermediate: int, *, threads: int = 0) -> None:
        self.num_experts = int(num_experts)
        self.intermediate = int(intermediate)
        # Issue #291: threads reaches the native pool's own width knob.
        # ``0`` means the pool's own "auto" (hardware_concurrency, clamped to
        # num_experts) -- see cpu_moe_forward_fast's docstring -- not "keep
        # the pure-PyTorch loop", the way it silently did before this issue.
        self.threads = int(threads)
        # Issue #291: the native GEMM epic #249 built (#250's naive port,
        # #252's AVX-512 + thread-pool fast path) is compiled/loaded once per
        # executor and used for every ``forward`` call from here on. A missing
        # C++ toolchain is the *only* reason to fall back to the pure-PyTorch
        # loop below -- it stays only as that fallback, never the default.
        try:
            self._native_module = cpu_moe()
        except ToolchainError:
            self._native_module = None

    def forward(
        self,
        flat: torch.Tensor,
        top_idx: torch.Tensor,
        top_w: torch.Tensor,
        gate_up: torch.Tensor,
        down: torch.Tensor,
    ) -> torch.Tensor:
        """Run the routed experts on the host.

        Args:
            flat: ``[T, H]`` token-major input activations (the block's
                post-norm residual), in the model's dtype/device.
            top_idx: ``[T, k]`` int tensor of routed *expert* ids per token
                (the router's ``topk`` output) -- expert ids, not slots.
            top_w: ``[T, k]`` router weights per routed expert (already
                renormalized to sum to 1 across the ``k``).
            gate_up: the layer's ``[E, 2I, H]`` host bank; row ``e`` packs the
                gate ``[I, H]`` (first ``I`` rows) then the up ``[I, H]``
                (next ``I`` rows) -- weight (``[out, in]``) orientation.
            down: the layer's ``[E, H, I]`` host bank (the down projection).

        Returns:
            ``[T, H]`` routed contribution on ``flat``'s device, accumulated
            in **expert-major then top-k-column** order so the float32
            accumulation order matches the in-VRAM reference (see
            ``_Qwen3MoE._forward_inram`` / ``_forward_offload``).
        """
        if self._native_module is not None:
            out = cpu_moe_forward_fast(
                self._native_module,
                flat,
                top_idx,
                top_w,
                gate_up,
                down,
                self.num_experts,
                self.intermediate,
                expert_mask=None,
                threads=self.threads,
            )
            return out.to(device=flat.device, dtype=flat.dtype)
        return self._python_forward(flat, top_idx, top_w, gate_up, down)

    def _python_forward(
        self,
        flat: torch.Tensor,
        top_idx: torch.Tensor,
        top_w: torch.Tensor,
        gate_up: torch.Tensor,
        down: torch.Tensor,
    ) -> torch.Tensor:
        """Pure-PyTorch fallback, used only when the native module (#250/#252)
        could not be compiled -- see ``__init__``. Same expert-major,
        top-k-column accumulation order as the native path."""
        dev = flat.device
        # Mirror the block's router: host-side routing (no device "which rows?"
        # query). top_idx / top_w are the fresh topk/weights from the block,
        # already on the device; snapshot the routed expert ids on the host and
        # group rows by (expert, column) in expert-major order.
        expert_ids = top_idx.to("cpu")
        B, k = expert_ids.shape
        num_experts = self.num_experts
        I = self.intermediate
        # gate_up / down are host bank views [E, ...]; read them straight (no
        # device copy -- ADR 0002: the CPU computes from the host, not a copy).
        gu = gate_up if gate_up.device.type == "cpu" else gate_up.to("cpu")
        dn = down if down.device.type == "cpu" else down.to("cpu")
        gu_t = gu  # [E, 2I, H]; per-expert gate/up are [I, H]
        dn_t = dn  # [E, H, I]

        out = torch.zeros_like(flat)
        for e in range(num_experts):
            for j in range(k):
                sel = (expert_ids[:, j] == e)
                if not bool(sel.any()):
                    continue
                # Per (expert, column) group: gather the input rows to the host,
                # run the expert there, and scatter the router-weighted output
                # back. index_select/index_add_ have static shapes -> no implicit
                # D2H/Nz-sync mid-loop on a device build.
                idx = torch.nonzero(sel, as_tuple=False).view(-1)
                idx_dev = idx.to(dev)
                x_cpu = flat.index_select(0, idx_dev).to("cpu")
                # SwiGLU: down(silu(gate(x)) * up(x)). down is applied to the
                # elementwise product (not the input), matching _expert_compute:
                # (F.silu(x @ gate_w.t()) * (x @ up_w.t())) @ down_w.t().
                gate = x_cpu @ gu_t[e, 0:I].t()
                up = x_cpu @ gu_t[e, I : 2 * I].t()
                y = (F.silu(gate) * up) @ dn_t[e].t()
                y_dev = y.to(dev)
                w = top_w.index_select(0, idx_dev)[:, j, None]
                out.index_add_(0, idx_dev, w * y_dev)
        return out
