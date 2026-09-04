"""Qwen3.8-Flash-Next (``qwen4_exp``) -- Intel Arc Pro B70 port. Epic #198.

Upstream NVIDIA path: python/freetoken/models/qwen4_exp/
Fill in: GitHub issues `models-qwen4-hc` (#206), `models-qwen4-ple` (#207),
`models-qwen4-qsa` (#208), `models-qwen4-e2e` (#209) -- one package, built
up incrementally across all four (this port's established one-``__init__.py``
-per-model convention, unlike upstream's multi-file-per-model layout).

This file currently ships #206 only: the hyper-connection (gated residual)
primitive below. ``parse_config``/``iter_weights``/``Qwen4ExpForCausalLM``
stay stubs (``unimplemented``) until #209 wires the full model -- the
primitive is unit-tested standalone in the meantime (see
``tests/test_models_qwen4_hc.py``).

## Hyper-connections (``GatedResidual`` / ``GroupedPlusOneRMSNorm``, #206)

Upstream NVIDIA path: python/freetoken/models/qwen4_exp/hc.py

Every hyper-connection decoder layer reads and writes ``hc_count`` PARALLEL
residual streams packed as ``R [..., hc_count*hidden]`` (stream outer,
hidden inner), instead of the single residual stream every other decoder
layer in this port assumes:

    x, s = hc.mix(R)          # R [..., hc_count*hidden] -> x [..., hidden], s [..., hc_count] or None
    y    = block(x)            # attention / GDN / MoE, plain [..., hidden] -> [..., hidden]
    R    = hc.combine(R, y, s)

Formulas (upstream ``hc.py``'s own docstring, HF
``Qwen4ExpTextGatedResidual``)::

    Rn      = groupRMSNorm(R) * (1 + hc_norm.weight)        # per hidden-size stream, fp32 stats
    lora, s = input_mix_weight_down_block_inject(Rn)         # merged GEMM: [lowrank | hc_count | pad]
    gate    = input_mix_weight_up(silu(lora / hc_count))
    x       = mean_i(sigmoid(gate_i) * Rn_i)
    R'_i    = R_i + 2*sigmoid(s_i / hc_count) * y

``s`` is the RAW inject logit slice (pre ``2*sigmoid``) -- ``combine``
applies the activation. ``use_combine=False`` is the top-level mixer: it
owns the unmerged ``input_mix_weight_down``, returns ``s = None`` and has
no ``combine``.

This port ships the pure-torch reference path only (this session's
established "reference correctness first" discipline for every new
mechanism -- GDN, MLA, DSA before it): upstream's vendored Triton/CUDA
kernels (``kernel/triton/hc.py``) are NOT ported. Unlike upstream's
``BaseOP``/``LinearReplicated`` class hierarchy, this port follows the
established per-model convention (see ``glm_moe_dsa``/``deepseek_v4``):
plain ``nn.Module`` + ``nn.Linear``/``nn.Parameter``.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def grouped_plus_one_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, num_groups: int
) -> torch.Tensor:
    """RMSNorm each of ``num_groups`` equal slices of the last dim on its own fp32 statistic, then scale by (1+w)."""
    xf = x.float().unflatten(-1, (num_groups, -1))
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf.flatten(-2) * (1.0 + weight.float())).to(x.dtype)


class GroupedPlusOneRMSNorm(nn.Module):
    """Per-stream RMSNorm of an ``[..., num_groups*group]`` tensor with one weight element per feature.

    HF ``Qwen4ExpTextRMSNorm(dim, group_size)``. The checkpoint weight is
    zero-centered and loaded RAW: ``(1+w)`` is applied at runtime in fp32,
    never folded into the stored weight.
    """

    def __init__(self, size: int, eps: float, num_groups: int, *, dtype=None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(size, dtype=dtype))
        self.eps = eps
        self.num_groups = num_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return grouped_plus_one_rms_norm(x, self.weight, self.eps, self.num_groups)


class GatedResidual(nn.Module):
    """One hyper-connection block: ``mix`` reads the residual streams, ``combine`` writes a block output back.

    Weight keys (checkpoint names, prefix stripped): ``hc_norm.weight``,
    ``input_mix_weight_down_block_inject.weight`` (loader: concat of
    ``input_mix_weight_down`` ``[lowrank, hc*hidden]``, ``block_inject_weight``
    ``[hc_count, hc*hidden]`` and ``pad`` zero rows), ``input_mix_weight_up.weight``.
    """

    def __init__(
        self,
        hidden_size: int,
        hc_count: int,
        lowrank: int,
        eps: float,
        *,
        use_combine: bool = True,
        dtype=None,
    ) -> None:
        super().__init__()
        self.hc_count = hc_count
        self.hidden_size = hidden_size
        self.lowrank = lowrank
        self.use_combine = use_combine
        width = hc_count * hidden_size
        self.hc_norm = GroupedPlusOneRMSNorm(width, eps, hc_count, dtype=dtype)
        if use_combine:
            # 16-row alignment for the merged skinny GEMM (vLLM hyperconnection.py:98)
            self.pad_size = (-(lowrank + hc_count)) % 16
            self.input_mix_weight_down_block_inject = nn.Linear(
                width, lowrank + hc_count + self.pad_size, bias=False, dtype=dtype
            )
        else:
            self.pad_size = 0
            self.input_mix_weight_down = nn.Linear(width, lowrank, bias=False, dtype=dtype)
        self.input_mix_weight_up = nn.Linear(lowrank, width, bias=False, dtype=dtype)

    def _down(self, rn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Run the down GEMM and split off the raw inject logits; the pad columns are dropped."""
        if not self.use_combine:
            return self.input_mix_weight_down(rn), None
        down = self.input_mix_weight_down_block_inject(rn)
        return down[..., : self.lowrank], down[..., self.lowrank : self.lowrank + self.hc_count]

    def mix(self, R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Return the block input ``x [..., hidden]`` and the inject logits ``s [..., hc_count]`` (None if no combine)."""
        rn = grouped_plus_one_rms_norm(R, self.hc_norm.weight, self.hc_norm.eps, self.hc_count)
        lora, s = self._down(rn)
        lora = F.silu(lora.float() / self.hc_count)
        gate = self.input_mix_weight_up(lora.to(R.dtype))
        mixed = torch.sigmoid(gate.float()).unflatten(-1, (self.hc_count, self.hidden_size))
        mixed = mixed * rn.float().unflatten(-1, (self.hc_count, self.hidden_size))
        return mixed.mean(-2).to(R.dtype), s

    def combine(self, R: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Inject the block output ``y [..., hidden]`` back into every stream of ``R``."""
        inject = 2.0 * torch.sigmoid(s.float() / self.hc_count)
        out = R.float().unflatten(-1, (self.hc_count, self.hidden_size))
        out = out + y.float().unsqueeze(-2) * inject.unsqueeze(-1)
        return out.flatten(-2).to(R.dtype)


# --------------------------------------------------------------------------- #
# Not yet implemented: PLE (#207), QSA (#208), full model wiring (#209).
# --------------------------------------------------------------------------- #

from freetoken._stub import unimplemented


def parse_config(*args, **kwargs):
    unimplemented("parse_config", "models-qwen4-e2e")


def iter_weights(*args, **kwargs):
    unimplemented("iter_weights", "models-qwen4-e2e")


class Qwen4ExpForCausalLM:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(self, *args, **kwargs):
        unimplemented("Qwen4ExpForCausalLM.forward", "models-qwen4-e2e")


__all__ = [
    "GatedResidual",
    "GroupedPlusOneRMSNorm",
    "grouped_plus_one_rms_norm",
    "parse_config",
    "iter_weights",
    "Qwen4ExpForCausalLM",
]
