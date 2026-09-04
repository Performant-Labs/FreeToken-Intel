"""LFM2(.5)-8B-A1B (``lfm2_moe``) -- Intel Arc Pro B70 port. Epic #229.

Upstream reference: HF ``transformers.models.lfm2_moe`` (real, registered
``Lfm2MoeConfig``/``Lfm2MoeForCausalLM``, not this session's cloned
FLashML/FreeToken upstream tree -- that project has no LFM2 port at all;
this is a different upstream entirely). Ground truth for the conv math
below is ``transformers/models/lfm2_moe/modeling_lfm2_moe.py``'s own
``Lfm2MoeShortConv``.

Fill in: GitHub issues `models-lfm2moe-conv` (#230), `models-lfm2moe-attn-moe`
(#231), `models-lfm2moe-e2e` (#232) -- one package, built up incrementally
(this port's established one-``__init__.py``-per-model convention).

This file currently ships #230 only: ``parse_config`` (real) and the short
gated-conv layer primitive (``ShortConv`` / ``short_conv_forward``) below.
``iter_weights``/``Lfm2MoeForCausalLM`` stay stubs (``unimplemented``) until
#232 wires the full model -- the conv primitive is unit-tested standalone
in the meantime (see ``tests/test_models_lfm2moe_conv.py``).

## Hybrid backbone

Unlike every other MoE model in this port, LFM2-MoE's decoder alternates
TWO different layer kinds per ``layer_types`` (confirmed against the real
``LiquidAI/LFM2.5-8B-A1B-Base`` checkpoint's own ``config.json``, NOT
assumed from the class default): most layers are a short causal gated
convolution (this issue), a periodic few (roughly every 4th, starting at
index 2) are full attention (issue #231). The first ``num_dense_layers``
layers use a plain dense MLP; the rest use the sparse MoE (also #231).

## Short gated conv (``ShortConv``, #230)

Ground truth: ``Lfm2MoeShortConv.forward`` in the real
``modeling_lfm2_moe.py``::

    BCx = in_proj(x).transpose(-1, -2)      # [.., 3*hidden, T]
    B, C, x = BCx.chunk(3, dim=-2)          # each [.., hidden, T]
    h = B * x                                # first gate
    h = causal_depthwise_conv1d(h, conv_weight, conv_bias, K=conv_L_cache)
    y = C * h                                # second gate
    y = out_proj(y.transpose(-1, -2))

The causal depthwise conv itself (the real upstream implementation calls
an external fused kernel, ``causal_conv1d_fn``/``causal_conv1d_update``,
whose semantics this port reproduces in pure torch): a standard
left-padded-only depthwise ``Conv1d`` -- ``nn.Conv1d(groups=hidden,
kernel_size=K, padding=K-1)(h)`` produces ``T+K-1`` outputs; the LAST
``K-1`` of those read from the (irrelevant) right zero-padding, so taking
only the first ``T`` outputs gives the causal result: position ``t``'s
output depends only on ``h[t-K+1 .. t]`` (zero-padded on the left for
``t < K-1``). This is exactly ``F.conv1d(..., padding=K-1)[..., :T]``,
confirmed by direct derivation from ``nn.Conv1d``'s own cross-correlation
definition -- ``output[t] = bias + sum_k weight[k] * input[t-(K-1)+k]``,
which for ``t <= T-1`` and ``k <= K-1`` never indexes past the original
input's last position, so the right-padding zeros are never read.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from freetoken._stub import unimplemented
from freetoken.models.config import ModelConfig


# --------------------------------------------------------------------------- #
# Checkpoint side (config parsing)
# --------------------------------------------------------------------------- #


def parse_config(hf_config: Any, model_path: str | None = None, **_kwargs) -> ModelConfig:
    """Build a :class:`ModelConfig` from a real HF ``Lfm2MoeConfig``.

    ``layer_types`` is read verbatim from the checkpoint (a real list of
    ``"conv"``/``"full_attention"`` strings, one per layer) -- do NOT
    assume a fixed ratio; the real ``LiquidAI/LFM2.5-8B-A1B-Base``
    checkpoint's own pattern is `[conv, conv, full_attention, conv, conv,
    conv, full_attention, ...]` (attention roughly every 4th layer,
    starting at index 2), confirmed directly from its ``config.json`` --
    but the config class ships no default builder for this field at all
    (unlike e.g. Mellum's ``["full_attention"] * num_hidden_layers``
    default), so a checkpoint that omits it entirely has no sane fallback
    and this raises rather than guessing.
    """
    src = hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(hf_config)
    layer_types = src.get("layer_types")
    num_layers = src.get("num_hidden_layers")
    if not layer_types:
        raise ValueError(
            "lfm2_moe checkpoint's config.json has no 'layer_types' -- this "
            "architecture has no derivable default (unlike e.g. Mellum's "
            "all-full_attention fallback), so the real conv/attention "
            "pattern must be present, not guessed"
        )
    if num_layers is not None and len(layer_types) != num_layers:
        raise ValueError(
            f"layer_types has {len(layer_types)} entries but num_hidden_layers={num_layers}"
        )

    cfg = ModelConfig(
        architectures=["Lfm2MoeForCausalLM"],
        hidden_size=src.get("hidden_size"),
        vocab_size=src.get("vocab_size"),
        num_layers=num_layers,
        num_experts=src.get("num_experts") or src.get("num_local_experts"),
        num_attention_heads=src.get("num_attention_heads"),
        num_key_value_heads=src.get("num_key_value_heads"),
        intermediate_size=src.get("intermediate_size"),
        moe_intermediate_size=src.get("moe_intermediate_size"),
        num_experts_per_tok=src.get("num_experts_per_tok"),
        first_k_dense_replace=src.get("num_dense_layers") or 0,
        is_moe=True,
        max_position_embeddings=src.get("max_position_embeddings"),
        tie_word_embeddings=src.get("tie_word_embeddings", False),
        rope_theta=(src.get("rope_parameters") or {}).get("rope_theta"),
        hidden_act=src.get("hidden_act", "silu"),
        dtype=src.get("dtype") or src.get("torch_dtype"),
    )
    cfg.attrs["layer_types"] = list(layer_types)
    cfg.attrs["conv_bias"] = bool(src.get("conv_bias", False))
    cfg.attrs["conv_L_cache"] = int(src.get("conv_L_cache", 3))
    cfg.attrs["num_dense_layers"] = int(src.get("num_dense_layers") or 0)
    cfg.attrs["norm_eps"] = src.get("norm_eps", 1e-5)
    cfg.attrs["norm_topk_prob"] = bool(src.get("norm_topk_prob", True))
    cfg.attrs["use_expert_bias"] = bool(src.get("use_expert_bias", False))
    cfg.attrs["routed_scaling_factor"] = src.get("routed_scaling_factor", 1.0)
    return cfg


def iter_weights(*args, **kwargs):
    unimplemented("iter_weights", "models-lfm2moe-e2e")


# --------------------------------------------------------------------------- #
# Forward side: the short gated-conv primitive (#230 only)
# --------------------------------------------------------------------------- #


def causal_depthwise_conv1d(
    h: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, kernel_size: int
) -> torch.Tensor:
    """Left-padded-only depthwise causal conv over the time axis.

    ``h [.., C, T]`` (channels, time -- matches ``nn.Conv1d``'s own layout),
    ``weight [C, kernel_size]`` (one filter per channel, depthwise),
    ``bias [C]`` or ``None``. Returns ``[.., C, T]``: position ``t`` depends
    only on ``h[.., t-kernel_size+1 .. t]`` (left-zero-padded).
    """
    padded = F.pad(h, (kernel_size - 1, 0))
    out = F.conv1d(padded, weight.unsqueeze(1), bias=bias, groups=h.shape[-2])
    return out


class ShortConv(nn.Module):
    """LFM2-MoE's short gated causal convolution (real math: see module docstring).

    Standalone/testable: takes and returns plain ``[T, hidden]`` (no batch
    dim, matching this port's own single-request-per-call convention used
    by every other model package here). Not yet wired into a decoder layer
    or the engine -- that is #232's job.
    """

    def __init__(self, hidden_size: int, kernel_size: int, has_bias: bool, dtype=None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.in_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=has_bias, dtype=dtype)
        self.conv_weight = nn.Parameter(torch.empty(hidden_size, kernel_size, dtype=dtype))
        self.conv_bias = nn.Parameter(torch.empty(hidden_size, dtype=dtype)) if has_bias else None
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=has_bias, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x [T, hidden] -> [T, hidden]``."""
        bcx = self.in_proj(x).transpose(0, 1)  # [3*hidden, T]
        b, c, gx = bcx.chunk(3, dim=0)  # each [hidden, T]
        h = b * gx
        h = causal_depthwise_conv1d(h.unsqueeze(0), self.conv_weight, self.conv_bias, self.kernel_size).squeeze(0)
        y = c * h
        return self.out_proj(y.transpose(0, 1))


# --------------------------------------------------------------------------- #
# Not yet implemented: attention + MoE router (#231), full model wiring (#232).
# --------------------------------------------------------------------------- #


class Lfm2MoeForCausalLM:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(self, *args, **kwargs):
        unimplemented("Lfm2MoeForCausalLM.forward", "models-lfm2moe-e2e")


__all__ = [
    "ShortConv",
    "causal_depthwise_conv1d",
    "iter_weights",
    "Lfm2MoeForCausalLM",
    "parse_config",
]
