"""Per-channel symmetric INT8 weight quantization (issue `quant-xpu`, #10 --
INT8, the other named dtype alongside block-FP8 #125 and MXFP4 #129).

Standard weight-only INT8 quantization (the scheme behind e.g. LLM.int8's
weight path, AWQ/GPTQ's simpler symmetric variant): one scale per output
channel (row), computed from that row's max absolute value --

    weight_int8[i, j] = round(weight[i, j] / scale[i]).clamp(-127, 127)
    scale[i]          = amax(weight[i, :]) / 127
    weight_bf16[i, j] = weight_int8[i, j] * scale[i]

Per-channel (not a single tensor-wide scale) because weight rows in a real
checkpoint have wildly different magnitudes -- a single global scale would
waste most of int8's 256 levels on the largest row and flatten every
smaller one to near-zero.

This is deliberately NOT GGUF's INT8/INT4 quant family (``Q8_0``, ``Q4_0``,
the ``Q4_K``/``Q5_K`` super-block k-quants): those are a different,
considerably larger format zoo (this port has no GGUF reader/dequant at
all yet -- ``models/gguf/`` is an empty stub) and a separate scope from
this plain, checkpoint-format-agnostic primitive. INT4 is also not
included here: a *useful* INT4 needs a block/group scheme (a single
per-channel INT4 scale is far too lossy at only 16 levels spanning a
whole row) which overlaps enough with MXFP4's block design (#129) that it
deserves its own follow-up rather than a rushed variant here.
"""
from __future__ import annotations

import torch

_INT8_MAX = 127


def quantize_int8_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a dense ``[N, K]`` (or higher-rank, last dim = K) tensor to
    per-row (dim -2, i.e. per output channel for an ``[out, in]`` weight)
    symmetric INT8. Returns ``(weight_int8, scale)`` where ``scale`` has
    shape ``weight.shape[:-1]`` (one scale per row)."""
    w = weight.to(torch.float32)
    amax = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = amax / _INT8_MAX
    quantized = (w / scale).round().clamp(-_INT8_MAX, _INT8_MAX).to(torch.int8)
    return quantized, scale.squeeze(-1)


def dequantize_int8_channel(
    weight_int8: torch.Tensor,
    scale: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Inverse of :func:`quantize_int8_channel`. ``scale`` is
    ``weight_int8.shape[:-1]`` (one value per row)."""
    if weight_int8.dtype != torch.int8:
        raise TypeError(f"weight_int8 must be int8, got {weight_int8.dtype}")
    if tuple(scale.shape) != tuple(weight_int8.shape[:-1]):
        raise ValueError(
            f"scale shape {tuple(scale.shape)} does not match weight_int8's row shape {tuple(weight_int8.shape[:-1])}"
        )
    return (weight_int8.to(torch.float32) * scale.unsqueeze(-1).to(torch.float32)).to(out_dtype)


def int8_linear(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    scale: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x @ dequantize_int8_channel(weight_int8, scale).T (+ bias)``.

    Dequant-on-forward drop-in for a quantized ``nn.Linear``-shaped weight
    (``[out_features, in_features]``), matching ``fp8_block_linear`` /
    ``mxfp4_linear``'s shape. A caller driving this repeatedly should
    dequantize once at load time instead (via
    :func:`dequantize_int8_channel` directly) and cache the result.
    """
    w = dequantize_int8_channel(weight_int8, scale, out_dtype=x.dtype)
    return torch.nn.functional.linear(x, w, bias)
