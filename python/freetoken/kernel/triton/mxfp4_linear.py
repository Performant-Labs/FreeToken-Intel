"""OCP MXFP4 (E2M1, 32-element block, E8M0 shared exponent) weights:
quantize / dequantize (issue `quant-xpu`, #10 -- MXFP4, the other named
dtype alongside block-FP8, #125).

Upstream NVIDIA path: python/freetoken/moe/fused_mxfp4.py (``dequant_mxfp4_blocks``)
                       python/freetoken/kernel/triton/mxfp4_moe.py (the real Triton
                       GEMV/GEMM kernels, native fp4 tensor-core compute)

Same scoping decision as ``fp8_block_linear.py``: this ports upstream's own
plain-torch ``dequant_mxfp4_blocks`` reference helper faithfully (the format
itself is a public spec -- OCP Microscaling, used by real GPT-OSS MXFP4
checkpoints -- not proprietary, so porting it is a direct, low-risk win),
adds the missing inverse (``quantize_mxfp4_blocks``, upstream has none --
it only ever *reads* real MXFP4 checkpoints) so this is round-trip
testable without a real MXFP4 checkpoint on hand, and a dequant-on-forward
``mxfp4_linear`` wrapper. The real Triton split-K GEMV / grouped-GEMM
kernels (native 4-bit tensor-core compute, GPT-OSS's actual decode/prefill
path) are NOT ported -- unverified whether Intel's Triton-XPU backend
supports the same bit-manipulation primitives (``exp2``, int8 packing)
those kernels lean on, and there is no local MXFP4 checkpoint to validate
a full loader-integration + real-model round trip against.

Format: 32 fp4 (E2M1) elements per block, packed 2-per-byte (16 bytes/block,
low nibble = even element, high nibble = odd element -- matches upstream's
``torch.stack((blocks & 0x0F, blocks >> 4), dim=-1)`` unpack order) plus one
shared ``uint8`` E8M0 exponent per block: ``scale = 2 ** (e8m0_byte - 127)``
(IEEE-754 float32's own bias -- upstream builds the scale by bit-casting the
exponent straight into a float32's exponent field, which is exactly this).
E2M1 has no subnormal *flush*: exponent-bits 00 is itself the subnormal case
(0.0 / 0.5), so the 8 non-negative magnitudes are
``[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]`` and their negatives -- a 16-entry
LUT indexed by the raw 4-bit code (sign is the MSB).
"""
from __future__ import annotations

import torch

_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
_MAX_MAGNITUDE = 6.0
_BLOCK = 32


def dequantize_mxfp4_blocks(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reconstruct a dense ``[..., K]`` tensor (``K = blocks.shape[-1] * 32``)
    from MXFP4 blocks + E8M0 scales. ``blocks`` is ``[..., n_blocks, 16]``
    ``uint8`` (16 packed bytes = 32 fp4 elements per block); ``scales`` is
    ``[..., n_blocks]`` ``uint8``, one shared exponent per block.

    Ported verbatim from upstream's ``freetoken.moe.fused_mxfp4
    .dequant_mxfp4_blocks`` (the plain-torch reference path, not the Triton
    kernel) -- the format is a public spec (OCP Microscaling), not
    proprietary.
    """
    if blocks.dtype != torch.uint8:
        raise TypeError(f"MXFP4 blocks must be uint8, got {blocks.dtype}")
    if scales.dtype != torch.uint8:
        raise TypeError(f"MXFP4 scales must be uint8, got {scales.dtype}")
    if blocks.shape[-1] != 16:
        raise ValueError(f"MXFP4 block pack dimension must be 16 bytes, got {blocks.shape[-1]}")
    if blocks.shape[:-1] != scales.shape:
        raise ValueError(f"MXFP4 blocks/scales shape mismatch: {tuple(blocks.shape[:-1])} vs {tuple(scales.shape)}")

    nibbles = torch.stack((blocks & 0x0F, blocks >> 4), dim=-1).reshape(*blocks.shape[:-1], 32)
    lut = _LUT.to(blocks.device)
    unpacked = lut[nibbles.long()]
    scale = torch.exp2(scales.float() - 127).unsqueeze(-1)
    dequantized = unpacked * scale
    return dequantized.reshape(*blocks.shape[:-2], blocks.shape[-2] * 32).to(out_dtype)


def quantize_mxfp4_blocks(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a dense ``[..., K]`` tensor (``K`` a multiple of 32) to MXFP4
    blocks + E8M0 scales. The inverse of :func:`dequantize_mxfp4_blocks` --
    upstream has no such function (it only ever reads an already-quantized
    real checkpoint); this exists so the format is round-trip testable
    without one, and for any future ``ft checkpoint`` conversion tool that
    quantizes a bf16 checkpoint down to MXFP4 rather than only reading one.

    Per-block scale is the smallest power-of-two exponent that keeps the
    block's largest-magnitude element within E2M1's representable range
    (``|x| <= 6.0``); each scaled element is then quantized to its nearest
    of the 16 LUT codes (round-to-nearest, not round-to-even).
    """
    if weight.shape[-1] % _BLOCK != 0:
        raise ValueError(f"MXFP4 quantization needs K a multiple of {_BLOCK}, got {weight.shape[-1]}")
    w = weight.to(torch.float32)
    blocks_f = w.reshape(*w.shape[:-1], w.shape[-1] // _BLOCK, _BLOCK)
    amax = blocks_f.abs().amax(dim=-1).clamp_min(1e-12)
    # Smallest scale s = 2**e such that amax / s <= _MAX_MAGNITUDE.
    exponent = torch.ceil(torch.log2(amax / _MAX_MAGNITUDE))
    exponent_byte = exponent.add(127).round().clamp(0, 254).to(torch.uint8)
    scale = torch.exp2(exponent_byte.float() - 127).unsqueeze(-1)
    scaled = blocks_f / scale

    lut = _LUT.to(weight.device)
    # Nearest-LUT-code search: [..., n_blocks, 32, 1] vs [16] -> argmin over the last dim.
    diffs = (scaled.unsqueeze(-1) - lut).abs()
    nibbles = diffs.argmin(dim=-1).to(torch.uint8)  # [..., n_blocks, 32], values in [0, 16)

    lo = nibbles[..., 0::2]
    hi = nibbles[..., 1::2]
    blocks_u8 = (lo | (hi << 4)).to(torch.uint8)  # [..., n_blocks, 16]
    return blocks_u8, exponent_byte


def mxfp4_linear(
    x: torch.Tensor,
    weight_blocks: torch.Tensor,
    weight_scales: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x @ dequantize_mxfp4_blocks(weight_blocks, weight_scales).T (+ bias)``.

    Dequant-on-forward drop-in for a quantized ``nn.Linear``-shaped weight
    (``[out_features, in_features]``). Like ``fp8_block_linear``, a caller
    driving this repeatedly should dequantize once at load time instead
    (via :func:`dequantize_mxfp4_blocks` directly) and cache the result --
    this per-call wrapper pays the dequant cost every call.
    """
    w = dequantize_mxfp4_blocks(weight_blocks, weight_scales, out_dtype=x.dtype)
    return torch.nn.functional.linear(x, w, bias)
