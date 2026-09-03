"""Compressed-tensors ``pack-quantized`` INT8 weights: unpack / dequantize
(issue `moe-quant-banks-int8`, #154).

This is the format a REAL deployed MoE checkpoint actually uses --
``rj1013/gemma-4-26B-A4B-it_q8`` (Gemma-4-26B-A4B, ``enable_moe_block: true``,
``num_experts: 128``), verified directly against its real ``config.json`` and
``model.safetensors.index.json``, and the raw safetensors tensor headers
(fetched via HTTP range request, no full checkpoint download). Its per-expert
keys are ``...experts.{e}.{gate_proj|up_proj|down_proj}.{weight_packed,
weight_scale,weight_shape}`` -- three tensors, not the plain ``weight`` +
``weight_scale`` this project's own earlier (wrong, unverified) guess
assumed. See issue #154's own comment trail for the full correction.

``python/freetoken/kernel/triton/int8_linear.py`` (the ORIGINAL, plain
per-channel primitive -- unpacked ``torch.int8`` weight + one scale per row,
no groups) stays as-is: it is a real, independently useful primitive in its
own right (issue `quant-xpu`, #10's general scope), just not what this real
checkpoint's on-disk bytes actually look like. This module is the one that
matches real bytes.

Format (per the ``compressed_tensors`` library's own
``compressors.pack_quantized.helpers.pack_to_int32`` /
``unpack_from_int32``, read directly from the pip package to confirm the bit
layout rather than guessing):

* ``weight_packed`` is ``[N, ceil(K/4)]`` ``int32`` (a ``[N, K]`` logical
  ``int8`` tensor, ``N`` = out_features, ``K`` = in_features, packed along
  the ``K`` axis). For ``num_bits=8`` (the only case this module
  implements -- this project's only real target so far) packing is DENSE
  and never splits an element across a word boundary (8 divides 32 evenly:
  each int32 word holds exactly 4 int8 values, byte 0 = the lowest 8 bits =
  the first (lowest-``k``) element of that word's 4). The stored byte value
  is the signed int8 value shifted into unsigned range by a flat ``+128``
  offset (``compressed_tensors``' own convention for ANY bit width, not an
  8-bit-specific quirk) -- i.e. ``stored_byte = int8_value + 128``, undone
  on read by a plain ``- 128``. This is arithmetic offset-binary encoding,
  not GPTQ's ``stored - 1`` reserved-code convention -- do not conflate the
  two; there is nothing to "+1 correct" here.
* ``weight_scale`` is ``[N, num_groups]`` (``bf16`` in the real checkpoint,
  but not guaranteed) -- one value per ``(output channel, group)`` pair.
  ``num_groups = K // group_size``; sequential groups (``g_idx[k] = k //
  group_size``), confirmed via ``compressed_tensors.quantization.quant_args
  .ActivationOrdering``: this checkpoint's ``actorder: "static"`` is an
  ALIAS for ``"weight"`` ordering (reordering only during calibration, not
  applied to the saved weights) -- NOT ``"group"``/``"dynamic"`` ordering
  (which WOULD require a separate ``g_idx`` permutation tensor). No such
  tensor exists in the real checkpoint's index, consistent with this.
  ``num_groups == 1`` (``group_size == K``) degenerates to pure per-channel
  quantization -- the same storage mechanism serves both "channel" and
  "group" ``quantization_config.config_groups.*.weights.strategy`` values
  (verified against two different real checkpoints: a "channel"-strategy
  one for dense attention layers, and this "group"-strategy one for MoE
  experts), so this module handles both via the same code path.
* ``weight_shape`` is ``[2]`` ``int64`` -- the real logical ``[N, K]``
  (``out_features, in_features``), stored separately because it cannot
  always be recovered from ``weight_packed``'s own shape alone (``K`` need
  not be an exact multiple of 4, though it is for every real tensor found
  so far).
* No ``weight_zero_point`` tensor for a symmetric checkpoint (confirmed:
  ``compressed_tensors.compressors.pack_quantized.base``'s own decompressor
  reads ``state_dict.get("weight_zero_point", None)`` and only asserts it
  is present for an ASYMMETRIC scheme) -- dequant is plain
  ``weight_int8 * scale``, no zero-point subtraction.
"""
from __future__ import annotations

import torch

_BITS = 8  # this module only implements the 8-bit case (the real checkpoint's format)
_ELEMS_PER_WORD = 32 // _BITS  # dense, no cross-word splitting at 8 bits
_OFFSET = 1 << (_BITS - 1)  # compressed_tensors' flat signed<->unsigned pack offset


def unpack_int8_from_int32(weight_packed: torch.Tensor, k: int) -> torch.Tensor:
    """Unpack a ``[N, ceil(K/4)]`` int32-packed tensor into ``[N, K]`` int8.

    ``k`` is the real logical ``K`` (from ``weight_shape``) -- the packed
    tensor's own column count is ``ceil(K/4)``, which only recovers ``K``
    exactly when ``K`` is itself a multiple of 4 (true of every real tensor
    checked so far, but not asserted here as a universal guarantee).
    """
    if weight_packed.dtype != torch.int32:
        raise TypeError(f"weight_packed must be int32, got {weight_packed.dtype}")
    n, packed_cols = weight_packed.shape
    # Byte i of each word (0..3) is element (word_col * 4 + i); masking after
    # the shift makes this correct regardless of the shift being arithmetic
    # (torch right-shifts a signed int32 tensor arithmetically) since only
    # the low 8 bits survive the mask either way.
    bytes_ = torch.stack(
        [(weight_packed >> (8 * i)) & 0xFF for i in range(_ELEMS_PER_WORD)], dim=-1
    )  # [N, packed_cols, 4]
    unpacked = bytes_.reshape(n, packed_cols * _ELEMS_PER_WORD)[:, :k]
    return (unpacked - _OFFSET).to(torch.int8)


def dequantize_int8_packed(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    k: int,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reconstruct the dense ``[N, K]`` weight from compressed-tensors
    pack-quantized INT8 tensors.

    ``weight_packed`` is ``[N, ceil(K/4)]`` int32; ``weight_scale`` is
    ``[N, num_groups]`` (``num_groups`` divides ``K`` evenly: ``num_groups
    == 1`` is pure per-channel, matching the real ``strategy: "channel"``
    case; ``num_groups > 1`` is sequential-group, matching the real
    ``strategy: "group"`` case -- both use this same formula).
    """
    n, packed_cols = weight_packed.shape
    if weight_scale.shape[0] != n:
        raise ValueError(
            f"weight_packed/weight_scale row mismatch: {n} vs {weight_scale.shape[0]}"
        )
    num_groups = weight_scale.shape[1]
    if k % num_groups != 0:
        raise ValueError(f"K={k} is not evenly divisible by num_groups={num_groups}")
    group_size = k // num_groups

    unpacked = unpack_int8_from_int32(weight_packed, k)  # [N, K] int8
    g_idx = torch.arange(k, device=weight_packed.device) // group_size
    per_col_scale = weight_scale.to(torch.float32)[:, g_idx]  # [N, K]
    return (unpacked.to(torch.float32) * per_col_scale).to(out_dtype)
