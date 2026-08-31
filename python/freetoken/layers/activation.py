"""Pure-torch ``*_and_mul`` fused activations for the B70 XPU port.

Upstream (NVIDIA) ``freetoken.layers.activation`` dispatches each variant to a
flashinfer or in-repo Triton kernel:

    y = act(x[..., :d]) * x[..., d:],   d = x.shape[-1] // 2

Triton has no XPU backend and flashinfer is CUDA-only, so this port expresses
the exact same math with torch ops (elementwise / memory-bound, so torch is fine
on XPU) instead of a fused kernel. The signatures and the ``out`` buffer
contract mirror upstream so callers can pass a preallocated output.

``import torch`` is deferred into the functions so ``import freetoken.layers``
stays torch-free in the CPU venv (the dual-venv contract).
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

__all__ = ["silu_and_mul", "gelu_and_mul", "gelu_tanh_and_mul", "swigluoai_and_mul"]

# sqrt(2/pi), the tanh-GELU coefficient, and log2(e) (the fast exp/sigmoid path).
_SQRT_2_OVER_PI = 0.7978845608028654
_GELU_C = 0.044715
_LOG2E = 1.4426950408889634
# 1/sqrt(2) == 0.7071067811865476 (the erf-GELU argument scale).


def _check_out(x, out, d):
    """Validate a caller-supplied ``out`` against the [*, d] output shape.

    If ``out`` is ``None`` there is nothing to check (the caller allocates). A
    mismatched dtype or shape raises ``ValueError``. The shape check is *strict*
    (same rank, per-dim sizes) rather than relying on ``copy_`` broadcasting: a rank
    mismatch (e.g. a 3-D ``out`` for a 2-D input) would otherwise let ``copy_``
    broadcast the y and write into the wrong layout instead of failing loudly.
    """
    if out is None:
        return
    if out.dtype != x.dtype:
        raise ValueError(f"out dtype {out.dtype} must match input dtype {x.dtype}")
    expected = x.shape[:-1] + (d,)
    if out.ndim != len(expected) or tuple(out.shape) != expected:
        raise ValueError(f"out shape {tuple(out.shape)} != expected {expected}")


# NOTE: _LOG2E / _GELU_C mirror the upstream fast-path constants; they are not all
# strictly needed by the torch port but are kept for reference parity with the kernel.


def silu_and_mul(x, out=None):
    """Fused SiLU gate: ``y = silu(gate) * up`` over uninterleaved halves.

    gate = x[..., :d], up = x[..., d:], d = x.shape[-1] // 2. SiLU matches
    upstream's fast-path ``x / (1 + exp(-x))``.
    """
    import torch

    d = x.shape[-1] // 2
    _check_out(x, out, d)
    gate = x[..., :d]
    up = x[..., d : 2 * d]
    y = gate / (1.0 + torch.exp(-gate))
    y = y * up
    if out is not None:
        out.copy_(y)
        return out
    return y


def gelu_and_mul(x, out=None):
    """Fused (erf) GELU gate: ``y = gelu(gate) * up``.

    gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2))) -- the exact-GELU form the
    upstream Triton kernel computes via libdevice.erf.
    """
    import torch

    d = x.shape[-1] // 2
    _check_out(x, out, d)
    gate = x[..., :d]
    up = x[..., d : 2 * d]
    act = 0.5 * gate * (1.0 + torch.erf(gate * 0.7071067811865476))
    y = act * up
    if out is not None:
        out.copy_(y)
        return out
    return y


def gelu_tanh_and_mul(x, out=None):
    """Fused tanh-approx GELU gate: ``y = gelu_tanh(gate) * up``.

    gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) --
    the ``gelu_pytorch_tanh`` approximation (HF's gelu_tanh), matching the
    upstream tanh.approx kernel.
    """
    import torch

    d = x.shape[-1] // 2
    _check_out(x, out, d)
    gate = x[..., :d]
    up = x[..., d : 2 * d]
    inner = _SQRT_2_OVER_PI * (gate + _GELU_C * gate * gate * gate)
    act = 0.5 * gate * (1.0 + torch.tanh(inner))
    y = act * up
    if out is not None:
        out.copy_(y)
        return out
    return y


def swigluoai_and_mul(x, out=None, *, alpha: float = 1.702, limit: float = 7.0):
    """SwiGLU-OAI (gpt-oss / MiniMax-M3 ``swigluoai``) over UNINTERLEAVED halves.

    ``y = clamp(gate, max=limit) * sigmoid(alpha * gate) * (clamp(up, +-limit) + 1)``
    with gate = x[..., :d], up = x[..., d:]. Same math as gpt-oss's interleaved
    kernel; the [gate; up] half layout matches the NVFP4 expert banks and the
    merged gate_up projections.
    """
    import torch

    d = x.shape[-1] // 2
    _check_out(x, out, d)
    gate = x[..., :d].clamp(max=limit)
    up = x[..., d : 2 * d].clamp(min=-limit, max=limit)
    act = gate / (1.0 + torch.exp(-alpha * gate))
    y = act * (up + 1.0)
    if out is not None:
        out.copy_(y)
        return out
    return y
