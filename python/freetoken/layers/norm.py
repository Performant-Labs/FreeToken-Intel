"""RMSNorm layer family (issue ``layers``, #24) — XPU.

Upstream NVIDIA path: python/freetoken/layers/norm.py

Upstream dispatches ``rmsnorm`` / ``fused_add_rmsnorm`` / the Gemma variants through
``flashinfer`` (CUDA) with a ``freetoken.kernel.triton.norm`` fallback. On the Intel
XPU neither is available: the Triton fallback is CUDA-specific (``launch_pdl`` / PDL
+ PTX inline-asm), which the Intel Triton backend rejects (``KeyError:
launch_pdl``), so this build computes the *same* RMSNorm math with PyTorch
primitives, which run natively on the XPU (``torch.rms_norm``). The public API —
class names, ``__init__`` / ``forward`` / ``forward_inplace`` /
``forward_add_residual`` signatures, and the ``eps`` / ``weight`` / ``size`` /
``with_scale`` attributes — matches upstream exactly so the model code is a drop-in
port.

``torch`` is imported lazily inside methods (``TYPE_CHECKING`` at module scope) so
that importing ``freetoken.layers`` stays torch-free: the per-PR CPU gate runs in a
venv with no torch, and the xpu-marked tests instantiate the real layers on the B70.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from .base import BaseOP

if TYPE_CHECKING:
    import torch


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        import torch

        self.eps = eps
        self.weight = torch.empty(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch

        return torch.rms_norm(x, (x.shape[-1],), self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        import torch

        x.copy_(torch.rms_norm(x, (x.shape[-1],), self.weight, self.eps))


class GemmaRMSNorm(BaseOP):
    """Gemma4-style RMSNorm: scales by the *raw* checkpoint weight via ``(1 + w)``.

    ``with_scale=False`` uses a runtime ones vector that is intentionally not part of
    ``state_dict`` (upstream parity).
    """

    def __init__(self, size: int, eps: float, with_scale: bool = True) -> None:
        import torch

        self.eps = eps
        self.size = size
        self.with_scale = with_scale
        if with_scale:
            self.weight = torch.empty(size)
        else:
            self._ones_weight: torch.Tensor | None = None

    def _kernel_weight(self, x: torch.Tensor) -> torch.Tensor:
        import torch

        if self.with_scale:
            return self.weight
        if self._ones_weight is None:
            self._ones_weight = torch.ones(self.size, device=x.device, dtype=x.dtype)
        return self._ones_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch

        if x.dim() == 2:
            w = 1.0 + self._kernel_weight(x)
            return torch.rms_norm(x, (x.shape[-1],), w, self.eps)
        original_shape = x.shape
        x = x.reshape(-1, original_shape[-1])
        w = 1.0 + self._kernel_weight(x)
        return torch.rms_norm(x, (original_shape[-1],), w, self.eps).reshape(original_shape)

    def forward_add_residual(self, x: torch.Tensor, residual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        import torch

        residual = residual + x
        w = 1.0 + self._kernel_weight(residual)
        x.copy_(torch.rms_norm(residual, (residual.shape[-1],), w, self.eps))
        return x, residual


class GemmaPlusOneRMSNorm(BaseOP):
    """``(1 + w)``-scaled RMSNorm (Gemma / MiniMax-M3 semantics).

    Upstream calls flashinfer's ``gemma_rmsnorm`` (or the Triton fallback); here it
    is the same ``(1 + w)`` rmsnorm in PyTorch. Per-head 3-D inputs are collapsed to
    2-D first (see upstream) and the original shape is restored on the way out.
    """

    def __init__(self, size: int, eps: float) -> None:
        import torch

        self.eps = eps
        self.size = size
        self.weight = torch.empty(size)

    def _flat(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x
        assert x.is_contiguous(), "per-head gemma norm needs a contiguous buffer"
        return x.view(-1, self.size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch

        flat = self._flat(x)
        w = 1.0 + self.weight
        return torch.rms_norm(flat, (self.size,), w, self.eps).view(x.shape)

    def forward_inplace(self, x: torch.Tensor) -> None:
        import torch

        flat = self._flat(x)
        w = 1.0 + self.weight
        flat.copy_(torch.rms_norm(flat, (self.size,), w, self.eps))


class GemmaPlusOneRMSNormFused(BaseOP):
    """``(1 + w)``-scaled RMSNorm with the fused-add-residual API.

    Drop-in for the decoder layernorm seam: ``forward(x)`` is a plain norm, and
    ``forward(x, residual)`` does the in-place ``residual += x`` then
    ``x = (1+w)-norm(residual)``, returning ``(x, residual)``.
    """

    def __init__(self, size: int, eps: float) -> None:
        import torch

        self.eps = eps
        self.size = size
        self.weight = torch.empty(size)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        import torch

        if residual is None:
            w = 1.0 + self.weight
            return torch.rms_norm(x, (x.shape[-1],), w, self.eps), x
        residual = residual + x
        w = 1.0 + self.weight
        x.copy_(torch.rms_norm(residual, (residual.shape[-1],), w, self.eps))
        return x, residual


class RMSNormFused(BaseOP):
    """Plain RMSNorm with the fused-add-residual API (upstream ``RMSNormFused``)."""

    def __init__(self, size: int, eps: float) -> None:
        import torch

        self.eps = eps
        self.weight = torch.empty(size)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        import torch

        if residual is None:
            return torch.rms_norm(x, (x.shape[-1],), self.weight, self.eps), x
        residual = residual + x
        x.copy_(torch.rms_norm(residual, (residual.shape[-1],), self.weight, self.eps))
        return x, residual
