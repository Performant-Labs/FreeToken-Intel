"""Token-key comparison for the radix cache.

Upstream NVIDIA path: python/freetoken/kernel/radix.py (``fast_compare_key``).

The reference radix tree (``radix_cache.py``) compares a cached token-key against
an incoming ``input_ids`` tensor and needs the length of their common prefix.
Upstream implements this as an AOT-compiled C++ kernel (``radix.cpp``) behind a
``tvm_ffi`` module. That AOT toolchain is not part of the Intel port, so this
module provides a pure-torch fallback with the identical signature and
semantics: the first position at which the two 1-D int tensors differ, or the
length of the shorter when one is a prefix of the other.

A C++ AOT port of ``radix.cpp`` (through the same ``icpx`` machinery the
attention kernels use) is a perf follow-up; until it lands, this torch
implementation is correct and runs on any torch device.
"""
from __future__ import annotations

import torch


def fast_compare_key(x: torch.Tensor, y: torch.Tensor) -> int:
    """Length of the common prefix of two 1-D int tensors.

    Returns the index of the first position where ``x`` and ``y`` differ, or
    ``min(len(x), len(y))`` when one is a prefix of the other. Both must be
    1-D int tensors of the same dtype (token ids / slot indices).
    """
    n = min(x.numel(), y.numel())
    if n == 0:
        return 0
    a = x[:n].to(torch.int64)
    b = y[:n].to(torch.int64)
    mask = a != b
    if not mask.any():
        return n
    return int(mask.nonzero()[0].item())


__all__ = ["fast_compare_key"]
