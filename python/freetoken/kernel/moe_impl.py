"""MoE kernel dispatch (XPU fused vs CPU vs hybrid).

Upstream NVIDIA path: python/freetoken/kernel/moe_impl.py
Fill in: GitHub issue `moe-fused` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


def fused_moe(*args, **kwargs):
    unimplemented("fused_moe", "moe-fused")

