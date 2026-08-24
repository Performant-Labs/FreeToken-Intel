"""CPU expert GEMM (AVX-512 / AMX). Upstream: cpu_moe_ext.cpp.

Upstream NVIDIA path: python/freetoken/moe/cpu_executor.py
Fill in: GitHub issue `moe-cpu` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


class CpuMoeExecutor:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def run(self, *args, **kwargs):
        unimplemented("CpuMoeExecutor.run", "moe-cpu")

