"""Python bindings around oneCCL (upstream: pynccl).

Upstream NVIDIA path: python/freetoken/kernel/pynccl.py
Fill in: GitHub issue `oneccl-tp` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


class OneCCLCommunicator:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def all_reduce(self, *args, **kwargs):
        unimplemented("OneCCLCommunicator.all_reduce", "oneccl-tp")

