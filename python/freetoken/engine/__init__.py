"""The engine package.

``EngineConfig`` is import-safe without torch (it only pulls ``DistributedInfo``
from ``freetoken.distributed``), so it is imported eagerly -- the scheduler
subclasses it and must be importable on the torch-free CPU venv (the dual-venv
contract). ``Engine`` / ``ForwardOutput`` / ``BatchSamplingArgs`` live in modules
that import torch, so they are exposed *lazily* via :func:`__getattr__` (PEP
562): importing ``freetoken.engine`` -- which ``freetoken.scheduler.config`` does
just to reach ``EngineConfig`` -- must NOT trigger ``import torch``.
"""
from .config import EngineConfig

__all__ = ["Engine", "EngineConfig", "ForwardOutput", "BatchSamplingArgs"]
__all_lazy__ = {"Engine", "ForwardOutput", "BatchSamplingArgs"}


def __getattr__(name):
    # PEP 562: lazy re-exports so the package stays torch-free at import time.
    if name == "Engine" or name == "ForwardOutput":
        from . import engine

        return getattr(engine, name)
    if name == "BatchSamplingArgs":
        from . import sample

        return getattr(sample, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(__all__))
