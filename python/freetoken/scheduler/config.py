"""Scheduler knobs.

Upstream NVIDIA path: python/freetoken/scheduler/config.py
Filled in: GitHub issue ``scheduler`` (see docs/architecture.md).

The upstream config carries ZMQ / overlap-scheduling knobs for a distributed
multi-worker daemon. The Intel port is a single-process loop inside
:class:`~freetoken.engine.engine.Engine`, so only the scheduling-relevant
knobs survive the port:

* ``max_extend_tokens`` -- the per-step prefill token budget. A prompt longer
  than this is *chunked* (see :class:`~freetoken.scheduler.prefill.PrefillAdder`):
  only ``min(budget, remaining)`` tokens are scheduled per step and the request
  stays pending until its whole prompt has been extended.
* ``max_running_req`` -- the cap on concurrently admitted requests. It is
  inherited from :class:`~freetoken.engine.config.EngineConfig` (the page table
  and the token pool are both sized from it upstream).

Everything else (ZMQ addrs, overlap cadence) is engine-internal and stays in
:mod:`freetoken.engine`. This module is import-safe without torch: it only
subclasses the engine config and adds plain-Python fields, which keeps the
dual-venv contract (``import freetoken.scheduler`` never pulls in torch).
"""
from __future__ import annotations

from dataclasses import dataclass

# Import the config *module* directly, not the `freetoken.engine` package: the
# package's __init__ eagerly imports engine.py, which imports torch. The
# scheduler must stay torch-free on the CPU venv (the dual-venv contract), so it
# must not trigger that import. config.py itself is torch-free (it only imports
# `DistributedInfo` from freetoken.distributed and stdlib dataclasses).
from freetoken.engine.config import EngineConfig


# Frozen, to match the parent (a non-frozen dataclass cannot inherit from a
# frozen one). Only the scheduling-relevant field is added; everything else is
# inherited from EngineConfig. ``max_running_req`` is inherited and *not*
# re-declared -- re-declaring a field already declared in the parent is a
# dataclass error in a frozen class.
@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    """Scheduling knobs layered on top of the engine config.

    ``Engine`` accepts any :class:`EngineConfig` and builds its own
    :class:`SchedulerConfig` view internally, so a plain engine config keeps
    working unchanged while callers may pass a :class:`SchedulerConfig` to tune
    the prefill budget.
    """

    # Per-step prefill budget (token count). Prompts longer than this chunk.
    # The upstream default is 8192; a shorter budget is valid and is exactly
    # what forces a long prompt to split across several prefill steps.
    max_extend_tokens: int = 8192
