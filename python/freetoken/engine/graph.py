"""SYCL / Level Zero graph capture for decode (upstream: CUDA graphs).

Upstream NVIDIA path: python/freetoken/engine/graph.py
Fill in: GitHub issue `engine-graph` (see docs/architecture.md).

``torch.xpu`` on this box (torch 2.13.0+xpu) exposes the same graph-capture
API CUDA does (``torch.xpu.XPUGraph`` / ``torch.xpu.graph`` / a warmup-on-a-
side-stream-then-capture sequence) -- verified with a minimal standalone
capture+replay of an ``nn.Linear`` forward, bit-exact against eager.

:class:`XpuGraphRunner` is a thin, model-agnostic wrapper around that API: it
captures whatever ``fn`` does into a graph and replays it. It does **not**
itself know anything about the engine, the model, or attention -- the caller
is responsible for making sure every tensor ``fn`` reads or writes has a
*fixed address* across capture and every replay (the same discipline CUDA
graphs require): pass data in via tensors you already hold a persistent
reference to and mutate in place (``.copy_()``), never ones ``fn`` allocates
fresh each call.

**Neither attention backend is actually graph-capturable today** (issue
``engine-graph`` / #15), for two distinct reasons found while wiring this up:

* ``TritonAttentionBackend`` (the pure-torch reference) reads a KV history of
  length ``req.device_len``, which *grows every decode step* -- a fresh
  ``torch.arange(written)`` each call. Graph capture replays the exact same
  kernel launches against the exact same tensor shapes every time, so a
  growing shape breaks it outright. Fixing this means rewriting the attention
  math to attend over a fixed-size ``max_seq_len`` buffer with a mask instead
  of a growing slice (the same design real paged-attention kernels use, and
  what upstream's own ``engine/graph.py`` assumes its attention backend
  already does) -- an algorithmic change, not a plumbing one.

* ``SyclAttentionBackend`` calls its compiled kernel through raw ``ctypes``
  (see ``attention/sycl.py``), and the C++ side (``attention.cpp``) opens its
  **own** ``sycl::queue`` internally rather than submitting onto torch's
  currently-recording stream -- so ``torch.xpu.graph()`` cannot see that work
  at all, capturable or not. ``forward()`` also calls ``torch.xpu.synchronize()``
  immediately before the kernel call (needed today so the USM inputs are
  ready), and a host sync during capture is a hard error under
  ``torch.xpu.graph()`` (confirmed directly: ``RuntimeError: wait cannot be
  called for a queue which is recording to a command graph``). Fixing this
  needs the kernel to accept and submit onto torch's active queue/stream, and
  the synchronize to be removed or made capture-safe -- C++/SYCL kernel
  surgery, not a Python change.

Both are real architectural gaps, not just missing plumbing, and each is its
own follow-up. What *is* proven correct and ready to build on: the
capture/replay mechanism itself. ``torch.xpu.graph`` / ``torch.xpu.XPUGraph``
work exactly like their CUDA counterparts on this box (torch 2.13.0+xpu,
verified with a minimal ``nn.Linear`` capture+replay, bit-exact against
eager -- see ``tests/test_engine_graph_xpu.py``), and additionally, a decode
step's per-request ``batch.extend_lens`` is read with a Python-level
``int(extend_lens[i])`` inside the model's decoder-layer loop -- a host sync
that would *also* block capture at the whole-model level even once an
attention backend supports it, and needs to be resolved (or worked around
per-request) before a whole ``model.forward()`` can be captured end to end.
"""
from __future__ import annotations

from freetoken.utils import init_logger

logger = init_logger(__name__)


class XpuGraphRunner:
    """Capture a sequence of XPU kernel launches once; replay without
    repeating the Python-level dispatch that produced them.

    Not thread-safe and not reentrant: one runner captures one graph. Build a
    new runner (or call :meth:`capture` again) to capture a different ``fn``.
    """

    def __init__(self, warmup_iters: int = 3) -> None:
        self.warmup_iters = warmup_iters
        self._graph = None

    @property
    def is_captured(self) -> bool:
        return self._graph is not None

    def capture(self, fn):
        """Warm up ``fn`` on a side stream, then capture one call of it.

        ``fn`` takes no arguments and should read/write only tensors the
        caller already holds fixed-address references to (see module
        docstring). Returns whatever ``fn()`` returns from the *capture*
        call -- useful for a quick sanity check, but not a live view of
        replay's output: read the static buffer(s) ``fn`` wrote into after
        calling :meth:`replay`, not this return value.

        Raises ``RuntimeError`` if no XPU is available -- graph capture is
        meaningless without one, and callers should catch this and fall back
        to eager execution (the issue's own accept criterion: "document if
        graphs are unsupported on a given driver").
        """
        import torch

        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError(
                "XPU graph capture requires a torch XPU device; use eager execution instead"
            )

        # Warmup on a side stream (required before capture: it lets any
        # lazy one-time allocations / kernel JIT happen off the capture
        # stream, matching torch.cuda.graph's own documented sequence).
        side_stream = torch.xpu.Stream()
        side_stream.wait_stream(torch.xpu.current_stream())
        with torch.xpu.stream(side_stream):
            for _ in range(self.warmup_iters):
                fn()
        torch.xpu.current_stream().wait_stream(side_stream)
        torch.xpu.synchronize()

        self._graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(self._graph):
            result = fn()
        torch.xpu.synchronize()
        return result

    def replay(self) -> None:
        """Replay the captured graph. Raises if :meth:`capture` was never called."""
        if self._graph is None:
            raise RuntimeError("XpuGraphRunner.replay() called before capture()")
        self._graph.replay()
