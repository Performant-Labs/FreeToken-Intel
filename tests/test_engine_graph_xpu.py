"""XPU tests: XpuGraphRunner's capture/replay mechanism (issue `engine-graph`, #15).

These test the capture/replay *primitive* in isolation (a plain nn.Linear
forward against static tensors) -- not attention or the model, which are not
yet graph-capturable (see graph.py's module docstring for the two distinct
reasons why, found while wiring this up). That is the honest scope this
issue closes today: the mechanism is proven correct and available on this
box; wiring it into either attention backend or the model is real follow-up
work, tracked separately.

``xpu``-marked: deselected on a torch-free / no-XPU box (see ``conftest.py``).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from freetoken.engine.graph import XpuGraphRunner

XPU = pytest.mark.skipif(not torch.xpu.is_available(), reason="no XPU available")


@XPU
@pytest.mark.xpu
def test_capture_replay_matches_eager():
    """A captured nn.Linear forward, replayed, matches eager exactly."""
    dev = torch.device("xpu")
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 64).to(dev)

    static_in = torch.randn(4, 64, device=dev)
    static_out = torch.empty(4, 64, device=dev)

    def _fn():
        static_out.copy_(lin(static_in))

    runner = XpuGraphRunner(warmup_iters=3)
    assert not runner.is_captured
    runner.capture(_fn)
    assert runner.is_captured

    with torch.no_grad():
        expected = lin(static_in)

    runner.replay()
    torch.xpu.synchronize()
    assert torch.equal(static_out, expected), "replay must match eager bit-exact for identical inputs"


@XPU
@pytest.mark.xpu
def test_replay_reflects_updated_static_input():
    """Replay reads the CURRENT contents of the static input tensor.

    This is the property multi-step reuse depends on: capture once, then
    mutate the static buffer in place and replay again -- the graph must
    read the new values, not the ones frozen at capture time.
    """
    dev = torch.device("xpu")
    torch.manual_seed(1)
    lin = torch.nn.Linear(32, 32).to(dev)

    static_in = torch.randn(2, 32, device=dev)
    static_out = torch.empty(2, 32, device=dev)

    def _fn():
        static_out.copy_(lin(static_in))

    runner = XpuGraphRunner(warmup_iters=3)
    runner.capture(_fn)

    new_in = torch.randn(2, 32, device=dev)
    with torch.no_grad():
        expected = lin(new_in)

    static_in.copy_(new_in)
    runner.replay()
    torch.xpu.synchronize()
    assert torch.equal(static_out, expected)


@XPU
@pytest.mark.xpu
def test_replay_before_capture_raises():
    runner = XpuGraphRunner()
    with pytest.raises(RuntimeError, match="capture"):
        runner.replay()


def test_capture_without_xpu_raises_not_crashes(monkeypatch):
    """Off an XPU box, capture() must raise a clear RuntimeError (the issue's
    own accept criterion: document unsupported drivers, eager fallback),
    never crash or hang."""
    import freetoken.engine.graph as graph_mod

    monkeypatch.setattr(torch, "xpu", type("_NoXpu", (), {"is_available": staticmethod(lambda: False)})())
    runner = graph_mod.XpuGraphRunner()
    with pytest.raises(RuntimeError, match="XPU"):
        runner.capture(lambda: None)
