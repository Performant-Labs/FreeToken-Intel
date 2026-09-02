"""Unit tests for freetoken.benchmark.client.run_client (issue `benchmarks`).

Torch-free: a fake engine (plain Python, no torch.Tensor) exercises
run_client's step loop -- TTFT is the first step, later steps are decode,
an empty next_token_ids stops the loop early or (on the very first step)
raises. Runs in the CPU venv same as the rest of the CPU suite.
"""
from __future__ import annotations

import pytest

from freetoken.benchmark.client import run_client


class _FakeOutput:
    def __init__(self, next_token_ids):
        self.next_token_ids = next_token_ids


class _FakeEngine:
    """Yields a fixed sequence of steps; each entry is the step's token list."""

    def __init__(self, step_token_lists):
        self._steps = iter(step_token_lists)
        self.added = []

    def add_request(self, req):
        self.added.append(req)

    def step(self):
        try:
            tokens = next(self._steps)
        except StopIteration:
            tokens = []
        return _FakeOutput(tokens)


def _patch_clock(monkeypatch, deltas):
    """Make each successive (t0, t1) call pair around ``engine.step()`` measure
    the next entry of ``deltas`` as its duration, with no gap between steps."""
    calls: list[float] = []
    t = 0.0
    for d in deltas:
        calls.append(t)
        t += d
        calls.append(t)
    it = iter(calls)
    monkeypatch.setattr("freetoken.benchmark.client.time.perf_counter", lambda: next(it))


def test_first_step_is_ttft_rest_are_decode(monkeypatch):
    # 4 steps, each token list non-empty. Step durations: 0.5, 0.1, 0.2, 0.3.
    _patch_clock(monkeypatch, [0.5, 0.1, 0.2, 0.3])
    engine = _FakeEngine([[1], [2], [3], [4]])

    result = run_client(engine, [10, 11], backend="offload", max_tokens=4, uid=0)

    assert result.backend == "offload"
    assert result.ttft_s == pytest.approx(0.5)
    assert result.decode_tokens == 3
    assert result.decode_s == pytest.approx(0.1 + 0.2 + 0.3)
    assert len(engine.added) == 1
    assert engine.added[0].input_ids == [10, 11]


def test_stops_early_on_empty_step_before_budget(monkeypatch):
    # max_tokens=10, but the engine only has 2 real steps then goes idle -- the
    # idle (empty) step still consumes one (t0, t1) pair before the loop breaks.
    _patch_clock(monkeypatch, [0.4, 0.2, 0.1])
    engine = _FakeEngine([[1], [2]])

    result = run_client(engine, [10], backend="cpu", max_tokens=10, uid=0)

    assert result.ttft_s == pytest.approx(0.4)
    assert result.decode_tokens == 1
    assert result.decode_s == pytest.approx(0.2)


def test_raises_when_no_tokens_are_produced_at_all(monkeypatch):
    _patch_clock(monkeypatch, [0.1])
    engine = _FakeEngine([[]])  # empty on the very first step

    with pytest.raises(RuntimeError, match="hybrid"):
        run_client(engine, [10], backend="hybrid", max_tokens=5, uid=0)


def test_single_step_run_has_zero_decode_tokens(monkeypatch):
    _patch_clock(monkeypatch, [0.3])
    engine = _FakeEngine([[1]])

    result = run_client(engine, [10], backend="offload", max_tokens=1, uid=0)

    assert result.ttft_s == pytest.approx(0.3)
    assert result.decode_tokens == 0
    assert result.decode_s == 0.0
    assert result.decode_tok_s != result.decode_tok_s  # NaN
