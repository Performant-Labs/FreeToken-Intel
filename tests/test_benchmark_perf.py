"""Unit tests for freetoken.benchmark.perf (issue `benchmarks`).

Torch-free: StepTiming / summarize / format_table are plain dataclasses and
arithmetic, so this runs in the CPU venv same as the rest of the CPU suite.
"""
from __future__ import annotations

from freetoken.benchmark.perf import StepTiming, format_table, summarize


def test_decode_tok_s_is_tokens_over_decode_seconds():
    s = StepTiming(backend="offload", ttft_s=0.5, decode_tokens=10, decode_s=2.0)
    assert s.decode_tok_s == 5.0


def test_decode_tok_s_is_nan_with_zero_decode_steps():
    s = StepTiming(backend="offload", ttft_s=0.5, decode_tokens=0, decode_s=0.0)
    assert s.decode_tok_s != s.decode_tok_s  # NaN != NaN


def test_summarize_groups_by_backend_and_averages():
    samples = [
        StepTiming(backend="offload", ttft_s=1.0, decode_tokens=10, decode_s=2.0),
        StepTiming(backend="offload", ttft_s=3.0, decode_tokens=10, decode_s=2.0),
        StepTiming(backend="hybrid", ttft_s=2.0, decode_tokens=20, decode_s=2.0),
    ]
    summary = summarize(samples)
    assert set(summary) == {"offload", "hybrid"}
    assert summary["offload"]["repeats"] == 2
    assert summary["offload"]["ttft_s_mean"] == 2.0  # mean(1.0, 3.0)
    assert summary["offload"]["decode_tok_s_mean"] == 5.0
    assert summary["hybrid"]["repeats"] == 1
    assert summary["hybrid"]["decode_tok_s_mean"] == 10.0


def test_summarize_preserves_first_seen_backend_order():
    samples = [
        StepTiming(backend="hybrid", ttft_s=1.0, decode_tokens=1, decode_s=1.0),
        StepTiming(backend="offload", ttft_s=1.0, decode_tokens=1, decode_s=1.0),
        StepTiming(backend="hybrid", ttft_s=1.0, decode_tokens=1, decode_s=1.0),
    ]
    assert list(summarize(samples)) == ["hybrid", "offload"]


def test_format_table_contains_every_backend_and_header():
    summary = summarize(
        [
            StepTiming(backend="offload", ttft_s=1.0, decode_tokens=10, decode_s=2.0),
            StepTiming(backend="hybrid", ttft_s=1.0, decode_tokens=10, decode_s=1.0),
        ]
    )
    table = format_table(summary)
    assert "backend" in table
    assert "offload" in table
    assert "hybrid" in table
