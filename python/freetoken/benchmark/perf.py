"""Perf counters: per-repeat decode timing -> per-backend summary table.

Upstream NVIDIA path: python/freetoken/benchmark/perf.py
Fill in: GitHub issue `benchmarks` (see docs/architecture.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass
class StepTiming:
    """One repeat's bs=1 decode timing, split prefill (TTFT) vs decode.

    ``ttft_s`` is the wall time of the *first* :meth:`Engine.step` call (the
    prefill step -- for a short, unchunked prompt this is exactly the model's
    time-to-first-token). ``decode_tokens`` / ``decode_s`` cover every step
    after that, so :attr:`decode_tok_s` is a pure decode-phase rate -- the
    number llama.cpp / vLLM call "tg" (text generation) throughput, comparable
    across MoE backends because it excludes the one-time prefill cost.
    """

    backend: str
    ttft_s: float
    decode_tokens: int
    decode_s: float

    @property
    def decode_tok_s(self) -> float:
        return self.decode_tokens / self.decode_s if self.decode_s > 0 else float("nan")


def summarize(samples: list[StepTiming]) -> dict[str, dict]:
    """Group repeats by backend and average TTFT / decode tok/s.

    Backends appear in first-seen order (the order the caller benchmarked
    them), not sorted -- so the printed table matches ``--backends`` order.
    """
    order: list[str] = []
    by_backend: dict[str, list[StepTiming]] = {}
    for s in samples:
        if s.backend not in by_backend:
            order.append(s.backend)
            by_backend[s.backend] = []
        by_backend[s.backend].append(s)

    out: dict[str, dict] = {}
    for backend in order:
        rows = by_backend[backend]
        out[backend] = {
            "repeats": len(rows),
            "ttft_s_mean": mean(r.ttft_s for r in rows),
            "decode_tok_s_mean": mean(r.decode_tok_s for r in rows),
            "decode_tokens": rows[0].decode_tokens,
        }
    return out


def format_table(summary: dict[str, dict]) -> str:
    """Render :func:`summarize`'s output as a fixed-width text table."""
    header = f"{'backend':<10} {'ttft_s':>10} {'decode_tok/s':>14} {'decode_tokens':>14} {'repeats':>8}"
    lines = [header, "-" * len(header)]
    for backend, row in summary.items():
        lines.append(
            f"{backend:<10} {row['ttft_s_mean']:>10.3f} {row['decode_tok_s_mean']:>14.2f} "
            f"{row['decode_tokens']:>14} {row['repeats']:>8}"
        )
    return "\n".join(lines)


__all__ = ["StepTiming", "summarize", "format_table"]
