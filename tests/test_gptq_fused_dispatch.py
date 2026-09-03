"""``prefer_fused_over_dequant``: the measured fused-vs-fallback crossover
for GPTQ-Int4 (issue `moe-quant-banks-native`, #139). Pure Python logic, no
torch/XPU needed -- companion to the real hardware correctness/perf tests in
test_gptq_fused_linear_xpu.py.
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

from freetoken.kernel.triton.gptq_fused_linear import prefer_fused_over_dequant


def test_prefers_fused_at_and_below_the_measured_crossover():
    for m in (1, 2, 4, 8, 16, 32):
        assert prefer_fused_over_dequant(m) is True


def test_prefers_fallback_above_the_measured_crossover():
    for m in (33, 64, 128):
        assert prefer_fused_over_dequant(m) is False
