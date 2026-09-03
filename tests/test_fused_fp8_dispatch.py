"""``prefer_fused_over_dequant`` (fp8): the measured fused-vs-fallback
crossover for block-FP8 (issue `moe-quant-banks-native-multi`, #163). Pure
Python logic, no torch/XPU needed -- companion to the real hardware
correctness/perf tests in test_fused_fp8_linear_xpu.py.
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

from freetoken.kernel.triton.fused_fp8_linear import prefer_fused_over_dequant


def test_prefers_fused_at_and_below_the_measured_crossover():
    for m in (1, 2, 4, 8):
        assert prefer_fused_over_dequant(m) is True


def test_prefers_fallback_above_the_measured_crossover():
    for m in (9, 16, 32, 64):
        assert prefer_fused_over_dequant(m) is False
