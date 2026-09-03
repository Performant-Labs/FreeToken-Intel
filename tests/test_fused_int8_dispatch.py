"""``prefer_fused_over_dequant`` (int8): the measured fused-vs-fallback
crossover for compressed-tensors pack-quantized INT8 (issue `moe-quant-
banks-native-multi`, #163). Pure Python logic, no torch/XPU needed --
companion to the real hardware correctness/perf tests in
test_fused_int8_linear_xpu.py.
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

from freetoken.kernel.triton.fused_int8_linear import prefer_fused_over_dequant


def test_prefers_fused_at_and_below_the_measured_crossover():
    for m in (1, 4, 8, 16, 32, 64):
        assert prefer_fused_over_dequant(m) is True


def test_prefers_fallback_above_the_measured_crossover():
    for m in (65, 128, 256):
        assert prefer_fused_over_dequant(m) is False
