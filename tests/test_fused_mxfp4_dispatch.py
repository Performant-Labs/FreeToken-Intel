"""``prefer_fused_over_dequant`` (mxfp4): the measured fused-vs-fallback
window for MXFP4 (issue `moe-quant-banks-native-multi`, #163). Pure Python
logic, no torch/XPU needed -- companion to the real hardware correctness/
perf tests in test_fused_mxfp4_linear_xpu.py.
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

from freetoken.kernel.triton.fused_mxfp4_linear import prefer_fused_over_dequant


def test_prefers_fused_within_the_measured_window():
    for m in (1, 4, 8, 16, 32, 64, 128):
        assert prefer_fused_over_dequant(m) is True


def test_prefers_fallback_beyond_the_measured_upper_bound():
    for m in (129, 256, 512):
        assert prefer_fused_over_dequant(m) is False
