"""device-layer (#2): the B70/Xe2 device probe must work on any box.

The Level Zero driver probe and the dtype resolver are exercised here. The
driver probe is CPU-safe (returns ``None`` without a GPU), so it runs in the
torch-free per-PR venv; the dtype resolver needs torch, so it is ``xpu``-marked
and runs on the B70 nightly.
"""
from __future__ import annotations

import pytest

from freetoken.kernel.pinned import driver_version
from freetoken.utils.arch import (
    is_xe2_family,
    is_xpu_available,
    level_zero_driver_version,
    xpu_total_memory,
)


def test_driver_version_returns_str_or_none_without_gpu():
    # No hard torch dependency: importable and callable on a CPU-only box.
    v = level_zero_driver_version()
    assert v is None or isinstance(v, str)
    assert isinstance(is_xe2_family(), bool)
    assert isinstance(is_xpu_available(), bool)


def test_pinned_driver_version_matches_arch_probe():
    # kernel.pinned.driver_version is the offload-path alias for the same
    # Level Zero driver version read by utils.arch -- the two must agree.
    assert driver_version() == level_zero_driver_version()


def test_xpu_total_memory_is_positive_when_available():
    if is_xpu_available():
        assert isinstance(xpu_total_memory(), int)
        assert xpu_total_memory() > 0


def test_dtype_resolver_is_torch_marked_runs_on_nightly():
    pytest.importorskip("torch")
    from freetoken.utils.torch_utils import torch_dtype

    import torch

    assert torch_dtype("bfloat16") is torch.bfloat16
    assert torch_dtype("float32") is torch.float32
    with pytest.raises(ValueError, match="unknown dtype"):
        torch_dtype("not_a_real_dtype")
