"""XPU smoke tests for the XPU nightly (``xpu.yml``).

The per-PR ``ci`` job runs on a torch-free CPU venv, so the torch / ``torch.xpu``
path has no per-PR coverage. The XPU nightly runs on the B70 fleet runner with
``.venv-xpu`` activated, and this module is the first ``xpu``-marked test: the
nightly's payload must not be empty, and these are the cheapest meaningful
assertions about "the XPU path is alive" -- the device is visible to torch and
the ``ft device`` report says so.

Every test here is ``xpu``-marked, so ``conftest.py``'s policy deselects the
whole module on a torch-less (CPU) venv -- these only ever run where a real
XPU is present. The module-level ``importorskip`` is the belt; the marker is
the suspenders (and is what ``-m xpu`` actually selects on).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

# xpu-marked: the nightly runs these; the CPU per-PR job deselects them.
pytestmark = pytest.mark.xpu


def test_torch_xpu_is_available():
    assert torch.xpu.is_available(), "torch.xpu.is_available() is False on the XPU nightly runner"
    assert torch.xpu.device_count() >= 1, "torch sees zero XPU devices"


def test_b70_device_name_is_reported():
    from freetoken.utils.arch import xpu_device_name

    name = xpu_device_name()
    assert name, "freetoken.utils.arch.xpu_device_name() returned None on the XPU nightly runner"


def test_ft_device_reports_xpu_available(capsys):
    from freetoken.cli import main

    assert main(["device"]) == 0
    out = capsys.readouterr().out
    assert "torch.xpu available: True" in out
