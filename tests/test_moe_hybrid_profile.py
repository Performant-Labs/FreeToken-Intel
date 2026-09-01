"""Torch-free tests for the ``ft bench bw`` profile reader (issue #9, moe-hybrid).

The reader (``freetoken.moe.bench_profile``) is the engine's startup seam: given a
per-XPU profile JSON (written by ``ft bench bw``), it (1) recommends ``hybrid``
over ``offload`` when the profile says the CPU MoE bandwidth beat the PCIe gather
by the threshold, and (2) yields the hybrid backend's per-step fetch fraction
(``q*``: pcie/(pcie+cpu) under overlap).

This file is CPU-venv (no torch): the reader is torch-free by contract (the CPU
venv must load it without a device, since the loader consults it to resolve
``--moe-backend auto`` before any XPU is touched). Tests pin a profile in
``tmp_path`` and pass it explicitly (``path=`` / ``FREETOKEN_BENCHBW_PATH``) so the
per-XPU identity lookup is bypassed -- an explicit path is trusted without a
name/uuid match, which is exactly the "I benched on a specific card" case.
"""
from __future__ import annotations

import json
import os

import pytest

from freetoken.moe.bench_profile import (
    default_profile_path,
    latest_profile_path,
    load_backend_recommendation,
    load_hybrid_fetch_fraction,
    resolve_backend,
)
from freetoken.moe import resolve_moe_backend

# A profile whose bf16 workload says the CPU MoE GEMV beat the PCIe gather 3:1
# (above the 2x threshold -> hybrid) and, under overlap, the two halves ran at a
# 4:2 (pcie:cpu) effective ratio -> the q* fetch fraction is 0.4.
def _hybrid_profile() -> dict:
    return {
        "xpu": {"name": "Test XPU", "uuid": "uuid-0"},
        "dtypes": {"bf16": "hybrid"},
        "dtype_kernels": {
            "bf16": {
                "cpu_moe_gbs": 30.0,
                "pcie_gather_gbs": 10.0,
                "cpu_moe_overlap_gbs": 20.0,
                "pcie_gather_overlap_gbs": 8.0,
            }
        },
    }


def _offload_profile() -> dict:
    """CPU MoE BW only 1.2x the PCIe gather (under the 2x threshold -> offload)."""
    return {
        "xpu": {"name": "Test XPU", "uuid": "uuid-0"},
        "dtypes": {"bf16": "offload"},
        "dtype_kernels": {
            "bf16": {
                "cpu_moe_gbs": 12.0,
                "pcie_gather_gbs": 10.0,
            }
        },
    }


def _write(tmp_path, prof: dict) -> str:
    path = str(tmp_path / "prof.json")
    with open(path, "w") as f:
        json.dump(prof, f)
    return path


def test_profile_path_keyed_per_xpu_uuid(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    p = default_profile_path("abcd-1234")
    assert p == str(tmp_path / "freetoken" / "benchbw" / "abcd-1234.json")
    # No uuid -> the legacy single-file path.
    assert default_profile_path() == str(tmp_path / "freetoken" / "benchbw.json")


def test_latest_profile_path_prefers_per_xpu(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    legacy = tmp_path / "freetoken" / "benchbw.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("{}")
    per = tmp_path / "freetoken" / "benchbw"
    per.mkdir()
    (per / "u1.json").write_text("{}")
    # The per-XPU file is newer than the legacy one -> wins.
    (per / "u1.json").touch()
    assert latest_profile_path() == str(per / "u1.json")


def test_latest_profile_path_falls_back_to_legacy(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    legacy = tmp_path / "freetoken" / "benchbw.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("{}")
    assert latest_profile_path() == str(legacy)


def test_latest_profile_path_none_when_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    (tmp_path / "freetoken").mkdir()
    assert latest_profile_path() is None


def test_recommends_hybrid_when_profile_says_so(tmp_path):
    path = _write(tmp_path, _hybrid_profile())
    assert load_backend_recommendation("bf16", path=path) == "hybrid"


def test_recommends_offload_when_below_threshold(tmp_path):
    path = _write(tmp_path, _offload_profile())
    assert load_backend_recommendation("bf16", path=path) == "offload"


def test_no_entry_for_format_is_none(tmp_path):
    path = _write(tmp_path, _hybrid_profile())
    # The profile has only a bf16 entry; an fp8_block query finds none -> None.
    assert load_backend_recommendation("fp8_block", path=path) is None


def test_mixed_verdict_resolves_to_offload(tmp_path):
    # A per-model profile whose two workloads disagree (one hybrid, one offload):
    # the reader is conservative and resolves the format to offload.
    prof = {
        "xpu": {"name": "Test XPU", "uuid": "uuid-0"},
        "workloads": {
            "qwen3-30b": {"kernels": {"bf16": {"recommended": "hybrid"}}},
            "qwen3-0.6b": {"kernels": {"bf16": {"recommended": "offload"}}},
        },
    }
    path = _write(tmp_path, prof)
    assert load_backend_recommendation("bf16", path=path) == "offload"


def test_name_mismatch_is_ignored(tmp_path):
    # A profile measured on a different XPU (name mismatch) must not be trusted:
    # the reader returns None so the caller keeps its own default (offload).
    prof = _hybrid_profile()
    prof["xpu"]["name"] = "Some Other XPU"
    path = _write(tmp_path, prof)
    # xpu_name pins the *expected* card; the profile's name disagrees -> ignored.
    assert load_backend_recommendation("bf16", xpu_name="Test XPU", path=path) is None


def test_unreadable_profile_is_none(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("not json {")
    assert load_backend_recommendation("bf16", path=path) is None


def test_fetch_fraction_prefers_overlapped_pair(tmp_path):
    # The reader's formula is pcie / (pcie + cpu). With the overlapped pair
    # (pcie_ov=8, cpu_ov=20) present, the fraction is 8 / (8 + 20) = 0.2857 --
    # and crucially the *overlapped* pair is preferred over the stale standalone
    # pair (10 / (10 + 30) = 0.25), so a profile carrying both must yield the
    # overlap value. An explicit path skips the per-uuid cache lookup, but the
    # profile's own xpu.uuid ("uuid-0") is still consulted as the mismatch key;
    # pinning xpu_uuid to it trusts the profile in either venv.
    path = _write(tmp_path, _hybrid_profile())
    assert load_hybrid_fetch_fraction("bf16", xpu_uuid="uuid-0", path=path) == pytest.approx(8 / 28, abs=1e-9)


def test_fetch_fraction_falls_back_to_standalone(tmp_path):
    # No overlap pair -> the standalone ratio pcie/cpu = 10/30 = 0.333.
    prof = _hybrid_profile()
    prof["dtype_kernels"]["bf16"].pop("cpu_moe_overlap_gbs")
    prof["dtype_kernels"]["bf16"].pop("pcie_gather_overlap_gbs")
    path = _write(tmp_path, prof)
    assert load_hybrid_fetch_fraction("bf16", path=path) == pytest.approx(10.0 / 30.0, abs=1e-9)


def test_fetch_fraction_none_when_no_profile(tmp_path):
    monkeypatch = pytest.MonkeyPatch()
    with monkeypatch.context() as m:
        m.setenv("XDG_CACHE_HOME", str(tmp_path))
        (tmp_path / "freetoken").mkdir()
    assert load_hybrid_fetch_fraction("bf16") is None


def test_fetch_fraction_clamped_to_unit(tmp_path):
    # A pathological profile (cpu 0) would give pcie/0; clamp keeps it in [0, 1].
    prof = {
        "xpu": {"name": "Test XPU", "uuid": "uuid-0"},
        "dtypes": {"bf16": "offload"},
        "dtype_kernels": {"bf16": {"cpu_moe_gbs": 0.0, "pcie_gather_gbs": 10.0}},
    }
    path = _write(tmp_path, prof)
    # cpu is 0.0 (falsy) -> the standalone fallback is skipped -> None, not inf.
    assert load_hybrid_fetch_fraction("bf16", path=path) is None


@pytest.mark.xpu
def test_resolve_backend_upgrades_auto_to_hybrid(monkeypatch, tmp_path):
    # With an XPU present + a hybrid-recommending profile, auto -> hybrid. This is
    # XPU-marked: the loader (and the reader's auto-lookup) resolve the *real*
    # XPU's name/uuid, so the profile's xpu.name must match the box's actual card
    # or the mismatch filter rejects it. In the torch-free CPU venv this test is
    # deselected at collection (conftest).
    from freetoken.utils.arch import xpu_device_name

    real_name = xpu_device_name()
    if real_name is None:  # no real XPU on this box (deselected anyway, but be safe)
        pytest.skip("no XPU on this box")
    prof = _hybrid_profile()
    prof["xpu"] = {"name": real_name, "uuid": "uuid-0"}
    path = _write(tmp_path, prof)
    # Mirror the loader's exact call: no path/uuid, only the env var. The reader's
    # auto-lookup then matches this XPU by name.
    monkeypatch.setenv("FREETOKEN_BENCHBW_PATH", path)
    assert resolve_moe_backend("auto", is_moe=True, quant_format="bf16") == "hybrid"


def test_resolve_backend_keeps_offload_without_profile(monkeypatch, tmp_path):
    import freetoken.utils.arch as arch

    monkeypatch.setattr(arch, "is_xpu_available", lambda: True)
    # No FREETOKEN_BENCHBW_PATH, empty cache dir -> no profile -> offload default.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    (tmp_path / "freetoken").mkdir()
    assert resolve_moe_backend("auto", is_moe=True, quant_format="bf16") == "offload"


def test_resolve_backend_non_moe_is_fused(monkeypatch):
    import freetoken.utils.arch as arch

    monkeypatch.setattr(arch, "is_xpu_available", lambda: True)
    assert resolve_moe_backend("auto", is_moe=False) == "fused"


def test_resolve_backend_explicit_backend_passthrough(monkeypatch):
    import freetoken.utils.arch as arch

    monkeypatch.setattr(arch, "is_xpu_available", lambda: True)
    # An explicit backend is never re-resolved (only "auto" is).
    assert resolve_moe_backend("cpu", is_moe=True, quant_format="bf16") == "cpu"
    assert resolve_moe_backend("offload", is_moe=True, quant_format="bf16") == "offload"


def test_resolve_backend_no_xpu_is_fused(monkeypatch):
    import freetoken.utils.arch as arch

    monkeypatch.setattr(arch, "is_xpu_available", lambda: False)
    assert resolve_moe_backend("auto", is_moe=True, quant_format="bf16") == "fused"


@pytest.mark.xpu
def test_env_var_selects_profile(tmp_path, monkeypatch):
    # The reader consults FREETOKEN_BENCHBW_PATH before the per-uuid default.
    # XPU-marked: the auto-lookup resolves the *real* XPU's name/uuid, so the
    # profile's xpu.name must match the box's card (same as the resolve test).
    # In the torch-free CPU venv this test is deselected at collection.
    from freetoken.utils.arch import xpu_device_name

    real_name = xpu_device_name()
    if real_name is None:
        pytest.skip("no XPU on this box")
    prof = _hybrid_profile()
    prof["xpu"] = {"name": real_name, "uuid": "uuid-0"}
    path = _write(tmp_path, prof)
    monkeypatch.setenv("FREETOKEN_BENCHBW_PATH", path)
    # No path/uuid passed -> the env-var profile is used; with a matching real
    # XPU it must resolve: the recommendation is hybrid, the fraction 8 / 28.
    assert load_backend_recommendation("bf16") == "hybrid"
    assert load_hybrid_fetch_fraction("bf16") == pytest.approx(8 / 28, abs=1e-9)
