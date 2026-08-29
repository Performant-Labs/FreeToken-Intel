"""Tests for the SYCL kernel toolchain (issue `kernel-sycl`).

CPU-safe tests (run by the per-PR gate, no XPU needed): module imports, the
oneAPI probe, and the AOT cache-key logic (the legs are monkeypatched so the
key behavior is deterministic without a GPU). The ``xpu``-marked tests (run by
the B70 nightly / preflight) actually compile and run the hello-copy kernel and
verify the "cache hits skip recompile across processes" acceptance criterion.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

xpu_only = pytest.mark.xpu


# --- Toolchain ----------------------------------------------------------------

def test_toolchain_module_imports():
    import freetoken.kernel._toolchain as tc  # noqa: F401

    for name in ("find_icpx", "sycl_flags", "icpx_version", "ToolchainError"):
        assert hasattr(tc, name), name
    assert issubclass(tc.ToolchainError, RuntimeError)


def test_find_icpx_missing_raises_clearly(monkeypatch):
    import freetoken.kernel._toolchain as tc

    for var in ("PATH", "FREETOKEN_ICPX", "ONEAPI_ROOT", "CMPLR_ROOT"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(tc.ToolchainError):
        tc.find_icpx()


def test_find_icpx_uses_explicit_override(monkeypatch, tmp_path):
    import freetoken.kernel._toolchain as tc

    fake = tmp_path / "icpx"
    fake.write_text("#!/bin/sh\n")
    os.chmod(fake, 0o755)
    monkeypatch.setenv("FREETOKEN_ICPX", str(fake))
    assert tc.find_icpx() == fake


def test_sycl_flags_requires_toolchain(monkeypatch):
    import freetoken.kernel._toolchain as tc

    for var in ("PATH", "FREETOKEN_ICPX", "ONEAPI_ROOT", "CMPLR_ROOT"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(tc.ToolchainError):
        tc.sycl_flags()


# --- Backend probes ------------------------------------------------------------

def test_backend_probe_names_exported():
    import freetoken.kernel as k

    for name in (
        "is_oneapi_dpcpp_installed",
        "is_triton_intel_installed",
        "is_ipex_installed",
        "is_sycl_ext_installed",
        "level_zero_driver_version",
    ):
        assert hasattr(k, name), name


def test_sycl_ext_alias_maps_to_toolchain_probe():
    import freetoken.kernel as k

    # is_sycl_ext_installed is a deprecated alias: it reports whether the oneAPI
    # toolchain is installed (it used to probe a non-existent python module).
    assert k.is_sycl_ext_installed() == k.is_oneapi_dpcpp_installed()


# --- AOT cache key (CPU-safe; legs monkeypatched) -----------------------------

def test_aot_cache_key_shape():
    import freetoken.kernel.utils as utils

    key = utils._build_key("hello_copy", "kernels/sycl/hello_copy.cpp")
    assert key.startswith("hello_copy-")
    # The hash suffix is 16 hex chars.
    assert len(key.split("-", 1)[1]) == 16


def test_aot_cache_key_changes_when_a_leg_changes(monkeypatch):
    import freetoken.kernel.aot as aot
    import freetoken.kernel.utils as utils

    src = "kernels/sycl/hello_copy.cpp"
    key = utils._build_key("hello_copy", src)

    # Perturb the legs the key is built from -> the key must change (a toolchain
    # / driver / ISA change forces a rebuild instead of loading a stale,
    # unloadable module). The oneapi leg is read via the toolchain, so patch that;
    # the driver/isa legs live in aot, so patch those.
    monkeypatch.setattr(utils, "_oneapi_version_for_key", lambda: "forged-compiler")
    monkeypatch.setattr(aot, "_driver_version", lambda: "forged-driver")
    monkeypatch.setattr(aot, "_isa_name", lambda: "lnc")
    changed = utils._build_key("hello_copy", src)
    assert changed != key


def test_aot_cache_key_deterministic():
    import freetoken.kernel.utils as utils

    src = "kernels/sycl/hello_copy.cpp"
    assert utils._build_key("hello_copy", src) == utils._build_key("hello_copy", src)


def test_aot_cache_key_source_change_changes_key(monkeypatch, tmp_path):
    import freetoken.kernel.aot as aot
    import freetoken.kernel.utils as utils

    # Two real sources with different bytes; freeze every other leg to a constant
    # so only the source fingerprint differs. (Missing files hash to the same
    # empty digest, so the sources must exist to exercise the fingerprint leg.)
    a = tmp_path / "a.cpp"
    a.write_text("int a = 1;")
    b = tmp_path / "b.cpp"
    b.write_text("int b = 2;")
    monkeypatch.setattr(utils, "_oneapi_version_for_key", lambda: "fixed")
    monkeypatch.setattr(aot, "_driver_version", lambda: "fixed")
    monkeypatch.setattr(aot, "_isa_name", lambda: "fixed")
    assert utils._build_key("hello_copy", str(a)) != utils._build_key("hello_copy", str(b))


# --- utils load / guards (CPU-safe) ------------------------------------------

def test_get_xpu_stream_none_without_xpu(monkeypatch):
    # With no torch importable (CPU venv) get_xpu_stream returns None, not raise.
    import freetoken.kernel.utils as utils

    try:
        import torch  # noqa: F401

        torch_present = True
    except Exception:
        torch_present = False
    if torch_present:
        pytest.skip("torch present; the no-torch path is not exercised in this venv")
    assert utils.get_xpu_stream() is None


def test_hello_copy_missing_source_raises(monkeypatch, tmp_path):
    import freetoken.kernel.utils as utils

    monkeypatch.setattr(utils, "HELLO_COPY_SRC", tmp_path / "nope.cpp")
    with pytest.raises(Exception):
        utils.hello_copy()


# --- XPU smoke test (B70 nightly / preflight) -------------------------------

@xpu_only
def test_hello_copy_compiles_and_runs_on_xpu():
    import torch

    assert torch.xpu.is_available()
    import freetoken.kernel.utils as utils

    module = utils.hello_copy()
    assert utils.run_hello_copy(module, count=1024) == 1024


@xpu_only
def test_hello_copy_cache_hit_skips_recompile(tmp_path):
    # Two separate interpreter invocations sharing one cache dir: the first
    # compiles (cold), the second must report a cache hit (the issue's
    # "cache hits skip recompile across processes" acceptance criterion).
    import freetoken.kernel.utils as utils  # noqa: F401  (drives HELLO_COPY_SRC)

    cache_dir = tmp_path / "cache"
    env = {**os.environ, "FREETOKEN_KERNEL_CACHE_DIR": str(cache_dir)}
    code = (
        "import freetoken.kernel.utils as u\n"
        "m = u.hello_copy()\n"
        "n = u.run_hello_copy(m, count=64)\n"
        "assert n == 64, n\n"
        "print('FROM_CACHE', m.from_cache)\n"
    )
    first = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert "FROM_CACHE False" in first.stdout, "first run should be a cold compile"

    second = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    assert "FROM_CACHE True" in second.stdout, "second process should be a cache hit"
