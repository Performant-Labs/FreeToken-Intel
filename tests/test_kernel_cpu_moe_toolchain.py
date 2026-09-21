"""Tests for the native CPU MoE kernel's toolchain lookup (issue #250).

CPU-safe, torch-free: only ``freetoken.kernel.cpu_moe``'s compiler-discovery
helpers (``find_cxx_compiler`` / ``cxx_version`` / ``cxx_flags``), not the
kernel itself (that needs torch to build tensors for -- see
``test_kernel_cpu_moe.py``). Mirrors ``test_kernel_toolchain.py``'s pattern
for the oneAPI ``icpx`` lookup: monkeypatch ``PATH``/env so resolution order
and the no-compiler error path are deterministic without depending on what
happens to be installed on the box running the test.
"""
from __future__ import annotations

import os

import pytest


def _clear_cxx_env(monkeypatch):
    monkeypatch.delenv("FREETOKEN_CXX", raising=False)
    monkeypatch.setenv("PATH", "")


def test_find_cxx_compiler_missing_raises_clearly(monkeypatch):
    import freetoken.kernel.cpu_moe as cm

    _clear_cxx_env(monkeypatch)
    with pytest.raises(cm.ToolchainError):
        cm.find_cxx_compiler()


def test_find_cxx_compiler_uses_explicit_override(monkeypatch, tmp_path):
    import freetoken.kernel.cpu_moe as cm

    fake = tmp_path / "my-cxx"
    fake.write_text("#!/bin/sh\n")
    os.chmod(fake, 0o755)
    _clear_cxx_env(monkeypatch)
    monkeypatch.setenv("FREETOKEN_CXX", str(fake))
    assert cm.find_cxx_compiler() == fake


def test_find_cxx_compiler_override_ignored_when_not_executable(monkeypatch, tmp_path):
    import freetoken.kernel.cpu_moe as cm

    fake = tmp_path / "not-a-compiler.txt"
    fake.write_text("nope\n")  # no chmod +x
    _clear_cxx_env(monkeypatch)
    monkeypatch.setenv("FREETOKEN_CXX", str(fake))
    # A non-executable override is not a usable compiler -- falls through to
    # the (empty) PATH search and raises, rather than silently "succeeding"
    # with a path that would fail at subprocess.run time.
    with pytest.raises(cm.ToolchainError):
        cm.find_cxx_compiler()


def test_find_cxx_compiler_falls_back_to_path(monkeypatch, tmp_path):
    import freetoken.kernel.cpu_moe as cm

    # A fake "g++" on PATH, with no FREETOKEN_CXX override set -- exercises
    # the PATH-fallback branch (shutil.which), not the override branch.
    fake = tmp_path / "g++"
    fake.write_text("#!/bin/sh\n")
    os.chmod(fake, 0o755)
    monkeypatch.delenv("FREETOKEN_CXX", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert cm.find_cxx_compiler() == fake


def test_find_cxx_compiler_path_order_prefers_cxx_over_gxx(monkeypatch, tmp_path):
    import freetoken.kernel.cpu_moe as cm

    # Both "c++" and "g++" present on PATH -- resolution order picks "c++"
    # first (the _CXX_CANDIDATES ordering this function documents).
    for name in ("c++", "g++", "clang++"):
        fake = tmp_path / name
        fake.write_text("#!/bin/sh\n")
        os.chmod(fake, 0o755)
    monkeypatch.delenv("FREETOKEN_CXX", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert cm.find_cxx_compiler().name == "c++"


def test_cxx_version_none_when_no_compiler(monkeypatch):
    import freetoken.kernel.cpu_moe as cm

    _clear_cxx_env(monkeypatch)
    assert cm.cxx_version() is None


def test_cxx_flags_requires_toolchain(monkeypatch):
    import freetoken.kernel.cpu_moe as cm

    _clear_cxx_env(monkeypatch)
    with pytest.raises(cm.ToolchainError):
        cm.cxx_flags()


def test_cxx_flags_excludes_simd_flags(monkeypatch, tmp_path):
    import freetoken.kernel.cpu_moe as cm

    fake = tmp_path / "c++"
    fake.write_text("#!/bin/sh\n")
    os.chmod(fake, 0o755)
    monkeypatch.delenv("FREETOKEN_CXX", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    flags = cm.cxx_flags()
    # No -mavx512f/-mamx-* etc. here on purpose (issue #252's job, not #250's) --
    # a regression that starts unconditionally requesting SIMD support here
    # would break this naive build on a box without those ISA extensions.
    assert not any("avx" in f.lower() or "amx" in f.lower() for f in flags)
