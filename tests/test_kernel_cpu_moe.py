"""Tests for the native CPU MoE expert-GEMM kernel (issue #250).

CPU-safe: no XPU needed. The native path is compared against the pure-Python
reference math it ports (``CpuMoeExecutor.forward`` /
``_Qwen3MoE._cpu_subset_math``), reimplemented inline here from the same
SwiGLU-and-accumulation-order description both those functions document, so
this test does not depend on constructing a full engine/model just to reach
the math.

Compiling and loading the kernel needs a real C++ compiler (``cpu_moe.py``
looks for ``FREETOKEN_CXX`` / ``c++`` / ``g++`` / ``clang++`` -- see that
module's docstring for why this is a plain system toolchain lookup, not
``icpx``). When none is found, :func:`freetoken.kernel.cpu_moe.cpu_moe`
raises :class:`~freetoken.kernel._toolchain.ToolchainError`, which this file
turns into a graceful skip (never a failure/error) -- mirroring how the SYCL
kernel tests treat an absent toolchain in ``test_kernel_toolchain.py``.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from freetoken.kernel._toolchain import ToolchainError  # noqa: E402


@pytest.fixture(scope="module")
def native_module():
    from freetoken.kernel.cpu_moe import cpu_moe, find_cxx_compiler

    # Only a missing toolchain is a skip. cpu_moe() itself can also raise
    # ToolchainError when the *compile* fails (a real bug in cpu_moe.cpp),
    # and that must fail this test, not skip it silently -- catching it here
    # too would let a broken kernel build merge with green CI.
    try:
        find_cxx_compiler()
    except ToolchainError as exc:
        pytest.skip(f"no C++ compiler available for the native CPU MoE kernel: {exc}")
    return cpu_moe()


def _reference_forward(x, top_idx, top_w, gate_up, down, num_experts, intermediate, candidates):
    """Expert-major then top-k-column SwiGLU reference (mirrors both Python sources).

    ``candidates`` restricts which experts contribute (``None`` = every
    expert, matching ``CpuMoeExecutor.forward``; a concrete set matches
    ``_Qwen3MoE._cpu_subset_math``'s ``cpu_experts`` restriction).
    """
    T, H = x.shape
    k = top_idx.shape[1]
    out = torch.zeros(T, H, dtype=torch.float32)
    I = intermediate
    for e in range(num_experts):
        if candidates is not None and e not in candidates:
            continue
        for j in range(k):
            sel = top_idx[:, j] == e
            if not bool(sel.any()):
                continue
            idx = sel.nonzero(as_tuple=True)[0]
            x_sel = x.index_select(0, idx)
            gate = x_sel @ gate_up[e, :I].t()
            up = x_sel @ gate_up[e, I : 2 * I].t()
            y = (F.silu(gate) * up) @ down[e].t()
            w = top_w.index_select(0, idx)[:, j, None]
            out.index_add_(0, idx, w * y)
    return out


def _synthetic_case(seed: int = 0):
    torch.manual_seed(seed)
    num_tokens, hidden, intermediate, num_experts, topk = 7, 6, 5, 4, 2
    x = torch.randn(num_tokens, hidden, dtype=torch.float32)
    gate_up = torch.randn(num_experts, 2 * intermediate, hidden, dtype=torch.float32)
    down = torch.randn(num_experts, hidden, intermediate, dtype=torch.float32)
    top_idx = torch.randint(0, num_experts, (num_tokens, topk))
    top_w = torch.rand(num_tokens, topk)
    top_w = top_w / top_w.sum(dim=1, keepdim=True)
    return x, top_idx, top_w, gate_up, down, num_experts, intermediate


def test_native_cpu_moe_matches_reference_full_backend(native_module):
    """expert_mask=None (the plain ``cpu`` backend's use case: every expert a candidate)."""
    from freetoken.kernel.cpu_moe import cpu_moe_forward

    x, top_idx, top_w, gate_up, down, num_experts, intermediate = _synthetic_case(seed=0)

    expected = _reference_forward(x, top_idx, top_w, gate_up, down, num_experts, intermediate, None)
    actual = cpu_moe_forward(
        native_module, x, top_idx, top_w, gate_up, down, num_experts, intermediate, expert_mask=None
    )

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-5), (actual - expected).abs().max()


def test_native_cpu_moe_matches_reference_subset(native_module):
    """expert_mask names a subset (the hybrid split's CPU-computed ``cpu_experts``)."""
    from freetoken.kernel.cpu_moe import cpu_moe_forward

    x, top_idx, top_w, gate_up, down, num_experts, intermediate = _synthetic_case(seed=1)
    cpu_experts = {0, 2}

    expected = _reference_forward(
        x, top_idx, top_w, gate_up, down, num_experts, intermediate, cpu_experts
    )
    actual = cpu_moe_forward(
        native_module,
        x,
        top_idx,
        top_w,
        gate_up,
        down,
        num_experts,
        intermediate,
        expert_mask=cpu_experts,
    )

    assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-5), (actual - expected).abs().max()
    # Rows that only routed to non-candidate experts must be exactly zero (the
    # hybrid split's contract: a row not served by this half contributes
    # nothing, the XPU half serves it instead).
    fully_excluded_rows = ~torch.isin(top_idx, torch.tensor(sorted(cpu_experts))).any(dim=1)
    if bool(fully_excluded_rows.any()):
        assert torch.all(actual[fully_excluded_rows] == 0)


def test_native_cpu_moe_empty_expert_set_is_zero(native_module):
    """An empty candidate set must return all zeros (mirrors ``_cpu_subset_math``'s early return)."""
    from freetoken.kernel.cpu_moe import cpu_moe_forward

    x, top_idx, top_w, gate_up, down, num_experts, intermediate = _synthetic_case(seed=2)

    actual = cpu_moe_forward(
        native_module,
        x,
        top_idx,
        top_w,
        gate_up,
        down,
        num_experts,
        intermediate,
        expert_mask=set(),
    )
    assert torch.all(actual == 0)


def _host_has_avx512() -> bool:
    """Best-effort runtime probe from Python, independent of the native module.

    Used to decide, from the test side, whether the AVX-512-specific
    assertions below are even exercisable on this sandbox -- mirrors issue
    #250's "skip what the host can't do" pattern applied to an ISA feature
    instead of a missing toolchain. Reads ``/proc/cpuinfo`` (Linux) rather
    than assuming ``platform`` exposes CPU flags, since it does not.
    """
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("flags") or line.startswith("Features"):
                    return "avx512f" in line.split(":", 1)[1].split()
    except OSError:
        pass
    return False


def test_native_cpu_moe_fast_avx512_probe_matches_module(native_module):
    """The C++ runtime probe and the Python /proc/cpuinfo probe must agree."""
    from freetoken.kernel.cpu_moe import cpu_moe_avx512_available

    assert cpu_moe_avx512_available(native_module) == _host_has_avx512()


@pytest.mark.parametrize("threads", [1, 0, 4])
def test_native_cpu_moe_fast_matches_naive_full_backend(native_module, threads):
    """Fast path (whatever this host supports) matches the naive scalar path."""
    from freetoken.kernel.cpu_moe import cpu_moe_forward, cpu_moe_forward_fast

    x, top_idx, top_w, gate_up, down, num_experts, intermediate = _synthetic_case(seed=3)

    expected = cpu_moe_forward(
        native_module, x, top_idx, top_w, gate_up, down, num_experts, intermediate, expert_mask=None
    )
    actual = cpu_moe_forward_fast(
        native_module, x, top_idx, top_w, gate_up, down, num_experts, intermediate,
        expert_mask=None, threads=threads,
    )
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, rtol=1e-3, atol=1e-4), (actual - expected).abs().max()


@pytest.mark.parametrize("threads", [1, 4])
def test_native_cpu_moe_fast_matches_naive_subset(native_module, threads):
    from freetoken.kernel.cpu_moe import cpu_moe_forward, cpu_moe_forward_fast

    x, top_idx, top_w, gate_up, down, num_experts, intermediate = _synthetic_case(seed=4)
    cpu_experts = {0, 2}

    expected = cpu_moe_forward(
        native_module, x, top_idx, top_w, gate_up, down, num_experts, intermediate,
        expert_mask=cpu_experts,
    )
    actual = cpu_moe_forward_fast(
        native_module, x, top_idx, top_w, gate_up, down, num_experts, intermediate,
        expert_mask=cpu_experts, threads=threads,
    )
    assert torch.allclose(actual, expected, rtol=1e-3, atol=1e-4), (actual - expected).abs().max()


def test_native_cpu_moe_fast_forced_scalar_matches_naive(native_module):
    """force_scalar=True must reproduce the naive path's exact scalar math,
    single-threaded, regardless of what this host's CPU actually supports."""
    from freetoken.kernel.cpu_moe import cpu_moe_forward, cpu_moe_forward_fast

    x, top_idx, top_w, gate_up, down, num_experts, intermediate = _synthetic_case(seed=5)

    expected = cpu_moe_forward(
        native_module, x, top_idx, top_w, gate_up, down, num_experts, intermediate, expert_mask=None
    )
    actual = cpu_moe_forward_fast(
        native_module, x, top_idx, top_w, gate_up, down, num_experts, intermediate,
        expert_mask=None, threads=1, force_scalar=True,
    )
    # Forced-scalar, single-threaded fast path shares the exact same (e, j, t)
    # loop order and row math as the naive path -- tight tolerance.
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6), (actual - expected).abs().max()


def test_native_cpu_moe_fast_avx512_path_matches_scalar_when_available(native_module):
    """Exercise the actual AVX-512 branch only when this sandbox's CPU has it."""
    from freetoken.kernel.cpu_moe import cpu_moe_avx512_available, cpu_moe_forward_fast

    if not cpu_moe_avx512_available(native_module):
        pytest.skip("this sandbox's CPU does not support AVX-512F -- nothing to exercise")

    x, top_idx, top_w, gate_up, down, num_experts, intermediate = _synthetic_case(seed=6)

    scalar = cpu_moe_forward_fast(
        native_module, x, top_idx, top_w, gate_up, down, num_experts, intermediate,
        expert_mask=None, threads=1, force_scalar=True,
    )
    vectorized = cpu_moe_forward_fast(
        native_module, x, top_idx, top_w, gate_up, down, num_experts, intermediate,
        expert_mask=None, threads=1, force_scalar=False,
    )
    assert torch.allclose(vectorized, scalar, rtol=1e-3, atol=1e-4), (vectorized - scalar).abs().max()


def test_native_cpu_moe_fast_env_var_forces_scalar(native_module, monkeypatch):
    from freetoken.kernel.cpu_moe import cpu_moe_forward, cpu_moe_forward_fast

    monkeypatch.setenv("FREETOKEN_CPU_MOE_FORCE_SCALAR", "1")
    x, top_idx, top_w, gate_up, down, num_experts, intermediate = _synthetic_case(seed=7)

    expected = cpu_moe_forward(
        native_module, x, top_idx, top_w, gate_up, down, num_experts, intermediate, expert_mask=None
    )
    actual = cpu_moe_forward_fast(
        native_module, x, top_idx, top_w, gate_up, down, num_experts, intermediate,
        expert_mask=None, threads=1,
    )
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6), (actual - expected).abs().max()


def test_native_cpu_moe_fast_empty_expert_set_is_zero(native_module):
    from freetoken.kernel.cpu_moe import cpu_moe_forward_fast

    x, top_idx, top_w, gate_up, down, num_experts, intermediate = _synthetic_case(seed=8)

    actual = cpu_moe_forward_fast(
        native_module, x, top_idx, top_w, gate_up, down, num_experts, intermediate,
        expert_mask=set(), threads=4,
    )
    assert torch.all(actual == 0)


def test_cpu_moe_module_cache_hit_skips_recompile(tmp_path, monkeypatch):
    """Same "cache hit skips recompile" shape as ``kernel.utils.hello_copy``."""
    import os
    import subprocess
    import sys

    cache_dir = tmp_path / "cache"
    env = {**os.environ, "FREETOKEN_JIT_CACHE_DIR": str(cache_dir)}
    # Only find_cxx_compiler() failing is a skip (no toolchain). cm.cpu_moe()
    # itself runs outside that try/except, so a real compile failure crashes
    # the subprocess (non-zero exit) and the assert below on `returncode == 0`
    # fails loudly with the compiler's stderr, instead of this test silently
    # skipping on a broken kernel build.
    code = (
        "import freetoken.kernel.cpu_moe as cm\n"
        "from freetoken.kernel._toolchain import ToolchainError\n"
        "try:\n"
        "    cm.find_cxx_compiler()\n"
        "except ToolchainError:\n"
        "    print('SKIP')\n"
        "else:\n"
        "    m = cm.cpu_moe()\n"
        "    print('FROM_CACHE', m.from_cache)\n"
    )
    first = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    if "SKIP" in first.stdout:
        pytest.skip("no C++ compiler available for the native CPU MoE kernel")
    assert "FROM_CACHE False" in first.stdout, "first run should be a cold compile"

    second = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    assert "FROM_CACHE True" in second.stdout, "second process should be a cache hit"
