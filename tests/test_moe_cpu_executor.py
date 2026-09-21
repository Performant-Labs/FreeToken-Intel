"""Tests for ``CpuMoeExecutor``'s native-GEMM wiring (issue #291).

Before this issue ``CpuMoeExecutor.forward`` was always the pure-PyTorch
expert-by-expert loop, and ``threads`` was parsed but never reached a GEMM.
These tests spy the native entry point (``cpu_moe_forward_fast``, #252) to
show ``--moe-backend cpu`` calls it -- including at ``threads=0`` -- and that
the pure-Python loop survives only as the fallback when the native module
can't be built.

CPU-safe: the "native module present" tests never compile real C++ (they
monkeypatch ``cpu_moe``/``cpu_moe_forward_fast`` directly), so they run
without a C++ toolchain. The real compiled-kernel numerics are covered by
``test_kernel_cpu_moe.py``.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from freetoken.kernel._toolchain import ToolchainError  # noqa: E402
from freetoken.moe.cpu_executor import CpuMoeExecutor  # noqa: E402


def _synthetic_case(seed: int = 0):
    torch.manual_seed(seed)
    num_tokens, hidden, intermediate, num_experts, topk = 7, 6, 5, 4, 2
    flat = torch.randn(num_tokens, hidden, dtype=torch.float32)
    gate_up = torch.randn(num_experts, 2 * intermediate, hidden, dtype=torch.float32)
    down = torch.randn(num_experts, hidden, intermediate, dtype=torch.float32)
    top_idx = torch.randint(0, num_experts, (num_tokens, topk))
    top_w = torch.rand(num_tokens, topk)
    top_w = top_w / top_w.sum(dim=1, keepdim=True)
    return flat, top_idx, top_w, gate_up, down, num_experts, intermediate


def _reference_forward(flat, top_idx, top_w, gate_up, down, num_experts, intermediate):
    T, H = flat.shape
    k = top_idx.shape[1]
    out = torch.zeros(T, H, dtype=torch.float32)
    I = intermediate
    for e in range(num_experts):
        for j in range(k):
            sel = top_idx[:, j] == e
            if not bool(sel.any()):
                continue
            idx = sel.nonzero(as_tuple=True)[0]
            x_sel = flat.index_select(0, idx)
            gate = x_sel @ gate_up[e, :I].t()
            up = x_sel @ gate_up[e, I : 2 * I].t()
            y = (F.silu(gate) * up) @ down[e].t()
            w = top_w.index_select(0, idx)[:, j, None]
            out.index_add_(0, idx, w * y)
    return out


def test_native_module_used_when_available(monkeypatch):
    """Issue #291 accept: moe_backend=cpu calls the native entry point when
    the module compiled/loaded successfully."""
    import freetoken.moe.cpu_executor as cpu_executor_mod

    sentinel_module = object()
    monkeypatch.setattr(cpu_executor_mod, "cpu_moe", lambda *a, **kw: sentinel_module)

    calls = []

    def _spy(module, flat, top_idx, top_w, gate_up, down, num_experts, intermediate, *, expert_mask=None, threads=0):
        calls.append({"module": module, "threads": threads, "expert_mask": expert_mask})
        return torch.zeros_like(flat)

    monkeypatch.setattr(cpu_executor_mod, "cpu_moe_forward_fast", _spy)
    # If the native path is skipped, this would run instead -- fail loudly.
    monkeypatch.setattr(
        CpuMoeExecutor,
        "_python_forward",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("python fallback must not run")),
    )

    executor = CpuMoeExecutor(num_experts=4, intermediate=5, threads=7)
    assert executor._native_module is sentinel_module

    flat, top_idx, top_w, gate_up, down, num_experts, intermediate = _synthetic_case()
    executor.forward(flat, top_idx, top_w, gate_up, down)

    assert len(calls) == 1
    assert calls[0]["module"] is sentinel_module
    assert calls[0]["threads"] == 7


def test_threads_zero_still_uses_native_path_not_python_loop(monkeypatch):
    """Issue #291 accept: threads=0 must reach the native pool (its own
    "auto" width), not silently fall back to the pure-PyTorch loop."""
    import freetoken.moe.cpu_executor as cpu_executor_mod

    monkeypatch.setattr(cpu_executor_mod, "cpu_moe", lambda *a, **kw: object())

    calls = []

    def _spy(module, flat, top_idx, top_w, gate_up, down, num_experts, intermediate, *, expert_mask=None, threads=0):
        calls.append(threads)
        return torch.zeros_like(flat)

    monkeypatch.setattr(cpu_executor_mod, "cpu_moe_forward_fast", _spy)
    monkeypatch.setattr(
        CpuMoeExecutor,
        "_python_forward",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("python fallback must not run at threads=0")),
    )

    executor = CpuMoeExecutor(num_experts=4, intermediate=5, threads=0)
    flat, top_idx, top_w, gate_up, down, num_experts, intermediate = _synthetic_case()
    executor.forward(flat, top_idx, top_w, gate_up, down)

    assert calls == [0]


def test_falls_back_to_python_loop_when_toolchain_missing(monkeypatch):
    """Issue #291 accept: with the native module absent, the PyTorch fallback
    still returns the same expert-major accumulation as before this issue."""
    import freetoken.moe.cpu_executor as cpu_executor_mod

    def _raise(*a, **kw):
        raise ToolchainError("no compiler in this sandbox")

    monkeypatch.setattr(cpu_executor_mod, "cpu_moe", _raise)

    def _must_not_be_called(*a, **kw):
        raise AssertionError("native entry point must not be called without a native module")

    monkeypatch.setattr(cpu_executor_mod, "cpu_moe_forward_fast", _must_not_be_called)

    executor = CpuMoeExecutor(num_experts=4, intermediate=5, threads=4)
    assert executor._native_module is None

    flat, top_idx, top_w, gate_up, down, num_experts, intermediate = _synthetic_case(seed=1)
    actual = executor.forward(flat, top_idx, top_w, gate_up, down)
    expected = _reference_forward(flat, top_idx, top_w, gate_up, down, num_experts, intermediate)

    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)
