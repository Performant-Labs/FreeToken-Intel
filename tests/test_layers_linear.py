"""Tests for the pure-torch tensor-parallel Linear layers (issue #24, WP5).

Upstream ``freetoken.layers.linear`` defines five TP-aware ``nn.Linear``-like
classes (``LinearReplicated``, ``LinearColParallelMerged``, ``LinearQKVMerged``,
``LinearOProj``, ``LinearRowParallel``) and computes ``F.linear(x, weight, bias)``.
The B70 runs TP=1 (a single XPU device), so the ``all_reduce`` branches never fire
and each class is just a ``x @ weight.T (+ bias)`` matmul with a particular
``[local_osize, local_isize]`` weight shape.

This suite drives each class on the B70 against an independent (hand-written)
reference computed as plain ``x @ w.T (+ b)`` (NOT ``F.linear``) with a freshly
random ``w``/``b``. That catches both a wrong matmul orientation (``x @ w`` vs
``x @ w.T``) and a wrong sharded shape: a transposed matmul has a different
``[out, in]`` shape, and a wrong local size produces a wrong output shape.

Weight tensors are allocated as ``torch.empty`` (garbage) in the constructor,
mirroring upstream, so each test seeds a known ``w``/``b`` before calling
``forward`` and the reference uses the same seeded values.

XPU repr hazard: every numerical check reduces to a plain Python float
(``.item()``) *outside* the ``assert`` -- a tensor inside a failing assert hands
pytest's rewriter a tensor to ``repr()`` and on this oneAPI runtime that can
OOM/loop the run (see test_layers_norm.py for the full incident).

The torch-free CPU venv only runs the module/export presence checks below; the
numerical tests are xpu-marked.
"""
from __future__ import annotations

import importlib.util

import pytest

torch = pytest.importorskip("torch")

DEVICE = "xpu"


def test_linear_layer_module_present():
    """No torch needed to check presence (self-skips on the torch-free CPU venv)."""
    spec = importlib.util.find_spec("freetoken.layers.linear")
    assert spec is not None, "freetoken.layers.linear is missing"


def test_package_exports_linear():
    import freetoken.layers as L

    for name in ("LinearReplicated", "LinearColParallelMerged", "LinearQKVMerged", "LinearOProj", "LinearRowParallel"):
        assert hasattr(L, name), f"freetoken.layers must export {name}"


@pytest.fixture(autouse=True)
def _tp1(monkeypatch):
    """Establish the TP=1 distributed context the B70 runs with.

    ``get_tp_info()`` raises until ``set_tp_info`` has been called once, and the
    upstream Linear classes read it in ``__init__``. The B70 is a single XPU
    device (TP=1), so the TP>1 all_reduce branches never fire and each class is a
    plain ``x @ w.T (+ b)``. ``set_tp_info`` refuses to be called twice, so the
    fixture resets the module global back to ``None`` on teardown to keep tests
    independent (and so an unrelated later test sees a clean state).
    """
    from freetoken.distributed import info as _tp
    from freetoken.distributed import set_tp_info

    monkeypatch.setattr(_tp, "_TP_INFO", None, raising=False)
    set_tp_info(rank=0, size=1)
    yield
    monkeypatch.setattr(_tp, "_TP_INFO", None, raising=False)


def _seed(op, seed: int):
    """Fill ``op.weight`` / ``op.bias`` with a known seed (torch.empty is garbage).

    Returns the *on-device* weight/bias the layer will actually use, so the
    hand-written reference (``x @ w.t() (+ b)``) is computed on the same device
    as ``x`` -- a CPU reference would raise a cross-device mm error on the XPU.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    op.weight = (
        torch.randn(op.weight.shape, device="cpu", generator=g) * 0.5
    ).to(device=DEVICE, dtype=op.weight.dtype)
    if op.bias is not None:
        op.bias = (
            torch.randn(op.bias.shape, device="cpu", generator=g) * 0.5
        ).to(device=DEVICE, dtype=op.bias.dtype)
    return op.weight, op.bias


def _in(n: int, d: int) -> "torch.Tensor":
    return torch.randn(n, d, device=DEVICE, dtype=torch.float32)


def _max_abs_err(got, want) -> float:
    return (got.float() - want.float()).abs().max().item()


# --------------------------------------------------------------------------- #
# XPU: drive each class on the B70 against the hand-written x @ w.T (+ b) reference.
# --------------------------------------------------------------------------- #
@pytest.mark.xpu
def test_linear_replicated_matches_reference():
    from freetoken.layers import LinearReplicated

    isz, osz = 16, 8
    op = LinearReplicated(isz, osz, has_bias=True)
    assert op.local_input_size == isz and op.local_output_size == osz
    assert op.weight.shape == (osz, isz) and op.bias.shape == (osz,)
    w, b = _seed(op, 11)
    x = _in(6, isz)
    err = _max_abs_err(op.forward(x), x @ w.t() + b)
    assert err < 1e-5, f"LinearReplicated max-err {err}"


@pytest.mark.xpu
def test_linear_replicated_no_bias():
    from freetoken.layers import LinearReplicated

    isz, osz = 12, 9
    op = LinearReplicated(isz, osz, has_bias=False)
    assert op.bias is None
    w, _ = _seed(op, 12)
    x = _in(5, isz)
    err = _max_abs_err(op.forward(x), x @ w.t())
    assert err < 1e-5, f"LinearReplicated(no-bias) max-err {err}"


@pytest.mark.xpu
def test_linear_col_parallel_merged_full_output():
    from freetoken.layers import LinearColParallelMerged

    isz = 16
    sizes = [10, 14]
    op = LinearColParallelMerged(isz, sizes, has_bias=True)
    # TP=1 -> local == full: output is the concatenated (sum) size, input full.
    assert op.local_output_size == sum(sizes)
    assert op.local_input_size == isz
    assert op.weight.shape == (sum(sizes), isz)
    w, b = _seed(op, 13)
    x = _in(7, isz)
    err = _max_abs_err(op.forward(x), x @ w.t() + b)
    assert err < 1e-5, f"LinearColParallelMerged max-err {err}"


@pytest.mark.xpu
def test_linear_qkv_merged_shape_and_math():
    from freetoken.layers import LinearQKVMerged

    hidden, head_dim, num_qo, num_kv = 16, 8, 4, 2
    op = LinearQKVMerged(hidden, head_dim, num_qo, num_kv, has_bias=False)
    full_osize = (num_qo + 2 * num_kv) * head_dim
    assert op.local_output_size == full_osize
    assert op.local_input_size == hidden
    assert op.weight.shape == (full_osize, hidden)
    w, _ = _seed(op, 14)
    x = _in(5, hidden)
    err = _max_abs_err(op.forward(x), x @ w.t())
    assert err < 1e-5, f"LinearQKVMerged max-err {err}"


@pytest.mark.xpu
def test_linear_opp_proj_matches_reference():
    from freetoken.layers import LinearOProj

    isz, osz = 16, 10
    op = LinearOProj(isz, osz, has_bias=True)
    # TP=1 -> local input == full input.
    assert op.local_input_size == isz and op.local_output_size == osz
    assert op.weight.shape == (osz, isz)
    w, b = _seed(op, 15)
    x = _in(6, isz)
    err = _max_abs_err(op.forward(x), x @ w.t() + b)
    assert err < 1e-5, f"LinearOProj max-err {err}"


@pytest.mark.xpu
def test_linear_row_parallel_matches_reference():
    from freetoken.layers import LinearRowParallel

    isz, osz = 16, 10
    op = LinearRowParallel(isz, osz, has_bias=True)
    # TP=1 -> local input == full input, output unsharded.
    assert op.local_input_size == isz and op.local_output_size == osz
    assert op.weight.shape == (osz, isz)
    w, b = _seed(op, 16)
    x = _in(6, isz)
    err = _max_abs_err(op.forward(x), x @ w.t() + b)
    assert err < 1e-5, f"LinearRowParallel max-err {err}"


@pytest.mark.xpu
def test_linear_out_buffer_is_respected():
    from freetoken.layers import LinearReplicated

    isz, osz = 16, 8
    op = LinearReplicated(isz, osz, has_bias=False)
    w, _ = _seed(op, 17)
    x = _in(4, isz)
    out = torch.zeros(4, osz, device=DEVICE)
    ret = op.forward(x, out=out)
    assert ret is out, "out must be the returned (and written) buffer"
    err = _max_abs_err(out, x @ w.t())
    assert err < 1e-5, f"LinearReplicated out-buffer max-err {err}"


@pytest.mark.xpu
def test_linear_bf16_input_stays_bf16():
    from freetoken.layers import LinearReplicated

    isz, osz = 16, 8
    op = LinearReplicated(isz, osz, has_bias=False)
    # Seed in bf16 so the reference is the same bf16 matmul (ulp-scale residual).
    g = torch.Generator(device="cpu").manual_seed(18)
    w = torch.randn(op.weight.shape, device="cpu", generator=g) * 0.5
    op.weight = w.to(device=DEVICE, dtype=torch.bfloat16)
    x = torch.randn(4, isz, device=DEVICE, dtype=torch.bfloat16)
    got = op.forward(x)
    assert got.dtype == torch.bfloat16, "output dtype must match input dtype"
    err = _max_abs_err(got, x @ w.to(device=DEVICE, dtype=torch.bfloat16).t())
    assert err < 0.5, f"bf16 Linear max-err {err}"
