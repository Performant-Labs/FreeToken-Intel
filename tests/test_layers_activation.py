"""Tests for the pure-torch ``*_and_mul`` fused activations (issue #24, WP4).

Upstream ``freetoken.layers.activation`` dispatches each variant to a flashinfer or
in-repo Triton kernel:

    y = act(x[..., :d]) * x[..., d:],   d = x.shape[-1] // 2

Triton has no XPU backend and flashinfer is CUDA-only, so this port computes the
same math with torch ops. This suite drives each variant on the B70 against an
independent (hand-written) reference:

* silu_and_mul     -> silu(gate) * up,  silu(x) = x * sigmoid(x)
* gelu_and_mul     -> gelu(gate) * up,  gelu(x) = 0.5*x*(1+erf(x/sqrt2))
* gelu_tanh_and_mul-> gelu_tanh(gate) * up (the tanh approximation)
* swigluoai_and_mul-> clamped swigluoai (gate/up clamp, alpha-gated sigmoid, +1 bias)

The gate/up values are drawn from a WIDE range (x4, plus an explicit large-negative
value) so any wrong pairing, wrong clamp bound, or missing +1 bias in swigluoai is
loudly caught rather than hidden by small inputs.

XPU repr hazard: every numerical check reduces to a plain Python float (`.item()`)
*outside* the ``assert`` -- a tensor inside a failing assert hands pytest's
rewriter a tensor to ``repr()`` and on this oneAPI runtime that can OOM/loop the
run (see test_layers_norm.py for the full incident).

The torch-free CPU venv only runs the module-presence check below; the numerical
tests are xpu-marked.
"""
from __future__ import annotations

import importlib.util

import pytest

torch = pytest.importorskip("torch")

DEVICE = "xpu"


def test_activation_layer_module_present():
    """No torch needed to check presence (self-skips on the torch-free CPU venv)."""
    spec = importlib.util.find_spec("freetoken.layers.activation")
    assert spec is not None, "freetoken.layers.activation is missing"


def test_package_exports_activation():
    import freetoken.layers as L

    for name in ("silu_and_mul", "gelu_and_mul", "gelu_tanh_and_mul", "swigluoai_and_mul"):
        assert hasattr(L, name), f"freetoken.layers must export {name}"


# --------------------------------------------------------------------------- #
# Independent references (hand-written; not the layer under test).
# --------------------------------------------------------------------------- #
def _split(x: "torch.Tensor"):
    d = x.shape[-1] // 2
    return x[..., :d], x[..., d : 2 * d]


def _silu_ref(x):
    gate, up = _split(x)
    return (gate * torch.sigmoid(gate)) * up


def _gelu_ref(x):
    gate, up = _split(x)
    act = 0.5 * gate * (1.0 + torch.erf(gate / (2.0 ** 0.5)))
    return act * up


def _gelu_tanh_ref(x):
    gate, up = _split(x)
    inner = 0.7978845608028654 * (gate + 0.044715 * gate**3)
    return 0.5 * gate * (1.0 + torch.tanh(inner)) * up


def _swigluoai_ref(x, alpha=1.702, limit=7.0):
    gate, up = _split(x)
    gate = torch.minimum(gate, torch.full_like(gate, limit))
    up = torch.maximum(torch.minimum(up, torch.full_like(up, limit)), torch.full_like(up, -limit))
    return gate * torch.sigmoid(alpha * gate) * (up + 1.0)


def _max_abs_err(got, want) -> float:
    return (got.float() - want.float()).abs().max().item()


def _wide(n: int, d: int, seed: int):
    # Wide values: a big-norm randn plus explicit large +/- values so clamps and
    # the exp/sigmoid saturations are actually exercised.
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, 2 * d, device="cpu", dtype=torch.float32, generator=g) * 4.0
    x[0, 0] = -30.0  # strong negative -> exp(-x) overflows for silu/sigmoid
    x[0, 1] = 30.0   # strong positive
    return x.to(device=DEVICE)


# --------------------------------------------------------------------------- #
# XPU: drive each variant on the B70 against the hand-written references.
# --------------------------------------------------------------------------- #
@pytest.mark.xpu
def test_silu_and_mul_matches_reference():
    from freetoken.layers.activation import silu_and_mul

    x = _wide(8, 32, 1)
    err = _max_abs_err(silu_and_mul(x), _silu_ref(x))
    assert err < 1e-5, f"silu_and_mul max-err {err}"


@pytest.mark.xpu
def test_gelu_and_mul_matches_reference():
    from freetoken.layers.activation import gelu_and_mul

    x = _wide(8, 32, 2)
    err = _max_abs_err(gelu_and_mul(x), _gelu_ref(x))
    assert err < 1e-5, f"gelu_and_mul max-err {err}"


@pytest.mark.xpu
def test_gelu_tanh_and_mul_matches_reference():
    from freetoken.layers.activation import gelu_tanh_and_mul

    x = _wide(8, 32, 3)
    err = _max_abs_err(gelu_tanh_and_mul(x), _gelu_tanh_ref(x))
    # torch.tanh vs the upstream tanh.approx PTX op differ by ~1e-6, so the
    # erf-based bound is relaxed here (the tanh-approx formula itself is exact).
    assert err < 1e-4, f"gelu_tanh_and_mul max-err {err}"


@pytest.mark.xpu
def test_swigluoai_and_mul_matches_reference():
    from freetoken.layers.activation import swigluoai_and_mul

    x = _wide(8, 32, 4)
    got = swigluoai_and_mul(x)
    err = _max_abs_err(got, _swigluoai_ref(x))
    assert err < 1e-5, f"swigluoai_and_mul max-err {err}"


@pytest.mark.xpu
def test_swigluoai_clamp_bounds_take_effect():
    from freetoken.layers.activation import swigluoai_and_mul

    # gate above the limit must be clamped to the limit (so a missing clamp is caught).
    d = 8
    x = torch.zeros(1, 2 * d, device=DEVICE)
    x[0, :d] = 100.0  # gate = +100 -> clamped to limit
    x[0, d:] = -100.0  # up = -100 -> clamped to -limit
    got = swigluoai_and_mul(x, limit=7.0)
    want = _swigluoai_ref(x, limit=7.0)
    err = _max_abs_err(got, want)
    # With gate=limit, up=-limit: y = limit * sigmoid(alpha*limit) * (-limit + 1)
    # which is a specific finite value; a missing clamp would give a huge/NaN result.
    assert torch.isfinite(got).all().item(), "clamped swigluoai must be finite"
    assert err < 1e-5, f"swigluoai clamp max-err {err}"


@pytest.mark.xpu
def test_out_buffer_is_respected():
    from freetoken.layers.activation import silu_and_mul

    x = _wide(4, 16, 5)
    d = 16
    out = torch.zeros(x.shape[:-1] + (d,), device=DEVICE, dtype=x.dtype)
    ret = silu_and_mul(x, out=out)
    assert ret is out, "out must be the returned (and written) buffer"
    want = _silu_ref(x)
    err = _max_abs_err(out, want)
    assert err < 1e-5, f"out-buffer silu max-err {err}"


@pytest.mark.xpu
def test_out_buffer_shape_mismatch_raises():
    from freetoken.layers.activation import silu_and_mul

    x = _wide(4, 16, 6)
    bad = torch.zeros(4, 16, 32, device=DEVICE, dtype=x.dtype)  # wrong last dim
    with pytest.raises(ValueError):
        silu_and_mul(x, out=bad)


@pytest.mark.xpu
def test_output_shape_is_half_width():
    from freetoken.layers.activation import gelu_and_mul

    x = _wide(5, 24, 7)  # last dim 48 -> output last dim 24
    assert gelu_and_mul(x).shape == (5, 24), "output must be x.shape[:-1] + (d,)"


@pytest.mark.xpu
def test_bf16_input_stays_bf16():
    from freetoken.layers.activation import silu_and_mul

    x = _wide(4, 16, 8).to(torch.bfloat16)
    got = silu_and_mul(x)
    assert got.dtype == torch.bfloat16, "output dtype must match input dtype"
    # The reference is the *same* torch ops on the same bf16 tensor, so any residual
    # error is just bf16 rounding of the intermediate products -- at magnitude ~32 that
    # is one ulp (2^-3 = 0.125), so allow a bound safely above that step.
    err = _max_abs_err(got, _silu_ref(x))
    assert err < 0.2, f"bf16 silu max-err {err}"
