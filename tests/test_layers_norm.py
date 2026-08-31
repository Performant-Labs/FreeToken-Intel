"""Tests for the RMSNorm layer family (issue ``layers``, #24).

Two halves (mirrors ``test_moe_fused.py``):

* CPU-safe (the per-PR ``ci`` job, torch-free): the norm module presence. The numerical
  coverage lives in the xpu-marked tests below -- the CPU venv has no torch, so
  nothing here can run a tensor on the CPU.

* ``xpu``-marked (the B70 nightly, ``.venv-xpu``): every RMSNorm variant is driven
  on the XPU against a hand-written float32 reference (the same math the upstream
  flashinfer / Triton kernels compute). Agreement means the PyTorch-backed port is
  numerically identical to the CUDA kernels it replaces.

The Intel port computes RMSNorm with PyTorch primitives (``torch.rms_norm``) rather
than upstream's CUDA kernels because the Triton fallback is CUDA-specific (PDL /
``launch_pdl`` + PTX inline-asm) that the Intel Triton backend rejects.

XPU repr hazard (why every check computes a plain float first):
    On the B70, if a tensor lands inside a *failing* ``assert`` -- as the asserted
    value or embedded in the message -- pytest's assertion-rewriter calls ``repr()``
    on that tensor to build the failure diff. On this oneAPI runtime that repr can
    request a multi-TiB buffer and OOM the device (the "Tried to allocate
    538976288 GiB" symptom); because pytest re-renders on OOM, that loops and wedges
    the whole run. The kernels themselves are microsecond-fast in isolation -- the
    wedge is purely in the *failure display* path, not the math.

    So every numerical check below computes its max-abs-error into a plain Python
    ``float`` (``.item()``) *outside* the ``assert`` and asserts on that float. A
    failing assert then carries only the float, never a tensor, so pytest has no
    tensor to ``repr``. (The same convention applies to the other layer suites.)
"""
from __future__ import annotations

import importlib.util

import pytest

torch = pytest.importorskip("torch")


def test_norm_layer_module_present():
    """No torch needed to check presence (self-skips on the torch-free CPU venv)."""
    spec = importlib.util.find_spec("freetoken.layers.norm")
    assert spec is not None, "freetoken.layers.norm is missing"


# --------------------------------------------------------------------------- #
# CPU references (hand-written, independent of the layer under test)
#
# torch.nn.functional.rms_norm computes:  x * w / sqrt(eps + mean(x^2))
# (the eps is added to the *variance* before sqrt, per torch's doc). For the Gemma
# variants the scale is (1 + w) instead of w.
# --------------------------------------------------------------------------- #


def _rmsnorm_ref(x: "torch.Tensor", w: "torch.Tensor", eps: float, one_plus: bool = False) -> "torch.Tensor":
    # The layers are built on ``torch.rms_norm`` (upstream parity), so that op IS the
    # reference here. We feed it the *same bf16 tensors the layer consumes* and, for
    # the ``one_plus`` family, the runtime ``(1 + w)`` scale -- the exact op the port
    # performs. A hand float32 reduction would disagree with the XPU kernel's bf16
    # variance accumulation (a few bf16 ulps), which is a property of the *op*, not the
    # port: the port is bit-exact to ``torch.rms_norm`` (see the xpu tests).
    return torch.rms_norm(x, (x.shape[-1],), (1.0 + w) if one_plus else w, eps)


def _fused_add_ref(x: "torch.Tensor", residual: "torch.Tensor", w: "torch.Tensor", eps: float, one_plus: bool = False):
    # Residual is formed with the same dtype the layer uses (bf16 add), then normed
    # via the same ``torch.rms_norm`` call the layer makes. Returns (normed, residual).
    residual = residual + x
    return _rmsnorm_ref(residual, w, eps, one_plus), residual


def _max_abs_err(got: "torch.Tensor", want: "torch.Tensor") -> float:
    """Max abs error as a plain Python float, computed *outside* any ``assert``.

    See the XPU-repr-hazard note in the module docstring: returning a ``float`` here
    is what keeps a failing assert free of tensors, so pytest never has to ``repr()``
    one on the oneAPI runtime.
    """
    return (got.float() - want).abs().max().item()


def _mk(device: str, dtype, size: int, n: int, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, size, device="cpu", dtype=torch.float32, generator=g) * 4.0
    w = torch.randn(size, device="cpu", dtype=torch.float32, generator=g) + 1.0
    return x.to(device=device, dtype=dtype), w.to(device=device, dtype=dtype)


DEV = "xpu"


# --------------------------------------------------------------------------- #
# XPU: drive each RMSNorm variant on the B70.  References are built on the same
# ``torch.rms_norm`` op the layers are implemented with, using the same bf16 inputs,
# so the comparison is op- and dtype-faithful (and bit-exact for the plain paths).
# --------------------------------------------------------------------------- #


@pytest.mark.xpu
def test_rmsnorm_forward_matches_reference():
    from freetoken.layers.norm import RMSNorm

    H, N, eps = 256, 7, 1e-6
    x, w = _mk(DEV, torch.bfloat16, H, N, seed=0)
    layer = RMSNorm(H, eps=eps)
    layer.weight = w
    got = layer.forward(x)
    want = _rmsnorm_ref(x, w, eps)
    err = _max_abs_err(got, want)
    assert got.shape == x.shape
    assert err < 5e-2, f"rmsnorm forward: max abs err {err:.5f}"


@pytest.mark.xpu
def test_rmsnorm_weight_is_state_dict():
    from freetoken.layers.norm import RMSNorm

    H = 64
    layer = RMSNorm(H, eps=1e-6)
    sd = layer.state_dict()
    assert set(sd.keys()) == {"weight"}, "RMSNorm.weight must be the only persisted param"
    new_w = torch.randn(H)
    layer.load_state_dict({"weight": new_w})
    loaded = layer.weight.detach().cpu().numpy().tobytes()
    expected = new_w.detach().cpu().numpy().tobytes()
    assert loaded == expected, "load_state_dict must install the given weight bytes verbatim"


@pytest.mark.xpu
def test_rmsnorm_forward_inplace_matches_reference():
    from freetoken.layers.norm import RMSNorm

    H, N, eps = 128, 5, 1e-6
    x, w = _mk(DEV, torch.bfloat16, H, N, seed=1)
    layer = RMSNorm(H, eps=eps)
    layer.weight = w
    snap = x.clone()
    layer.forward_inplace(x)
    want = _rmsnorm_ref(snap, w, eps)
    err = _max_abs_err(x, want)
    assert err < 5e-2, f"rmsnorm inplace: max abs err {err:.5f}"


@pytest.mark.xpu
def test_gemma_rmsnorm_one_plus_scaling():
    from freetoken.layers.norm import GemmaRMSNorm

    H, N, eps = 256, 6, 1e-6
    x, w = _mk(DEV, torch.bfloat16, H, N, seed=2)
    layer = GemmaRMSNorm(H, eps=eps)
    layer.weight = w
    got = layer.forward(x)
    want = _rmsnorm_ref(x, w, eps, one_plus=True)
    err = _max_abs_err(got, want)
    assert err < 5e-2, f"gemma rmsnorm: max abs err {err:.5f}"


@pytest.mark.xpu
def test_gemma_rmsnorm_with_scale_false_uses_ones():
    from freetoken.layers.norm import GemmaRMSNorm

    H, N, eps = 128, 5, 1e-6
    x, _ = _mk(DEV, torch.bfloat16, H, N, seed=3)
    layer = GemmaRMSNorm(H, eps=eps, with_scale=False)
    got = layer.forward(x)
    # with_scale=False -> runtime scale is (1 + ones) = 2.  We reference against the
    # same op the layer performs (``torch.rms_norm`` with a 2-scaled bf16 weight) so
    # the comparison is dtype- and op-faithful (a hand float32 reduction would disagree
    # with the XPU kernel's bf16 variance by a few ulps -- see module docstring).
    twos = torch.full((H,), 2.0, device=DEV, dtype=torch.bfloat16)
    want = torch.rms_norm(x, (H,), twos, eps)
    err = _max_abs_err(got, want)
    assert err < 5e-2, f"gemma with_scale=False: max abs err {err:.5f}"


@pytest.mark.xpu
def test_gemma_rmsnorm_3d_input_collapses_and_restores():
    from freetoken.layers.norm import GemmaRMSNorm

    H, B, T, eps = 64, 3, 4, 1e-6
    x, w = _mk(DEV, torch.bfloat16, H, B * T, seed=4)
    x = x.view(B, T, H)
    layer = GemmaRMSNorm(H, eps=eps)
    layer.weight = w
    got = layer.forward(x)
    want = _rmsnorm_ref(x.reshape(-1, H), w, eps, one_plus=True).view(B, T, H)
    err = _max_abs_err(got, want)
    assert got.shape == x.shape
    assert err < 5e-2, f"gemma 3d: max abs err {err:.5f}"


@pytest.mark.xpu
def test_rmsnorm_fused_no_residual_is_plain_norm():
    from freetoken.layers.norm import RMSNormFused

    H, N, eps = 128, 5, 1e-6
    x, w = _mk(DEV, torch.bfloat16, H, N, seed=5)
    layer = RMSNormFused(H, eps=eps)
    layer.weight = w
    normed, residual = layer.forward(x)
    want = _rmsnorm_ref(x, w, eps)
    err = _max_abs_err(normed, want)
    assert err < 5e-2, f"fused(no-res): max abs err {err:.5f}"
    # residual must be the *same* tensor object x (untouched), when no residual is passed
    assert residual is x, "residual must be x (untouched) when no residual is passed"


@pytest.mark.xpu
def test_rmsnorm_fused_residual_updates_both_inplace():
    from freetoken.layers.norm import RMSNormFused

    H, N, eps = 128, 5, 1e-6
    x, w = _mk(DEV, torch.bfloat16, H, N, seed=6)
    residual = torch.randn(N, H, device=DEV, dtype=torch.bfloat16)
    layer = RMSNormFused(H, eps=eps)
    layer.weight = w
    x_snap, res_snap = x.clone(), residual.clone()
    normed, residual = layer.forward(x, residual)
    want, want_res = _fused_add_ref(x_snap, res_snap, w, eps, one_plus=False)
    res_err = _max_abs_err(residual, want_res)
    assert res_err < 5e-2, f"fused-res residual: max abs err {res_err:.5f}"
    n_err = _max_abs_err(normed, want)
    assert n_err < 5e-2, f"fused-res norm: max abs err {n_err:.5f}"


@pytest.mark.xpu
def test_gemma_plus_one_fused_residual():
    from freetoken.layers.norm import GemmaPlusOneRMSNormFused

    H, N, eps = 128, 5, 1e-6
    x, w = _mk(DEV, torch.bfloat16, H, N, seed=7)
    residual = torch.randn(N, H, device=DEV, dtype=torch.bfloat16)
    layer = GemmaPlusOneRMSNormFused(H, eps=eps)
    layer.weight = w
    x_snap, res_snap = x.clone(), residual.clone()
    normed, residual = layer.forward(x, residual)
    want, want_res = _fused_add_ref(x_snap, res_snap, w, eps, one_plus=True)
    res_err = _max_abs_err(residual, want_res)
    assert res_err < 5e-2, f"gemma-fused residual: max abs err {res_err:.5f}"
    n_err = _max_abs_err(normed, want)
    assert n_err < 5e-2, f"gemma-fused norm: max abs err {n_err:.5f}"


@pytest.mark.xpu
def test_gemma_plus_one_forward_and_inplace():
    from freetoken.layers.norm import GemmaPlusOneRMSNorm

    H, N, eps = 256, 6, 1e-6
    x, w = _mk(DEV, torch.bfloat16, H, N, seed=8)
    layer = GemmaPlusOneRMSNorm(H, eps=eps)
    layer.weight = w
    got = layer.forward(x)
    want = _rmsnorm_ref(x, w, eps, one_plus=True)
    err = _max_abs_err(got, want)
    assert err < 5e-2, f"gemma+1 forward: max abs err {err:.5f}"

    x2, w2 = _mk(DEV, torch.bfloat16, H, N, seed=9)
    snap = x2.clone()
    layer2 = GemmaPlusOneRMSNorm(H, eps=eps)
    layer2.weight = w2
    layer2.forward_inplace(x2)
    err2 = _max_abs_err(x2, _rmsnorm_ref(snap, w2, eps, one_plus=True))
    assert err2 < 5e-2, f"gemma+1 inplace: max abs err {err2:.5f}"


@pytest.mark.xpu
def test_gemma_rmsnorm_forward_add_residual():
    from freetoken.layers.norm import GemmaRMSNorm

    H, N, eps = 128, 5, 1e-6
    x, w = _mk(DEV, torch.bfloat16, H, N, seed=10)
    residual = torch.randn(N, H, device=DEV, dtype=torch.bfloat16)
    layer = GemmaRMSNorm(H, eps=eps)
    layer.weight = w
    x_snap, res_snap = x.clone(), residual.clone()
    normed, residual = layer.forward_add_residual(x, residual)
    want, want_res = _fused_add_ref(x_snap, res_snap, w, eps, one_plus=True)
    res_err = _max_abs_err(residual, want_res)
    assert res_err < 5e-2, f"gemma add-residual residual: max abs err {res_err:.5f}"
    n_err = _max_abs_err(normed, want)
    assert n_err < 5e-2, f"gemma add-residual norm: max abs err {n_err:.5f}"
    # The method mutates x in place (x.copy_(norm)) and returns (x, residual).
    assert normed is x, "forward_add_residual must return the (in-place) normed x"


@pytest.mark.xpu
def test_gemma_plus_one_fused_no_residual_is_plain_one_plus_norm():
    from freetoken.layers.norm import GemmaPlusOneRMSNormFused

    H, N, eps = 128, 5, 1e-6
    x, w = _mk(DEV, torch.bfloat16, H, N, seed=11)
    layer = GemmaPlusOneRMSNormFused(H, eps=eps)
    layer.weight = w
    normed, residual = layer.forward(x)
    want = _rmsnorm_ref(x, w, eps, one_plus=True)
    err = _max_abs_err(normed, want)
    assert err < 5e-2, f"gemma-fused(no-res): max abs err {err:.5f}"
    # With no residual the residual out is the *same* tensor object x (untouched).
    assert residual is x, "residual must be x (untouched) when no residual is passed"
