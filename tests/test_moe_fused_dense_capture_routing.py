"""Dense (capture-mode) MoE routing matches the gather (eager) routing exactly.

Issue moe-fused-graph-capture (#123): the fused in-VRAM MoE forward's eager
routing does a CPU round-trip (`top_idx.to("cpu")`, needed to work around a
broken on-device `nonzero()`, see #114) that is itself a device->host sync --
incompatible with graph capture. While `model._capturing` is set, `_Qwen3MoE
.forward` instead computes every expert for every token and weight-sums with
a mask built from plain elementwise ops (`==` / `torch.where`, no
`nonzero()`, no sync). This test proves that dense path is mathematically
identical to the (already correctness-tested, see test_moe_fused_xpu.py)
gather path it replaces while capturing -- CPU-only, no XPU needed, since the
property under test (dense routing == gather routing) is device-independent.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from freetoken.models.loader import load_model

from tests.test_moe_offload_forward import offload_ckpt  # noqa: F401


class _CapturingModel:
    """Minimal stand-in for the model= arg _Qwen3MoE.forward reads _capturing off."""

    def __init__(self, capturing: bool) -> None:
        self._capturing = capturing


def test_dense_routing_matches_gather_routing(offload_ckpt):
    model, _ = load_model(offload_ckpt, torch.device("cpu"), dtype=torch.float32, moe_backend="fused")
    mlp = model.layers[0].mlp
    torch.manual_seed(0)
    flat = torch.randn(5, model.config.hidden_size)  # 5 tokens, bs > 1 so routing varies per row

    out_gather = mlp.forward(flat, model=_CapturingModel(capturing=False))
    out_dense = mlp.forward(flat, model=_CapturingModel(capturing=True))

    # Not bit-exact: the dense path calls each expert on the FULL 5-token
    # batch while the gather path calls it on a per-expert SUBSET (only the
    # routed rows) -- different batch shapes can dispatch to different BLAS
    # matmul blocking/vectorization, which is not bit-associative even for
    # mathematically identical per-row sums (the same non-determinism the
    # rest of this codebase already tolerates with allclose / greedy-token
    # comparisons rather than exact tensor equality across different call
    # shapes). The two routings select and weight the *same* experts per
    # token, so they must agree closely, not bit-for-bit.
    torch.testing.assert_close(out_gather, out_dense, rtol=1e-3, atol=1e-3)
