"""qwen3_5_moe's dense (capture-mode) MoE routing matches gather (eager) routing.

Same property as tests/test_moe_fused_dense_capture_routing.py (issue #123),
for the qwen3_5_moe model family's `_Qwen35MoE._forward_inram`, which was not
covered by that test (PR-Agent review on #124: the qwen3_moe test's
`mlp.forward(flat, model=...)` call shape does not exist on qwen3_5_moe's
`_forward_inram(flat, top_idx, top_w, ctx)`, so a regression there would not
have been caught).

Built directly (not through freetoken.models.loader.load_model): that loader
path is currently broken for every qwen3_5 checkpoint by an unrelated
pre-existing bug (`parse_config() got an unexpected keyword argument
'model_path'`, tracked separately, not touched here) -- so this constructs
the plain `_Qwen35Expert` / `_Qwen35MoE` classes directly (via the module's
own lazy `_ensure_torch()` rebind, the same mechanism `load_model` uses) and
calls `_forward_inram` as an unbound method against a minimal stand-in
object exposing just the attributes it reads (`num_experts`, `top_k`,
`experts`). CPU-only: the property under test is device-independent.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import freetoken.models.qwen3_5_moe as q35


class _FakeMoE:
    """Exposes exactly what _Qwen35MoE._forward_inram reads off self."""

    def __init__(self, num_experts: int, top_k: int, experts) -> None:
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = experts


class _FakeCtx:
    def __init__(self, model) -> None:
        self.model = model


class _FakeModel:
    def __init__(self, capturing: bool) -> None:
        self._capturing = capturing


def test_qwen35_dense_routing_matches_gather_routing():
    q35._ensure_torch()

    hidden, inter, num_experts, top_k = 16, 8, 4, 2

    class _Cfg:
        hidden_size = hidden
        moe_intermediate_size = inter

    torch.manual_seed(0)
    experts = [q35._Qwen35Expert(_Cfg(), torch.device("cpu"), torch.float32) for _ in range(num_experts)]
    for e in experts:
        for p in (e.gate_proj, e.up_proj, e.down_proj):
            for param in p.parameters():
                torch.nn.init.normal_(param, std=0.1)

    fake_moe = _FakeMoE(num_experts=num_experts, top_k=top_k, experts=experts)

    flat = torch.randn(5, hidden)  # 5 tokens
    gate = torch.nn.Linear(hidden, num_experts, bias=False)
    routing = gate(flat)
    gate_log = torch.softmax(routing, dim=-1)
    top_w, top_idx = torch.topk(gate_log, top_k, dim=-1)
    top_w = (top_w / top_w.sum(dim=-1, keepdim=True)).to(flat.dtype)

    out_gather = q35._Qwen35MoE._forward_inram(fake_moe, flat, top_idx, top_w, _FakeCtx(_FakeModel(False)))
    out_dense = q35._Qwen35MoE._forward_inram(fake_moe, flat, top_idx, top_w, _FakeCtx(_FakeModel(True)))

    # Not bit-exact -- same reason as the qwen3_moe version of this test:
    # dense calls each expert on the full 5-token batch, gather calls it on
    # a per-expert subset, and different batch shapes can dispatch to
    # different (non-bit-associative) BLAS matmul blocking.
    torch.testing.assert_close(out_gather, out_dense, rtol=1e-3, atol=1e-3)
