"""CPU-runnable wiring test: load_model() places a GPTQ-Int4-quantized
checkpoint's routed experts through the fully-resident (``moe_backend=
"fused"``) path correctly, without needing a real Triton-XPU compile
(issue `moe-fused-gptq`, #258).

Companion to ``test_qwen35_gptq_fused_e2e_xpu.py`` (the XPU-marked forward
test -- :func:`freetoken.kernel.triton.gptq_fused_linear.
fused_gptq_expert_forward` needs a real Xe2 tensor-core dispatch, no
meaningful CPU-only version exists), the same split
``test_gptq_fused_dispatch.py`` / ``test_gptq_fused_linear_xpu.py`` already
use for the underlying kernel. This module proves two things that ARE
CPU-testable:

1. ``_place_expert_weights_any`` dispatches a ``GptqExpertBank`` to
   ``_Qwen35GptqExpert`` instead of crashing with the opaque
   ``'GptqExpertBank' object is not subscriptable`` ``TypeError`` #258
   reported (or falling through to some other silently-wrong path).
2. ``_Qwen35GptqExpert.forward`` calls
   :func:`freetoken.kernel.triton.gptq_fused_linear.fused_gptq_expert_forward`
   with the right packed tensors, ``group_size``, and ``intermediate`` --
   verified by monkeypatching that function to a recording stub (its actual
   body is Triton-XPU-only).
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from freetoken.models.loader import load_model
from freetoken.models.qwen3_5_moe import _Qwen35GptqExpert

from tests.test_qwen35_gptq_e2e_loader import E, GROUP, H, I, V, _qwen35_gptq_weights


@pytest.fixture(scope="module")
def qwen35_gptq_ckpt(tmp_path_factory):
    from safetensors.torch import save_file

    path = tmp_path_factory.mktemp("qwen35_gptq_fused_dispatch")
    text_config = {
        "hidden_size": H,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_experts": E,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": I,
        "shared_expert_intermediate_size": I,
        "vocab_size": V,
        "max_position_embeddings": 128,
        "head_dim": 16,
        "attn_output_gate": False,
        "partial_rotary_factor": 0.5,
        "full_attention_interval": 1,
        "layer_types": ["full_attention"],
        "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.5},
    }
    config = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "tie_word_embeddings": True,
        "text_config": text_config,
        "quantization_config": {
            "bits": 4,
            "group_size": GROUP,
            "sym": True,
            "desc_act": False,
            "quant_method": "gptq",
            "dynamic": {
                "-:.*attn.*": {},
                "-:.*shared_expert.*": {},
            },
        },
    }
    weights = _qwen35_gptq_weights()
    (path / "config.json").write_text(json.dumps(config))
    save_file({k: v.contiguous() for k, v in weights.items()}, str(path / "model.safetensors"))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in weights}})
    )
    return str(path)


def test_load_model_places_gptq_experts_fully_resident_on_cpu(qwen35_gptq_ckpt):
    """The real point of #258: moe_backend="fused" no longer crashes on a
    GPTQ-packed bank (the opaque 'GptqExpertBank' object is not subscriptable
    TypeError the issue reports) -- each expert becomes a real
    _Qwen35GptqExpert holding packed qweight/qzeros/scales, not a bf16
    nn.Linear."""
    model, _ = load_model(qwen35_gptq_ckpt, torch.device("cpu"), dtype=torch.float32, moe_backend="fused")

    experts = model.layers[0].mlp.experts
    assert len(experts) == E
    for expert in experts:
        assert isinstance(expert, _Qwen35GptqExpert)
        assert expert.qweight_gate_up.device.type == "cpu"
        assert expert.qweight_gate_up.dtype == torch.int32
        assert expert.qzeros_gate_up.dtype == torch.int32
        assert expert.group_size == GROUP
        assert expert.intermediate == I
        # gate_up: K = H, packed rows = H // 8; down: K = I, packed rows = I // 8.
        assert expert.qweight_gate_up.shape == (H // 8, 2 * I)
        assert expert.qweight_down.shape == (I // 8, H)


def test_gptq_expert_forward_calls_the_native_fused_kernel_with_the_right_args(qwen35_gptq_ckpt, monkeypatch):
    """Verify the wiring, not the (Triton-XPU-only) kernel math: patch
    fused_gptq_expert_forward to a recording stub and confirm
    _Qwen35GptqExpert.forward hands it this expert's own packed tensors,
    group_size, and intermediate -- exactly the call
    test_gptq_fused_linear_xpu.py proves is bit-for-bit correct on real
    hardware."""
    import freetoken.kernel.triton.gptq_fused_linear as gptq_fused_linear

    model, _ = load_model(qwen35_gptq_ckpt, torch.device("cpu"), dtype=torch.float32, moe_backend="fused")
    expert = model.layers[0].mlp.experts[0]

    calls = []

    def _fake_forward(x, qw_gu, qz_gu, s_gu, qw_dn, qz_dn, s_dn, *, group_size, intermediate, out_dtype=None):
        calls.append(
            {
                "x_shape": tuple(x.shape),
                "qweight_gate_up": qw_gu,
                "qweight_down": qw_dn,
                "group_size": group_size,
                "intermediate": intermediate,
            }
        )
        return torch.zeros(x.shape[0], H, dtype=out_dtype or x.dtype)

    monkeypatch.setattr(gptq_fused_linear, "fused_gptq_expert_forward", _fake_forward)

    x = torch.randn(3, H)
    out = expert.forward(x)

    assert out.shape == (3, H)
    assert len(calls) == 1
    call = calls[0]
    assert call["group_size"] == GROUP
    assert call["intermediate"] == I
    assert call["qweight_gate_up"] is expert.qweight_gate_up
    assert call["qweight_down"] is expert.qweight_down
