from freetoken.attention import SUPPORTED_ATTENTION_BACKENDS
from freetoken.models.register import _MODEL_REGISTRY, get_model_spec
from freetoken.moe import OFFLOAD_MOE_BACKENDS, SUPPORTED_MOE_BACKENDS


def test_moe_backends():
    assert set(SUPPORTED_MOE_BACKENDS.supported_names()) == {
        "fused",
        "offload",
        "cpu",
        "hybrid",
    }
    assert OFFLOAD_MOE_BACKENDS == frozenset({"offload", "cpu", "hybrid"})


def test_attention_backends():
    assert set(SUPPORTED_ATTENTION_BACKENDS.supported_names()) == {"triton", "sycl"}


def test_qwen35_registered():
    spec = get_model_spec("Qwen3_5MoeForConditionalGeneration")
    assert spec.module == "freetoken.models.qwen3_5_moe"
    assert "Qwen3MoeForCausalLM" in _MODEL_REGISTRY
