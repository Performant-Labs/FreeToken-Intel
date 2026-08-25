from .blocks import BaseLLMModel
from .config import (
    AttentionGroupConfig,
    BaseAttentionGroupConfig,
    DSV4AttentionGroupConfig,
    FullAttentionGroupConfig,
    KVCacheGroupSpec,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
    SWAAttentionGroupConfig,
)
from .register import get_model_class

# ``weight`` (the safetensors reader + MoE bank builder) and ``loader`` import
# torch. They are exported lazily (see ``__getattr__``) so that a CPU-only box
# without torch -- which still imports ``freetoken.models`` for the config /
# registry / base-model pieces -- does not fail at import time. The loader and
# weight entry points are only *called* from torch-bound paths.
_TORCH_BOUND_EXPORTS = {
    "load_weight": "freetoken.models.weight",
    "load_moe_expert_sources": "freetoken.models.weight",
    "load_model": "freetoken.models.loader",
}


def __getattr__(name: str):
    if name in _TORCH_BOUND_EXPORTS:
        import importlib

        module = importlib.import_module(_TORCH_BOUND_EXPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def create_model(model_config: ModelConfig) -> BaseLLMModel:
    return get_model_class(model_config.architectures[0], model_config)


__all__ = [
    "BaseLLMModel",
    "create_model",
    "load_model",
    "load_weight",
    "load_moe_expert_sources",
    "AttentionGroupConfig",
    "BaseAttentionGroupConfig",
    "DSV4AttentionGroupConfig",
    "FullAttentionGroupConfig",
    "KVCacheGroupSpec",
    "LinearGatedDeltaGroupConfig",
    "ModelConfig",
    "RotaryConfig",
    "SWAAttentionGroupConfig",
]
