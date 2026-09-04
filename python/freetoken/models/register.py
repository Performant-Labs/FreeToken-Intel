from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    module: str
    model_cls: str
    parse_config: str = "parse_config"
    iter_weights: str = "iter_weights"


_MODEL_REGISTRY: dict[str, ModelSpec] = {
    "LlamaForCausalLM": ModelSpec("freetoken.models.llama", "LlamaForCausalLM"),
    "Qwen2ForCausalLM": ModelSpec("freetoken.models.qwen2", "Qwen2ForCausalLM"),
    "Qwen3ForCausalLM": ModelSpec("freetoken.models.qwen3", "Qwen3ForCausalLM"),
    "Qwen3MoeForCausalLM": ModelSpec("freetoken.models.qwen3_moe", "Qwen3MoeForCausalLM"),
    "Qwen3_5MoeForConditionalGeneration": ModelSpec(
        "freetoken.models.qwen3_5_moe", "Qwen3_5MoEForCausalLM"
    ),
    "Qwen3_5ForConditionalGeneration": ModelSpec(
        "freetoken.models.qwen3_5_moe", "Qwen3_5MoEForCausalLM"
    ),
    "DeepseekV4ForCausalLM": ModelSpec("freetoken.models.deepseek_v4", "DeepseekV4ForCausalLM"),
    "GptOssForCausalLM": ModelSpec("freetoken.models.gpt_oss", "GptOssForCausalLM"),
    "Glm4MoeForCausalLM": ModelSpec("freetoken.models.glm4_moe", "Glm4MoeForCausalLM"),
    "GlmMoeDsaForCausalLM": ModelSpec("freetoken.models.glm_moe_dsa", "GlmMoeDsaForCausalLM"),
    "Qwen4ExpForConditionalGeneration": ModelSpec("freetoken.models.qwen4_exp", "Qwen4ExpForCausalLM"),  # gitleaks:allow -- architecture name, not a secret
    "Gemma4ForCausalLM": ModelSpec("freetoken.models.gemma4", "Gemma4ForCausalLM"),
    "Gemma4ForConditionalGeneration": ModelSpec("freetoken.models.gemma4", "Gemma4ForCausalLM"),
    "MistralForCausalLM": ModelSpec("freetoken.models.mistral", "MistralForCausalLM"),
    "MiniMaxM2ForCausalLM": ModelSpec("freetoken.models.minimax_m2", "MiniMaxM2ForCausalLM"),
    "MiniMaxM3SparseForCausalLM": ModelSpec("freetoken.models.minimax_m3", "MiniMaxM3ForCausalLM"),
    "MuseGlimmerForConditionalGeneration": ModelSpec(
        "freetoken.models.muse_glimmer", "MuseGlimmerForCausalLM"
    ),
}


def get_model_spec(model_architecture: str) -> ModelSpec:
    try:
        return _MODEL_REGISTRY[model_architecture]
    except KeyError as exc:
        raise ValueError(f"Model architecture {model_architecture} not supported") from exc


def _load_attr(module_path: str, attr_name: str) -> Any:
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def get_model_class(model_architecture: str, model_config, **kwargs):
    """Resolve the model class for ``model_architecture`` and instantiate it.

    Extra ``kwargs`` (e.g. ``device``) are forwarded to the class constructor;
    models that don't take them are built with defaults.
    """
    spec = get_model_spec(model_architecture)
    model_cls = _load_attr(spec.module, spec.model_cls)
    return model_cls(model_config, **kwargs)


__all__ = ["ModelSpec", "get_model_spec", "get_model_class"]
