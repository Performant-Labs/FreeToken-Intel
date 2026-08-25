"""Qwen3-MoE model (first-class Intel Arc Pro B70 port).

Upstream NVIDIA path: python/freetoken/models/qwen3_moe/
Fill in: GitHub issue `models-qwen3-moe` (see docs/architecture.md).

Only the *checkpoint-side* half is implemented here for the loader (`#17`):
``parse_config`` (HF config -> :class:`ModelConfig`) and ``iter_weights`` (route
each checkpoint tensor to its destination device -- dense to the XPU, MoE
experts to host offload banks). The forward pass (the actual MoE model that the
engine runs) is a stub and lands with the engine issue (`#14`).
"""
from __future__ import annotations

import torch

from freetoken.models.blocks import BaseLLMModel
from freetoken.models.config import ModelConfig
from freetoken.models.weight import iter_safetensors


def parse_config(hf_config) -> ModelConfig:
    """Build a :class:`ModelConfig` from a HF Qwen3-MoE config.

    ``hf_config`` is the lru-cached object shared across callers, so it is
    copied (``to_dict``) before the parsed fields are derived -- never mutated.
    """
    # The cached HF config is shared across callers, so derive from a plain-dict
    # copy (read by key, never by attribute on the cached object).
    src = hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(hf_config)
    # transformers' Qwen3MoeConfig stores the expert count under
    # ``num_local_experts`` (its ``num_experts`` attribute is the public alias and
    # is dropped by to_dict); accept either spelling.
    cfg = ModelConfig(
        architectures=["Qwen3MoeForCausalLM"],
        hidden_size=src.get("hidden_size"),
        num_layers=src.get("num_hidden_layers"),
        num_experts=src.get("num_local_experts") or src.get("num_experts"),
        intermediate_size=src.get("intermediate_size"),
        moe_intermediate_size=src.get("moe_intermediate_size"),
        num_experts_per_tok=src.get("num_experts_per_tok"),
    )
    # FreeToken's MoE plumbing keys off config.is_moe; expose it.
    cfg.is_moe = True
    return cfg


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool = True,
    include_non_moe: bool = True,
) -> "iter[tuple[str, torch.Tensor]]":
    """Yield the checkpoint's tensors, each on its destination device.

    MoE expert tensors (``...mlp.experts...``) stay on **host** memory -- the XPU
    holds dense weights and serves experts from host banks on demand. Every other
    (dense) tensor is yielded on ``device`` (the XPU).
    """
    for name, tensor in iter_safetensors(model_path, device):
        is_expert = ".experts." in name
        if is_expert and not include_moe_experts:
            continue
        if not is_expert and not include_non_moe:
            continue
        # Dense -> destination device; experts -> host offload banks.
        dest = torch.device("cpu") if is_expert else device
        yield name, tensor.to(dest)


class Qwen3MoeForCausalLM(BaseLLMModel):
    """The Qwen3-MoE model.

    The checkpoint side (config parse + weight routing) is real; the forward
    pass is a stub that lands with ``engine-loop`` (#14). It inherits the
    base model's (empty) parameter set so the loader's ``named_parameters``
    contract holds.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config

    def forward(self, *args, **kwargs):
        from freetoken._stub import unimplemented

        unimplemented("Qwen3MoeForCausalLM.forward", "models-qwen3-moe")


__all__ = ["parse_config", "iter_weights", "Qwen3MoeForCausalLM"]
