"""Shared model config dataclasses.

Upstream NVIDIA path: python/freetoken/models/config.py
Fill in: GitHub issue `models-qwen35` (see docs/architecture.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from freetoken._stub import unimplemented


@dataclass
class ModelConfig:
    """Parsed model configuration, derived from a checkpoint's HF config.

    Upstream stores ~40 fields; the port defines the core set here and each
    ``models-*`` issue adds the architecture-specific fields it needs. ``is_moe``
    is the flag the MoE plumbing (``weight.stream_moe_expert_sources``) keys off.
    """

    architectures: list[str] = field(default_factory=list)
    hidden_size: int | None = None
    vocab_size: int | None = None
    num_layers: int | None = None
    num_moe_layers: int | None = None
    num_experts: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    # The per-head dimension. ``None`` means "derive it" (hidden // num_heads) --
    # correct for most models, but Qwen3.5/3.6 (qwen3_5_moe) sets this explicitly
    # (head_dim=256 while hidden//heads would give 128). The KV pool reads this.
    head_dim: int | None = None
    intermediate_size: int | None = None
    moe_intermediate_size: int | None = None
    num_experts_per_tok: int | None = None
    hidden_act: str = "silu"
    first_k_dense_replace: int = 0
    is_moe: bool = False
    # ADR 0002: when True the MoE experts are never XPU-resident; the model
    # builds only the router and reads the routed experts from the host-offload
    # LRU slot pool (OffloadMoeCache) inside the forward pass. The loader sets
    # this from the engine's moe_backend choice (it is NOT a checkpoint field).
    use_offload_moe: bool = False
    dtype: Any = None
    max_position_embeddings: int | None = None
    tie_word_embeddings: bool = False
    kv_lora_rank: int | None = None
    q_lora_rank: int | None = None
    rope_theta: float | None = None
    rope_scaling: Any = None
    attrs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # The MoE plumbing and the engine size their per-layer structures off
        # ``num_moe_layers``; derive it from the total layer count minus the
        # leading dense layers when the adapter did not set it explicitly.
        if self.num_moe_layers is None and self.is_moe and self.num_layers:
            object.__setattr__(self, "num_moe_layers", max(0, self.num_layers - self.first_k_dense_replace))
class RotaryConfig:
    def __init__(self, *args, **kwargs) -> None:
        pass
class AttentionGroupConfig:
    def __init__(self, *args, **kwargs) -> None:
        pass
class BaseAttentionGroupConfig:
    def __init__(self, *args, **kwargs) -> None:
        pass
class FullAttentionGroupConfig:
    def __init__(self, *args, **kwargs) -> None:
        pass
class SWAAttentionGroupConfig:
    def __init__(self, *args, **kwargs) -> None:
        pass
class DSV4AttentionGroupConfig:
    def __init__(self, *args, **kwargs) -> None:
        pass
class LinearGatedDeltaGroupConfig:
    def __init__(self, *args, **kwargs) -> None:
        pass
class KVCacheGroupSpec:
    def __init__(self, *args, **kwargs) -> None:
        pass

