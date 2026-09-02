from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List

from freetoken.distributed import DistributedInfo

if TYPE_CHECKING:
    from freetoken.models import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: object  # torch.dtype once the XPU runtime is wired
    device: object = None  # torch.device | str; None -> XPU if available else CPU
    max_running_req: int = 4
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    # Xe2 XMX GEMM path for MXFP4 / INT8 experts (replaces --nvfp4-backend).
    mxfp4_backend: str = "auto"
    expert_load: str = "auto"
    moe_cache_size: int = 0
    moe_cache_rate: float | None = None
    moe_cache_auto: bool = False
    kv_reserve_tokens: int = 8192
    moe_cache_policy: str = "lru"
    moe_prefill_overlap: bool = True
    moe_prefill_hit_d2d: bool = False
    moe_collect_stats: bool = False
    moe_cpu_threads: int = 0
    moe_cpu_layers: str | None = None
    moe_hybrid_max_fetch: int = -1
    # Level Zero command-list / SYCL-graph capture sizes (upstream: cuda_graph_bs).
    xpu_graph_bs: List[int] | None = None
    xpu_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    linear_state_cache_ratio: float = 2.0
    swa_full_tokens_ratio: float = 0.2
    swa_num_pages_override: int | None = None
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_oneccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None
    num_token_override: int | None = None

    def __post_init__(self) -> None:
        # A pinned moe_cache_size is the operator saying "use exactly this many
        # slots"; auto-planning the split from VRAM would contradict that. So a
        # positive pin disables auto (the engine checks moe_cache_auto). 0 =
        # unset, i.e. let auto plan.
        if self.moe_cache_size and not self.moe_cache_auto:
            return
        if self.moe_cache_size and self.moe_cache_auto:
            object.__setattr__(self, "moe_cache_auto", False)

    @cached_property
    def hf_config(self):
        from freetoken.utils import cached_load_hf_config

        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        from freetoken.models.register import _load_attr, get_model_spec

        spec = get_model_spec(self.hf_config.architectures[0])
        parse_config = _load_attr(spec.module, spec.parse_config)
        return parse_config(self.hf_config)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        # The parsed ModelConfig carries the context length directly; the
        # optional RotaryConfig (when present) may override it.
        rotary = getattr(self.model_config, "rotary_config", None)
        if rotary is not None and getattr(rotary, "max_position", None):
            return rotary.max_position
        return self.model_config.max_position_embeddings

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
