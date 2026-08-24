from .arch import is_battlemage, is_xe2_family, is_xpu_available
from .hf import (
    cached_load_hf_config,
    download_hf_weight,
    load_eos_token_ids,
    load_generation_sampling,
    load_tokenizer,
    load_toolcall_anchor_id,
)
from .logger import init_logger
from .misc import UNSET, Unset, align_ceil, align_down, call_if_main, div_ceil, div_even, mem_GB
from .mp import (
    ZmqAsyncPullQueue,
    ZmqAsyncPushQueue,
    ZmqPubQueue,
    ZmqPullQueue,
    ZmqPushQueue,
    ZmqSubQueue,
)
from .registry import Registry
from .torch_utils import itt_annotate, torch_dtype

# Upstream aliases so later ports of NVIDIA code compile with fewer edits.
nvtx_annotate = itt_annotate
is_arch_supported = is_xpu_available
is_sm90_family = is_xe2_family
is_sm90_supported = is_xe2_family
is_sm100_family = is_xe2_family
is_sm100_supported = is_xe2_family

__all__ = [
    "cached_load_hf_config",
    "download_hf_weight",
    "load_eos_token_ids",
    "load_generation_sampling",
    "load_tokenizer",
    "load_toolcall_anchor_id",
    "init_logger",
    "is_xpu_available",
    "is_xe2_family",
    "is_battlemage",
    "is_arch_supported",
    "is_sm90_family",
    "is_sm90_supported",
    "is_sm100_family",
    "is_sm100_supported",
    "call_if_main",
    "div_even",
    "div_ceil",
    "align_ceil",
    "align_down",
    "mem_GB",
    "UNSET",
    "Unset",
    "torch_dtype",
    "itt_annotate",
    "nvtx_annotate",
    "Registry",
    "ZmqPushQueue",
    "ZmqPullQueue",
    "ZmqPubQueue",
    "ZmqSubQueue",
    "ZmqAsyncPushQueue",
    "ZmqAsyncPullQueue",
]
