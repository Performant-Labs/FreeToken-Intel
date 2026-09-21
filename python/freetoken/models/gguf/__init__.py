"""Model package stub: gguf.

Upstream NVIDIA path: python/freetoken/models/gguf/
Parent epic: `models-gguf` (#199, see docs/architecture.md).

``reader.py`` (issue `models-gguf-reader`, #270) is implemented: GGUF binary
format parsing (header, typed KV-metadata, tensor-info) is re-exported below.
``parse_config`` (issue `models-gguf-config-tokenizer`, #272) is implemented
below: it maps a GGUF file's raw KV-metadata store (as
:func:`load_gguf_metadata` parses it) into this port's own
:class:`freetoken.models.config.ModelConfig`, the same job every other
architecture's own ``parse_config`` (e.g. ``freetoken.models.qwen3_moe.
parse_config``) does for a HF ``config.json``. GGUF's embedded tokenizer is
``tokenizer.py`` (also #272)'s job -- see that module. ``iter_weights`` /
``GgufModel`` remain stubs -- `models-gguf-iter-and-loader-wiring` (#273)'s job.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from freetoken._stub import unimplemented
from freetoken.models.config import ModelConfig

from .reader import (
    GGUF_MAGIC,
    GGML_TYPE_NAMES,
    GGUFFile,
    GGUFFormatError,
    GGUFTensorInfo,
    GGUFValueType,
    gguf_tensor_info,
    gguf_tensor_names,
    is_gguf_path,
    load_gguf,
    load_gguf_metadata,
)

__all__ = [
    "GGUF_MAGIC",
    "GGML_TYPE_NAMES",
    "GGUFFile",
    "GGUFFormatError",
    "GGUFTensorInfo",
    "GGUFValueType",
    "gguf_tensor_info",
    "gguf_tensor_names",
    "is_gguf_path",
    "load_gguf",
    "load_gguf_metadata",
    "parse_config",
    "iter_weights",
    "GgufModel",
]

# --------------------------------------------------------------------------- #
# KV-metadata -> ModelConfig (issue #272)
# --------------------------------------------------------------------------- #

# llama.cpp's own per-architecture GGUF namespacing convention (verified
# against the real ``gguf`` pip package's ``constants.py`` -- ``Keys.LLM`` /
# ``Keys.Attention`` / ``Keys.Rope`` / ``Keys.SSM`` -- and against
# ``conversion/qwen.py``'s ``Qwen3NextModel``/``Qwen3_5MoeTextModel``
# ``set_gguf_parameters``, both fetched from ggml-org/llama.cpp@master while
# implementing this): every key is ``"{arch}.<suffix>"`` where ``{arch}`` is
# the file's own ``general.architecture`` string (e.g. ``"llama"``,
# ``"qwen35moe"``) -- NOT a fixed namespace shared across architectures.
_DENSE_KEYS = {
    "vocab_size": "vocab_size",
    "context_length": "max_position_embeddings",
    "embedding_length": "hidden_size",
    "block_count": "num_layers",
    "feed_forward_length": "intermediate_size",
    "leading_dense_block_count": "first_k_dense_replace",
}
_ATTENTION_KEYS = {
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "attention.key_length": "head_dim",
}
_MOE_KEYS = {
    "expert_count": "num_experts",
    "expert_used_count": "num_experts_per_tok",
    "expert_feed_forward_length": "moe_intermediate_size",
}

# The GGUF architecture strings this port maps for real (verified against
# ggml-org/llama.cpp@master's conversion/qwen.py, since neither this repo's
# vendored ``gguf`` pip package (0.19.0) nor its own spec doc lists them):
# ``qwen3.5``/``qwen3.6``'s hybrid linear-attention MoE registers as
# ``Qwen3_5MoeForCausalLM`` / ``Qwen3_5MoeForConditionalGeneration`` on the HF
# side and converts to GGUF architecture ``"qwen35moe"`` (``MODEL_ARCH.
# QWEN35MOE`` -> ``"qwen35moe"`` in ``gguf-py/gguf/constants.py``) -- the real
# target checkpoint per epic #199's "Why". Every other architecture (``llama``,
# ``qwen2``, ``qwen3``, ``qwen3moe``, ...) maps through the generic dense/
# attention/MoE key tables above alone -- this port doesn't need a
# per-architecture adapter for them the way qwen35moe's linear-attention
# layers require one.
_QWEN35_MOE_ARCHITECTURES = {"qwen35moe", "qwen35", "qwen3next"}


def _arch_get(metadata: Dict[str, Any], arch: str, suffix: str, default: Any = None) -> Any:
    return metadata.get(f"{arch}.{suffix}", default)


def _apply_key_map(cfg: ModelConfig, metadata: Dict[str, Any], arch: str, key_map: Dict[str, str]) -> None:
    for gguf_suffix, field_name in key_map.items():
        value = _arch_get(metadata, arch, gguf_suffix)
        if value is not None:
            setattr(cfg, field_name, value)


def _qwen35_moe_attrs(metadata: Dict[str, Any], arch: str) -> Dict[str, Any]:
    """The Qwen3.5/3.6 (``qwen35moe``) hybrid-attention-specific fields that
    have no first-class :class:`ModelConfig` slot -- stashed in ``cfg.attrs``,
    mirroring ``freetoken.models.qwen3_5_moe.parse_config``'s own
    ``attrs["text_config"]`` pattern for the same (HF-config-sourced) fields.
    GGUF key spellings per ``conversion/qwen.py``'s ``Qwen3NextModel.
    set_gguf_parameters`` (``qwen35moe`` inherits it via
    ``_LinearAttentionVReorderBase``): the linear-attention (Gated-Delta-Net)
    head/kernel dims live under ``{arch}.ssm.*``, keyed by GDN role rather than
    by GGUF's own generic SSM naming (``ssm.state_size`` is really
    ``linear_key_head_dim``, ``ssm.time_step_rank`` is really
    ``linear_num_value_heads``, etc. -- the conversion script deliberately
    reuses Mamba's SSM key slots for GDN's differently-shaped state rather
    than mint new keys, so the mapping here undoes that reuse by name, not by
    guessing from the generic SSM spec).
    """
    return {
        "linear_conv_kernel_dim": _arch_get(metadata, arch, "ssm.conv_kernel"),
        "linear_key_head_dim": _arch_get(metadata, arch, "ssm.state_size"),
        "linear_num_key_heads": _arch_get(metadata, arch, "ssm.group_count"),
        "linear_num_value_heads": _arch_get(metadata, arch, "ssm.time_step_rank"),
        # ssm.inner_size is linear_value_head_dim * linear_num_value_heads
        # (Qwen3NextModel.set_gguf_parameters writes exactly that product);
        # derive value_head_dim back out when both pieces are present.
        "linear_value_head_dim": (
            (_arch_get(metadata, arch, "ssm.inner_size") // _arch_get(metadata, arch, "ssm.time_step_rank"))
            if _arch_get(metadata, arch, "ssm.inner_size") and _arch_get(metadata, arch, "ssm.time_step_rank")
            else None
        ),
        "full_attention_interval": _arch_get(metadata, arch, "full_attention_interval"),
        # A per-layer bool array: True where that layer is full attention
        # (the GGUF "recurrent_layers" array is the inverse -- True where a
        # layer is the linear/recurrent GDN kind).
        "recurrent_layers": _arch_get(metadata, arch, "attention.recurrent_layers"),
        "partial_rotary_dim": _arch_get(metadata, arch, "rope.dimension_count"),
    }


def parse_config(
    path_or_metadata: "str | Dict[str, Any]",
    *,
    use_offload_moe: bool = False,
    use_cpu_moe: bool = False,
    use_hybrid: bool = False,
    moe_cpu_layers: Optional[str] = None,
    model_path: Optional[str] = None,
) -> ModelConfig:
    """Build a :class:`ModelConfig` from a GGUF file's raw KV-metadata store.

    ``path_or_metadata`` is either a GGUF file/directory path (parsed via
    :func:`load_gguf_metadata`) or an already-parsed metadata dict (a caller
    that already holds a :class:`GGUFFile` / called ``load_gguf_metadata``
    itself never re-parses the file). Torch-free, mirroring every other
    architecture's own ``parse_config``.

    Every field this port's MoE plumbing / attention builder read is mapped
    through llama.cpp's own per-architecture GGUF key convention
    (``"{arch}.<suffix>"``, ``arch`` = the file's ``general.architecture``):
    the dense fields (hidden size, layer count, ...) and MoE fields (expert
    count, top-k, ...) are architecture-agnostic in *spelling* (every
    architecture uses ``{arch}.block_count`` etc., verified against both the
    real ``llama``-architecture test fixture and ``qwen35moe``'s own
    conversion script), so one generic mapping covers every architecture
    GGUF ships. ``qwen35moe`` (the real target's family, Qwen3.5/3.6's hybrid
    linear-attention MoE) additionally carries GDN-specific fields with no
    first-class ``ModelConfig`` slot -- those are stashed in ``cfg.attrs``,
    the same pattern ``freetoken.models.qwen3_5_moe.parse_config`` already
    uses for its own HF-config-sourced extras.

    ``use_offload_moe`` / ``use_cpu_moe`` / ``use_hybrid`` / ``moe_cpu_layers``
    mirror every other architecture's ``parse_config`` signature (the loader
    calls every architecture uniformly); ``model_path`` is accepted for the
    same reason and unused here (GGUF's metadata never needs a checkpoint-
    shape probe the way ``qwen3_moe``'s ``head_dim`` recovery does).
    """
    del model_path  # accepted for call-signature parity with the other architectures' parse_config; unused.
    metadata = path_or_metadata if isinstance(path_or_metadata, dict) else load_gguf_metadata(path_or_metadata)

    arch = metadata.get("general.architecture")
    if not arch:
        raise GGUFFormatError("GGUF metadata has no general.architecture key")

    cfg = ModelConfig(architectures=[str(metadata.get("general.name") or arch)])
    _apply_key_map(cfg, metadata, arch, _DENSE_KEYS)
    _apply_key_map(cfg, metadata, arch, _ATTENTION_KEYS)

    # vocab_size: prefer the checkpoint's own explicit KV entry, but every
    # real GGUF file also carries the tokenizer's own token list -- fall back
    # to its length when the (optional) vocab_size KV key is absent, rather
    # than leaving vocab_size unset (the embedding/lm_head shapes need it).
    if cfg.vocab_size is None:
        tokens = metadata.get("tokenizer.ggml.tokens")
        if isinstance(tokens, list) and tokens:
            cfg.vocab_size = len(tokens)

    # head_dim: GGUF's "{arch}.attention.key_length" (mapped above via
    # _ATTENTION_KEYS) is the real per-head dim when the file sets it
    # explicitly (extended-head architectures, mirroring qwen3_moe's own
    # head_dim-vs-derive distinction). ModelConfig.__post_init__ / every
    # consumer already treats ``None`` as "derive: hidden // heads", so a
    # checkpoint that never sets the key is left unset rather than computed
    # here a second time.

    rope_theta = _arch_get(metadata, arch, "rope.freq_base")
    if rope_theta is not None:
        cfg.rope_theta = float(rope_theta)

    file_type = metadata.get("general.file_type")
    if file_type is not None:
        cfg.attrs["gguf_file_type"] = file_type
    rms_eps = _arch_get(metadata, arch, "attention.layer_norm_rms_epsilon")
    if rms_eps is not None:
        cfg.attrs["rms_norm_eps"] = float(rms_eps)

    # MoE fields: only present in the metadata for a MoE checkpoint (the
    # generic dense-transformer architectures -- e.g. this issue's own
    # ``llama`` test fixture -- never set {arch}.expert_count at all).
    num_experts = _arch_get(metadata, arch, "expert_count")
    if num_experts:
        _apply_key_map(cfg, metadata, arch, _MOE_KEYS)
        cfg.is_moe = True
        cfg.num_moe_layers = (cfg.num_layers - cfg.first_k_dense_replace) if cfg.num_layers else None

    if arch in _QWEN35_MOE_ARCHITECTURES:
        cfg.attrs["gguf_linear_attention"] = _qwen35_moe_attrs(metadata, arch)

    cfg.use_offload_moe = bool(use_offload_moe)
    cfg.use_cpu_moe = bool(use_cpu_moe)
    cfg.use_hybrid = bool(use_hybrid)
    cfg.moe_cpu_layers = moe_cpu_layers
    return cfg


def iter_weights(*args, **kwargs):
    unimplemented("iter_weights", "models-gguf-iter-and-loader-wiring")


class GgufModel:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(self, *args, **kwargs):
        unimplemented("GgufModel.forward", "models-gguf-iter-and-loader-wiring")
