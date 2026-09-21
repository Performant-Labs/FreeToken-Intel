"""Model package stub: gguf.

Upstream NVIDIA path: python/freetoken/models/gguf/
Parent epic: `models-gguf` (#199, see docs/architecture.md).

``reader.py`` (issue `models-gguf-reader`, #270) is implemented: GGUF binary
format parsing (header, typed KV-metadata, tensor-info) is re-exported below.
``dequant.py`` (issue `models-gguf-dequant`, #271) is also implemented: GGML
quant-type dequantization (``Q4_0``/``Q8_0``/``Q4_K``/``Q6_K``). Unlike
``reader.py``, ``dequant.py`` requires torch, so it is *not* imported here at
package-import time (that would break this package's own torch-free
CPU-CLI-smoke import path -- see ``reader.py``'s docstring); instead
``dequantize``/``DequantError``/``GGML_NAME_TO_TYPE`` are re-exported lazily
below via module ``__getattr__`` (PEP 562), so ``import freetoken.models.gguf``
stays torch-free but ``freetoken.models.gguf.dequantize(...)`` still works,
importing torch only at that first access.
``parse_config`` (issue `models-gguf-config-tokenizer`, #272) is also
implemented below: it maps a GGUF file's raw KV-metadata store (as
:func:`load_gguf_metadata` parses it) into this port's own
:class:`freetoken.models.config.ModelConfig`, the same job every other
architecture's own ``parse_config`` (e.g. ``freetoken.models.qwen3_moe.
parse_config``) does for a HF ``config.json``. GGUF's embedded tokenizer is
``tokenizer.py`` (also #272)'s job -- see that module. ``iter_weights`` /
``GgufModel`` remain stubs -- `models-gguf-iter-and-loader-wiring` (#273)'s
job, which builds directly on all of the above.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

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
    "GGML_NAME_TO_TYPE",
    "GGUFFile",
    "GGUFFormatError",
    "GGUFTensorInfo",
    "GGUFValueType",
    "DequantError",
    "dequantize",
    "gguf_tensor_info",
    "gguf_tensor_names",
    "is_gguf_path",
    "load_gguf",
    "load_gguf_metadata",
    "parse_config",
    "iter_weights",
    "iter_moe_expert_raw_banks",
    "gguf_expert_row_bytes",
    "GgufModel",
    "GGUF_ARCH_TO_REGISTRY_KEY",
]

# --------------------------------------------------------------------------- #
# Lazy torch-requiring re-exports (issue #271)
# --------------------------------------------------------------------------- #

_DEQUANT_LAZY_NAMES = {"dequantize", "DequantError", "GGML_NAME_TO_TYPE"}


def __getattr__(name: str):
    # PEP 562 lazy attribute access: defers importing `dequant.py` (and
    # thus torch) until a caller actually touches one of its names, so
    # `import freetoken.models.gguf` alone stays torch-free.
    if name in _DEQUANT_LAZY_NAMES:
        from . import dequant

        return getattr(dequant, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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


def _gguf_real_num_layers(metadata: Dict[str, Any], arch: str, block_count: int) -> int:
    """The checkpoint's real decoder-layer count -- ``block_count`` minus any
    ``nextn_predict_layers`` (issue #287).

    Root cause of #287's "loads and generates, but garbage output" bug:
    some GGUF checkpoints (this port's own real target, Qwen3.6-35B-A3B,
    included) append MTP (multi-token-prediction speculative-decode draft
    head) blocks onto the END of the tensor namespace, using the exact same
    ``blk.N.*`` naming convention a real decoder layer uses -- including a
    full spare ``attn_q``/``attn_k``/``attn_v``/``ffn_gate_inp``/
    ``ffn_*_exps``/``ffn_*_shexp`` tensor set, which is what made this look
    like a real "extra full-attention layer the interval formula doesn't
    predict" (see the pre-fix ``_qwen35_moe_layer_types`` docstring, and
    issue #279's own investigation) rather than what it actually is: an
    entirely separate, auxiliary module this port's engine never runs.

    The GGUF spec has an explicit, unambiguous signal for this --
    ``{arch}.nextn_predict_layers`` (verified against the real target
    checkpoint's own header: ``qwen35moe.block_count=41``,
    ``qwen35moe.nextn_predict_layers=1``) -- which #279's own
    implementation read the checkpoint's tensor names for evidence of an
    extra layer but never cross-checked against this metadata field. An
    independent reference implementation (``llama.cpp``, built locally with
    Vulkan) confirms this directly: loading the same real checkpoint file
    logs ``model has unused tensor blk.40.*.weight -- ignoring`` for
    *every* tensor of the last block (attn/ffn/shexp AND the ``nextn.*``
    tensors this port already dropped) and produces a coherent answer,
    proving upstream treats block 40 as 100% outside the main 40-layer
    decoder stack, not merely "MTP metadata to ignore within an otherwise
    real 41st layer."

    Every caller that walks ``blk.N.*`` tensors by real per-layer index
    (``iter_weights``, ``iter_moe_expert_raw_banks``, and -- via
    ``cfg.num_layers`` -- ``_qwen35_moe_layer_types``'s own ``range(
    num_layers)`` loop) must stop before the MTP block(s), or the engine
    builds and runs a spurious extra decoder layer whose weights are
    semantically a different module's, corrupting the final hidden state
    right before ``output_norm``/``lm_head`` -- silently, since it never
    produces a NaN/Inf, only wrong values (exactly what #287's own
    numerical tracing found: healthy layer-by-layer hidden-state norms, but
    incoherent final output).
    """
    nextn = _arch_get(metadata, arch, "nextn_predict_layers")
    if not nextn:
        return block_count
    return max(0, block_count - int(nextn))


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


def _qwen35_moe_layer_types(model_path: Optional[str], num_layers: int) -> Optional[list]:
    """Per-layer ``'full_attention'`` / ``'linear_attention'`` split, read
    directly from a real ``qwen35moe`` GGUF file's own per-layer tensor
    names (issue #279) -- a full-attention layer carries ``attn_q.weight``,
    a linear-attention (Gated-Delta-Net) layer carries ``attn_qkv.weight``
    instead (see ``_QWEN35_MOE_SUFFIX_MAP``'s own docstring for the full
    tensor-name evidence).

    This is NOT redundant with ``_qwen35_moe_attrs``'s own
    ``full_attention_interval`` field: the real target checkpoint (Zot:
    ``general/qwen3.6-35b-a3b:q4_k_m-gguf``, directly inspected via an HTTP
    range request while building this issue) does *not* follow a clean
    interval pattern within its real decoder stack -- ``full_attention_
    interval=4`` predicts exactly one full-attention layer per 4 (indices
    3, 7, 11, ..., 39). ``num_layers`` here is already the checkpoint's
    REAL decoder-layer count (issue #287's own fix,
    :func:`_gguf_real_num_layers` -- ``block_count`` minus any trailing MTP
    block(s), e.g. 40 for the real target, not the raw ``block_count`` of
    41), so this loop never reaches the MTP block at all and needs no
    special-case for it. (An earlier version of this function treated that
    MTP block as an extra, formula-defying 41st full-attention layer --
    that was #287's root cause: an independent reference implementation,
    ``llama.cpp``, proved the block's attn/ffn tensors are the MTP draft
    head's own, not a real decoder layer's, by ignoring every one of them
    during ordinary generation.) Returns ``None`` (falls back to the
    ``qwen3_5_moe`` forward's own interval-formula default) when
    ``model_path`` is absent or not a readable GGUF file -- e.g. a caller
    that only has the raw metadata dict (this module's own synthetic
    config-only tests), matching the pre-#279 behavior for that case.
    """
    if not model_path:
        return None
    try:
        names = set(gguf_tensor_names(model_path))
    except Exception:
        return None
    layer_types = []
    for i in range(num_layers):
        if f"blk.{i}.attn_qkv.weight" in names:
            layer_types.append("linear_attention")
        elif f"blk.{i}.attn_q.weight" in names:
            layer_types.append("full_attention")
        else:
            # An unrecognized/missing layer shape (e.g. a caller-crafted
            # metadata-only path that doesn't correspond to this file's real
            # tensors) -- rather than guess, defer to the interval-formula
            # fallback for every layer.
            return None
    return layer_types


def _qwen35_moe_text_config(
    metadata: Dict[str, Any],
    arch: str,
    cfg: ModelConfig,
    linear_attrs: Dict[str, Any],
    model_path: Optional[str],
) -> Dict[str, Any]:
    """The ``cfg.attrs["text_config"]`` shape ``freetoken.models.qwen3_5_moe``'s
    own forward pass (``_Qwen35DecoderLayer.__init__``,
    ``Qwen3_5MoEForCausalLM.__init__``, ``_Qwen35MoE.__init__``) actually
    reads -- built from the same GDN-role fields ``_qwen35_moe_attrs``
    already extracted (``linear_attrs``, this module's own #272 convention),
    plus the handful of extra fields that convention doesn't carry
    (``partial_rotary_factor`` as a *fraction*, not a raw dim count;
    ``shared_expert_intermediate_size``; the exact per-layer ``layer_types``
    split, see :func:`_qwen35_moe_layer_types`).
    """
    head_dim = cfg.head_dim or (
        (cfg.hidden_size // cfg.num_attention_heads) if cfg.num_attention_heads else None
    )
    partial_rotary_dim = linear_attrs.get("partial_rotary_dim")
    partial_rotary_factor = (
        (partial_rotary_dim / head_dim) if partial_rotary_dim and head_dim else 1.0
    )
    text_config: Dict[str, Any] = {
        "rms_norm_eps": cfg.attrs.get("rms_norm_eps", 1e-6),
        "linear_num_key_heads": linear_attrs.get("linear_num_key_heads"),
        "linear_num_value_heads": linear_attrs.get("linear_num_value_heads"),
        "linear_key_head_dim": linear_attrs.get("linear_key_head_dim"),
        "linear_value_head_dim": linear_attrs.get("linear_value_head_dim"),
        "linear_conv_kernel_dim": linear_attrs.get("linear_conv_kernel_dim"),
        "partial_rotary_factor": partial_rotary_factor,
    }
    if linear_attrs.get("full_attention_interval") is not None:
        text_config["full_attention_interval"] = linear_attrs["full_attention_interval"]
    shared_inter = _arch_get(metadata, arch, "expert_shared_feed_forward_length")
    if shared_inter is not None:
        text_config["shared_expert_intermediate_size"] = shared_inter
    layer_types = _qwen35_moe_layer_types(model_path, int(cfg.num_layers or 0))
    if layer_types is not None:
        text_config["layer_types"] = layer_types
    return text_config


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
    same reason and is mostly unused here (GGUF's metadata never needs a
    checkpoint-shape probe the way ``qwen3_moe``'s ``head_dim`` recovery
    does) -- EXCEPT for ``qwen35moe``'s per-layer hybrid split (issue
    #279): the real target checkpoint's own header does not follow a clean
    ``full_attention_interval`` pattern (it carries one extra full-attention
    layer beyond what the formula predicts, confirmed against the real
    Qwen3.6-35B-A3B checkpoint's own header), so when ``model_path`` names a
    readable GGUF file, ``_qwen35_moe_layer_types`` reads the actual
    per-layer tensor names to build an exact ``layer_types`` list instead of
    trusting the formula.
    """
    if isinstance(path_or_metadata, dict):
        metadata = path_or_metadata
    elif hasattr(path_or_metadata, "to_dict"):
        # issue #273: loader.py calls every architecture's parse_config
        # uniformly as parse_config(hf_config, ...), where hf_config is
        # normally a real HF PretrainedConfig. For a GGUF checkpoint the
        # loader instead hands this function a lightweight shim
        # (_GgufConfigShim, weight.py) carrying the same .to_dict() ->
        # metadata contract, so this one extra branch is all that's needed
        # to make every downstream hf_config.architectures[0] /
        # re-parse-on-backend-change code path in loader.py work
        # unchanged for GGUF too.
        metadata = path_or_metadata.to_dict()
    else:
        metadata = load_gguf_metadata(path_or_metadata)
        # A caller that passed the GGUF path directly as path_or_metadata
        # (rather than via the model_path kwarg the real loader always
        # supplies) still gets the tensor-name-verified qwen35moe
        # layer_types split below -- there is no reason to require both.
        if model_path is None:
            model_path = path_or_metadata

    arch = metadata.get("general.architecture")
    if not arch:
        raise GGUFFormatError("GGUF metadata has no general.architecture key")

    cfg = ModelConfig(architectures=[str(metadata.get("general.name") or arch)])
    _apply_key_map(cfg, metadata, arch, _DENSE_KEYS)
    _apply_key_map(cfg, metadata, arch, _ATTENTION_KEYS)
    if cfg.num_layers is not None:
        cfg.num_layers = _gguf_real_num_layers(metadata, arch, int(cfg.num_layers))

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
        linear_attrs = _qwen35_moe_attrs(metadata, arch)
        cfg.attrs["gguf_linear_attention"] = linear_attrs
        # freetoken.models.qwen3_5_moe's own forward (_Qwen35DecoderLayer /
        # _Qwen35Attention / Qwen3_5MoEForCausalLM.__init__) does NOT read
        # the GDN dims from cfg.attrs["gguf_linear_attention"] above (that
        # key is this module's own #272 convention) -- it reads
        # cfg.attrs["head_dim"], cfg.attrs["rope_theta"], and
        # cfg.attrs["text_config"][...] (its own HF-sourced convention, see
        # that module's own parse_config). A GGUF-loaded qwen35moe config
        # must therefore ALSO populate those exact keys, or model
        # construction crashes with a KeyError the moment the loader builds
        # the real forward-side model -- found empirically while wiring
        # this issue end to end (#278's own qwen3moe path never hits this:
        # Qwen3MoeForCausalLM's forward reads plain ModelConfig fields, not
        # config.attrs["text_config"]).
        cfg.attrs["head_dim"] = cfg.head_dim
        cfg.attrs["rope_theta"] = cfg.rope_theta
        cfg.attrs["text_config"] = _qwen35_moe_text_config(metadata, arch, cfg, linear_attrs, model_path)

    cfg.use_offload_moe = bool(use_offload_moe)
    cfg.use_cpu_moe = bool(use_cpu_moe)
    cfg.use_hybrid = bool(use_hybrid)
    cfg.moe_cpu_layers = moe_cpu_layers
    return cfg


# --------------------------------------------------------------------------- #
# GGUF tensor name -> this port's HF-style parameter name (issue #273)
# --------------------------------------------------------------------------- #

import re as _re

_LAYER_RE = _re.compile(r"^blk\.(\d+)\.(.+)$")

# The standard (non-GDN) transformer block's GGUF tensor-name suffixes ->
# this port's HF-style trailing parameter name. Verified against
# gguf-py's own MODEL_TENSORS[MODEL_ARCH.QWEN3MOE] table (matches upstream
# llama.cpp's real tensor names 1:1 with Qwen3MoeForCausalLM's own HF
# key spelling -- q/k/v are separate projections with their own q_norm/
# k_norm), used for every architecture EXCEPT qwen35moe (dispatched via
# GGUF_ARCH_TO_SUFFIX_MAP below): qwen35moe's fused attn_qkv +
# Gated-Delta-Net linear-attention layers + shared-expert MoE use a
# genuinely different tensor set, mapped separately by
# _QWEN35_MOE_SUFFIX_MAP (issue #279).
_DENSE_SUFFIX_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
    # Dense (non-MoE) MLP -- a dense-transformer architecture's own layers,
    # or a MoE architecture's leading dense layers (first_k_dense_replace).
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}

_TOP_LEVEL_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}

# qwen35moe (Qwen3.5/3.6's hybrid linear-attention + shared-expert MoE,
# issue #279): the real target's own tensor set. Verified directly against
# the real target checkpoint's own header (Zot registry:
# ``general/qwen3.6-35b-a3b:q4_k_m-gguf``, read via an HTTP range request,
# no full 22.7GB pull needed) AND against gguf-py's own
# ``MODEL_TENSORS[MODEL_ARCH.QWEN35MOE]`` table -- the real file's actual
# per-layer tensor names for both layer kinds, cross-checked against
# freetoken.models.qwen3_5_moe's own real parameter names (that module's
# ``_GatedDeltaNet`` / ``_Qwen35Attention`` / ``_Qwen35MoE`` /
# ``_Qwen35DecoderLayer`` __init__ bodies, read in full while building this
# mapping -- NOT guessed):
#
# Full-attention (gated GQA) layer -- identical spelling to _DENSE_SUFFIX_MAP's
# qwen3moe entries (separate q/k/v, their own q_norm/k_norm), EXCEPT the
# post-attention norm's own GGUF name: qwen35moe spells it
# "post_attention_norm.weight", not qwen3moe's "ffn_norm.weight" (confirmed
# against the real checkpoint's own header -- both layer kinds share this
# tensor).
#
# Linear-attention (Gated-Delta-Net) layer -- the real per-layer tensor set,
# with shapes (GGUF's ne-order, reversed per _gguf_to_pytorch_shape) checked
# against _GatedDeltaNet's own constructor one by one on the real checkpoint:
#   attn_qkv.weight    [key_dim*2+value_dim, hidden]  -> in_proj_qkv.weight
#   attn_gate.weight   [value_dim, hidden]             -> in_proj_z.weight
#     (NOT the full-attention output gate -- qwen35moe's full-attention gate
#     is fused into attn_q's own doubled output width instead, matching
#     _Qwen35Attention.q_proj's own [num_heads*head_dim*2, hidden] shape; a
#     linear-attention layer never has an attn_q tensor at all, so there is
#     no ambiguity between the two roles.)
#   ssm_alpha.weight   [num_v_heads, hidden]           -> in_proj_a.weight
#     (_GatedDeltaNet.forward's own "a" -- the decay-rate input; GGUF's own
#     "alpha" name matches this role's own math exactly, not a guess.)
#   ssm_beta.weight    [num_v_heads, hidden]           -> in_proj_b.weight
#     (_GatedDeltaNet.forward's own "b" -- the delta-rule beta input.)
#   ssm_conv1d.weight  [conv_dim, kernel] (2-D on disk) -> conv1d.weight
#     (nn.Conv1d's own weight is 3-D, [conv_dim, 1, kernel] -- groups=conv_dim
#     means in_channels/groups=1 -- so this ONE tensor needs an inserted
#     middle dim at load time; see _QWEN35_MOE_RESHAPE / iter_weights.)
#   ssm_dt.bias        [num_v_heads]                   -> dt_bias
#     (stored under a ".bias" GGUF suffix, not ".weight" -- copied as-is,
#     already the exact target shape.)
#   ssm_a              [num_v_heads] (no suffix at all) -> A_log
#   ssm_norm.weight    [head_v_dim]                     -> norm.weight
#     (_RMSNormGated's own weight -- the Gated-Delta-Net output norm.)
#   ssm_out.weight     [hidden, value_dim]               -> out_proj.weight
#
# Shared expert (always-on, dense -- never in the routed-expert banks):
#   ffn_gate_inp_shexp.weight  [hidden] (1-D on disk)  -> shared_expert_gate.weight
#     (LinearReplicated(hidden, 1)'s weight is 2-D, [1, hidden] -- this ONE
#     tensor needs a leading dim inserted at load time; see
#     _QWEN35_MOE_RESHAPE / iter_weights.)
#   ffn_gate_shexp.weight / ffn_up_shexp.weight / ffn_down_shexp.weight
#     -> shared_expert.gate_proj.weight / up_proj.weight / down_proj.weight
#     (plain dense Linear weights, same shape convention as a routed
#     expert's own gate_proj/up_proj/down_proj -- but NOT packed into the
#     [E, ...] bank format: the shared expert is always dense/on-device, see
#     _Qwen35MoE.__init__'s own ``self.shared_expert`` -- a single
#     _Qwen35Expert, not an nn.ModuleList.)
#
# Routed experts (ffn_gate_exps / ffn_up_exps / ffn_down_exps) and the
# router (ffn_gate_inp.weight -> mlp.gate.weight) are NOT listed here: the
# generic per-expert packing logic in iter_weights (issue #273/#278) already
# handles those identically for every MoE architecture, unconditional on
# this suffix map.
#
# Also NOT listed (and so silently dropped by iter_weights, by design): the
# real checkpoint's very last layer additionally carries a
# ``nextn.{hnorm,enorm,eh_proj,shared_head_norm}.weight`` MTP (multi-token-
# prediction) draft-head tensor set (matching its ``nextn_predict_layers=1``
# KV entry) -- confirmed directly against the real checkpoint's own header.
# This port's engine does not run MTP, so these are intentionally never
# mapped, mirroring ``freetoken.models.qwen3_5_moe.iter_weights``'s own
# ``mtp.*`` drop for the safetensors checkpoint shape of this same model.
_QWEN35_MOE_SUFFIX_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
    # Full-attention (gated GQA) layers.
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    # Linear-attention (Gated-Delta-Net) layers.
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_a": "linear_attn.A_log",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    # Shared expert (dense, always-on).
    "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",
    "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
}

# The two qwen35moe tensors whose on-disk shape is not already the exact
# target parameter shape (see _QWEN35_MOE_SUFFIX_MAP's own docstring for
# why each needs exactly this reshape, verified against the real target
# checkpoint's header): applied to the dequantized (already
# _gguf_to_pytorch_shape-reversed) tensor, keyed by the GGUF suffix (the
# same key _QWEN35_MOE_SUFFIX_MAP uses), in iter_weights below.
_QWEN35_MOE_RESHAPE = {
    "ssm_conv1d.weight": lambda t: t.unsqueeze(1),  # [C, K] -> [C, 1, K]
    "ffn_gate_inp_shexp.weight": lambda t: t.unsqueeze(0),  # [H] -> [1, H]
}

# GGUF architecture string -> the suffix map iter_weights uses for that
# architecture's per-layer non-MoE-expert tensors (issue #279: qwen35moe
# needs its own map; every other supported architecture keeps using the
# generic _DENSE_SUFFIX_MAP, unchanged from #273/#278).
GGUF_ARCH_TO_SUFFIX_MAP: Dict[str, Dict[str, str]] = {
    arch: _QWEN35_MOE_SUFFIX_MAP for arch in _QWEN35_MOE_ARCHITECTURES
}

# GGUF's per-architecture ``general.architecture`` string -> this port's
# model-registry key (``register.py``'s ``_MODEL_REGISTRY`` dict), needed
# because ``get_model_spec`` dispatches on the registry key, not the raw
# GGUF architecture string (which is lowercase/no-suffix, e.g. "qwen3moe"
# vs. the registry's "Qwen3MoeForCausalLM"). Verified against gguf-py's
# own ``MODEL_ARCH_NAMES`` table for each architecture this port's
# registry actually knows -- only architectures both sides support are
# listed; anything else falls back to the raw GGUF string (which
# ``get_model_spec`` then rejects with a clear "not supported" error,
# same as an unregistered HF ``architectures[0]`` value would).
GGUF_ARCH_TO_REGISTRY_KEY: Dict[str, str] = {
    "qwen3moe": "Qwen3MoeForCausalLM",
    "qwen35moe": "Qwen3_5MoeForConditionalGeneration",
    "llama": "LlamaForCausalLM",
    "qwen2": "Qwen2ForCausalLM",
    "qwen3": "Qwen3ForCausalLM",
}


def _gguf_to_pytorch_shape(shape) -> tuple:
    """GGUF's on-disk tensor-info shape is ggml's own ``ne[]`` order (the
    *fastest-varying* dimension listed first); every consumer of a
    dequantized tensor here needs standard PyTorch/numpy row-major shape,
    which is that same tuple **reversed**.

    Empirically verified while building this issue (#273): the reference
    ``gguf`` pip package's own ``ReaderTensor.shape`` property reports the
    identical raw, un-reversed ``ne[]`` order this port's own
    :class:`~freetoken.models.gguf.reader.GGUFTensorInfo` does (confirmed
    byte-for-byte identical in #270/#271's own tests) -- but
    ``gguf.GGUFReader._build_tensors`` builds that tensor's *actual data
    array* with ``np_dims = tuple(reversed(dims))``, i.e. gguf-py's own
    tensor **values** are only in the semantically-correct PyTorch/numpy
    orientation after this same reversal. #270/#271's tests never needed
    to apply it themselves: they only cross-checked raw dequantized *byte
    values* against the reference package using its own (matching,
    un-reversed) ``.shape`` attribute for the reshape target on *both*
    sides, never against its separately-and-correctly-oriented ``.data``
    array -- so their passing tests validated bit-exact dequantization,
    not tensor *orientation*, which is why this reversal wasn't caught
    (or needed) until a real weight matrix had to be used in a real
    matmul here. Directly re-verified for this issue against
    ``gguf.GGUFReader(...).tensors[i].data`` on a real checkpoint's real
    ``token_embd.weight`` (see this PR's own test/description).
    """
    return tuple(reversed(shape))


def _gguf_block_size_and_type_size(ggml_type: int) -> Tuple[int, int]:
    """(block_size, type_size) for a GGML quant type -- how many raw bytes
    one block of the on-disk tensor occupies, needed to read exactly a
    tensor's own byte span (not into the next tensor). Delegates to the
    reference ``gguf`` pip package's own ``GGML_QUANT_SIZES`` table (a
    real runtime dependency of this port already, see ``dequant.py``'s own
    docstring and ``pyproject.toml``) rather than duplicating a second
    copy of this table here.
    """
    import gguf as _gguf_ref

    qtype = _gguf_ref.GGMLQuantizationType(ggml_type)
    return _gguf_ref.GGML_QUANT_SIZES[qtype]


def gguf_expert_row_bytes(ggml_type: int, n_elements: int) -> int:
    """Real packed byte length of ONE expert's raw GGML-quantized row, given
    just ``ggml_type`` and the (architecture-constant) element count of one
    expert's weight matrix -- issue `models-gguf-lazy-packed-dequant`, #282.

    This is the same arithmetic :func:`iter_moe_expert_raw_banks` uses to
    slice a whole packed ``ffn_*_exps`` tensor's raw bytes into per-expert
    rows, exposed here so a caller that only holds a (possibly zero-padded,
    see :func:`freetoken.models.weight.stream_moe_expert_sources_gguf_kquant`'s
    own docstring for why a bank row can be padded) bank row can recover the
    REAL, unpadded byte length before decoding it --
    :class:`freetoken.moe.offload_cache.SlotWeightAccessor` is exactly this
    caller, at compute time.
    """
    block_size, type_size = _gguf_block_size_and_type_size(ggml_type)
    if n_elements % block_size != 0:
        raise GGUFFormatError(
            f"gguf_expert_row_bytes: {n_elements} elements is not a multiple of "
            f"block size {block_size} for ggml_type {ggml_type}"
        )
    return (n_elements // block_size) * type_size


def iter_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool = True,
    include_non_moe: bool = True,
):
    """Yield ``(name, tensor)`` for every tensor of a GGUF checkpoint,
    dequantized (:func:`dequantize`, issue #271) and renamed to this
    port's HF-style parameter names -- the same contract every other
    architecture's own ``iter_weights`` fulfills (e.g.
    ``freetoken.models.qwen3_moe.iter_weights``'s docstring): dense
    tensors on ``device``, MoE expert tensors on host memory (the offload
    banks the engine streams from).

    GGUF stores a MoE layer's experts as one already-packed
    ``[num_experts, ...]`` tensor per projection (``ffn_gate_exps`` /
    ``ffn_up_exps`` / ``ffn_down_exps``), not per-expert tensors --
    verified against the real target checkpoint's own header (Zot:
    ``general/qwen3.6-35b-a3b:q4_k_m-gguf``, read via an HTTP range
    request rather than the full 22.7GB pull). ``gate`` and ``up`` are
    concatenated on dim 1 into a single ``gate_up_proj`` tensor before
    yielding, matching :func:`freetoken.models.weight
    .stream_moe_expert_sources`'s own packed-form contract exactly
    (``[E, 2I, H]``, gate then up -- see that function's docstring), so
    this reaches the loader's *existing*, already-tested packed-bank path
    unchanged, with no new bank-building logic needed. This packing is
    architecture-agnostic (every MoE architecture GGUF ships names its
    routed-expert tensors this same way), unlike the *non*-expert per-layer
    tensors, whose suffix map is dispatched by architecture (issue #279:
    ``qwen35moe``'s fused-QKV / Gated-Delta-Net / shared-expert tensor set
    is genuinely different from the generic ``_DENSE_SUFFIX_MAP`` every
    other supported architecture uses -- see ``_QWEN35_MOE_SUFFIX_MAP``'s
    own docstring).
    """
    import torch as _torch

    from .dequant import dequantize as _dequantize

    gguf_file = load_gguf(model_path)
    arch = gguf_file.metadata.get("general.architecture")
    suffix_map = GGUF_ARCH_TO_SUFFIX_MAP.get(arch, _DENSE_SUFFIX_MAP)
    reshape_map = _QWEN35_MOE_RESHAPE if arch in _QWEN35_MOE_ARCHITECTURES else {}
    block_count = _arch_get(gguf_file.metadata, arch, "block_count")
    real_num_layers = (
        _gguf_real_num_layers(gguf_file.metadata, arch, int(block_count)) if block_count else None
    )

    by_layer: Dict[int, Dict[str, GGUFTensorInfo]] = {}
    top_level: Dict[str, GGUFTensorInfo] = {}
    for name, info in gguf_file.tensors.items():
        m = _LAYER_RE.match(name)
        if m:
            layer_idx = int(m.group(1))
            # Issue #287: a trailing MTP (multi-token-prediction) block uses
            # the exact same blk.N.* naming as a real decoder layer -- see
            # _gguf_real_num_layers's own docstring for why every one of its
            # tensors (not just nextn.*) must be excluded here, not just
            # skipped by suffix.
            if real_num_layers is not None and layer_idx >= real_num_layers:
                continue
            by_layer.setdefault(layer_idx, {})[m.group(2)] = info
        else:
            top_level[name] = info

    with open(gguf_file.path, "rb") as fh:

        def _read_and_dequant(info: GGUFTensorInfo) -> "_torch.Tensor":
            block_size, type_size = _gguf_block_size_and_type_size(info.ggml_type)
            n_bytes = (info.n_elements // block_size) * type_size
            fh.seek(info.offset)
            raw = fh.read(n_bytes)
            return _dequantize(info.ggml_type, raw, _gguf_to_pytorch_shape(info.shape))

        if include_non_moe:
            for gguf_name, hf_name in _TOP_LEVEL_MAP.items():
                info = top_level.get(gguf_name)
                if info is not None:
                    yield hf_name, _read_and_dequant(info).to(device)

        for layer in sorted(by_layer):
            suffixes = by_layer[layer]
            prefix = f"model.layers.{layer}"
            if include_non_moe:
                for gguf_suffix, hf_suffix in suffix_map.items():
                    info = suffixes.get(gguf_suffix)
                    if info is not None:
                        tensor = _read_and_dequant(info)
                        reshape = reshape_map.get(gguf_suffix)
                        if reshape is not None:
                            tensor = reshape(tensor)
                        yield f"{prefix}.{hf_suffix}", tensor.to(device)
            if include_moe_experts:
                gate = suffixes.get("ffn_gate_exps.weight")
                up = suffixes.get("ffn_up_exps.weight")
                down = suffixes.get("ffn_down_exps.weight")
                if gate is not None and up is not None:
                    gate_up = _torch.cat([_read_and_dequant(gate), _read_and_dequant(up)], dim=1)
                    yield f"{prefix}.mlp.experts.gate_up_proj", gate_up.to("cpu")
                if down is not None:
                    yield f"{prefix}.mlp.experts.down_proj", _read_and_dequant(down).to("cpu")


def iter_moe_expert_raw_banks(model_path: str):
    """Yield ``(layer, bank_name, raw, ggml_type)`` for every MoE layer's
    packed routed-expert tensor (``ffn_gate_exps`` / ``ffn_up_exps`` /
    ``ffn_down_exps``, ``bank_name`` one of ``"gate"``/``"up"``/``"down"``)
    -- reading the RAW on-disk GGML-quantized bytes straight into a
    ``[num_experts, row_bytes]`` uint8 tensor, WITHOUT ever dequantizing
    (issue `models-gguf-lazy-packed-dequant`, #282).

    This is the fix for the RAM blowup :func:`iter_weights` has for the MoE
    expert banks specifically: that function reads each ``ffn_*_exps``
    tensor and immediately dequantizes the WHOLE ``[num_experts, ...]``
    layer to bf16 (a real target checkpoint's own math, from #282's own
    issue body: 256 experts x 41 layers -> ~66GB of bf16 host RAM, when the
    on-disk ``q4_k_m`` quant is only 22.7GB). This function instead keeps
    every expert's bytes in their original packed, quantized form -- the
    same "packed host bank, dequantize lazily at compute time" design this
    port already uses for GPTQ/FP8/MXFP4/INT8 (see
    :mod:`freetoken.moe.offload_cache`'s own module docstring) -- so the
    resident host RAM after this function is fully consumed is bounded by
    roughly the checkpoint's own on-disk MoE-tensor size, not its
    dequantized-to-bf16 expansion.

    GGUF stores each ``ffn_*_exps`` tensor with the expert axis as its own
    (raw, ne-order) SLOWEST-varying dimension -- verified against the real
    target checkpoint's own header (Zot: ``general/qwen3.6-35b-a3b:q4_k_m-
    gguf``): ``ffn_gate_exps.weight``'s raw shape is ``(hidden,
    moe_intermediate, num_experts)`` (``num_experts`` last), same for
    ``ffn_up_exps``; ``ffn_down_exps``'s raw shape is ``(moe_intermediate,
    hidden, num_experts)``. Since GGUF's on-disk byte layout is C-order in
    that same (ne-index-descending) axis order, expert ``e``'s bytes occupy
    one CONTIGUOUS ``row_bytes``-sized span, in ascending expert order, with
    no interleaving between experts -- so the whole tensor's raw byte blob
    can be read in one ``file.read()`` and reshaped directly into
    ``[num_experts, row_bytes]``, with no per-expert seek/read and no
    dequantization at all.

    Verified against the real target checkpoint (issue #282's own safety
    instructions: read real bytes, never assume a per-expert-varying quant
    type without checking): the real checkpoint's ``ffn_gate_exps``/
    ``ffn_up_exps`` are uniformly ``Q4_K`` across all 41 layers, but
    ``ffn_down_exps`` is NOT uniform -- 38 layers are ``Q5_K`` and 3 are
    ``Q6_K`` (llama.cpp's own "mixed" quantization heuristic keeps a few
    layers at higher precision). This is a real, per-LAYER varying
    ``ggml_type`` (not per-expert WITHIN one tensor -- GGUF's tensor-info
    section stores exactly one ``ggml_type`` per tensor, so a single
    ``ffn_*_exps`` tensor's experts always share one quant type), so
    ``ggml_type`` is yielded per ``(layer, bank_name)``, and
    :func:`freetoken.models.weight.stream_moe_expert_sources_gguf_kquant`
    (the caller that folds this generator's output into per-layer banks)
    handles the resulting per-layer row-byte-length mismatch.
    """
    import numpy as _np
    import torch as _torch

    gguf_file = load_gguf(model_path)
    arch = gguf_file.metadata.get("general.architecture")
    num_experts = _arch_get(gguf_file.metadata, arch, "expert_count")
    if not num_experts:
        return  # not a MoE checkpoint -- nothing to yield
    num_experts = int(num_experts)
    block_count = _arch_get(gguf_file.metadata, arch, "block_count")
    real_num_layers = (
        _gguf_real_num_layers(gguf_file.metadata, arch, int(block_count)) if block_count else None
    )

    by_layer: Dict[int, Dict[str, GGUFTensorInfo]] = {}
    wanted = {
        "ffn_gate_exps.weight": "gate",
        "ffn_up_exps.weight": "up",
        "ffn_down_exps.weight": "down",
    }
    for name, info in gguf_file.tensors.items():
        m = _LAYER_RE.match(name)
        if not m:
            continue
        layer_idx = int(m.group(1))
        # Issue #287: exclude a trailing MTP block's routed-expert-shaped
        # tensors too -- see _gguf_real_num_layers's own docstring.
        if real_num_layers is not None and layer_idx >= real_num_layers:
            continue
        suffix = m.group(2)
        if suffix in wanted:
            by_layer.setdefault(layer_idx, {})[suffix] = info

    with open(gguf_file.path, "rb") as fh:
        for layer in sorted(by_layer):
            suffixes = by_layer[layer]
            for gguf_suffix, bank_name in wanted.items():
                info = suffixes.get(gguf_suffix)
                if info is None:
                    continue
                if info.shape[-1] != num_experts:
                    raise GGUFFormatError(
                        f"{info.name}: expected the expert axis (last ne-order dim) "
                        f"to be {num_experts}, got {info.shape[-1]}"
                    )
                n_elements_per_expert = info.n_elements // num_experts
                row_bytes = gguf_expert_row_bytes(info.ggml_type, n_elements_per_expert)
                n_bytes = row_bytes * num_experts
                fh.seek(info.offset)
                raw = fh.read(n_bytes)
                if len(raw) != n_bytes:
                    raise GGUFFormatError(
                        f"{info.name}: read {len(raw)} bytes, expected {n_bytes} "
                        "(truncated file?)"
                    )
                arr = _np.frombuffer(raw, dtype=_np.uint8).reshape(num_experts, row_bytes)
                tensor = _torch.from_numpy(arr.copy())  # owns its memory, independent of `raw`
                yield layer, bank_name, tensor, info.ggml_type


class GgufModel:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(self, *args, **kwargs):
        unimplemented("GgufModel.forward", "models-gguf-iter-and-loader-wiring")
