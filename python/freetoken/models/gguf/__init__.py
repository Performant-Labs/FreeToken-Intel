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
# k_norm, unlike qwen35moe's fused attn_qkv + Gated-Delta-Net layers,
# which are NOT covered here: qwen35moe is excluded from the hybrid
# split already (see qwen3_5_moe._forward_hybrid's "gguf" exclusion)
# and its GDN tensor wiring (ssm_*, attn_gate, shared-expert ffn_*_shexp)
# is real, separable follow-up work, not in this issue's scope).
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
    unchanged, with no new bank-building logic needed.
    """
    import torch as _torch

    from .dequant import dequantize as _dequantize

    gguf_file = load_gguf(model_path)

    by_layer: Dict[int, Dict[str, GGUFTensorInfo]] = {}
    top_level: Dict[str, GGUFTensorInfo] = {}
    for name, info in gguf_file.tensors.items():
        m = _LAYER_RE.match(name)
        if m:
            by_layer.setdefault(int(m.group(1)), {})[m.group(2)] = info
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
                for gguf_suffix, hf_suffix in _DENSE_SUFFIX_MAP.items():
                    info = suffixes.get(gguf_suffix)
                    if info is not None:
                        yield f"{prefix}.{hf_suffix}", _read_and_dequant(info).to(device)
            if include_moe_experts:
                gate = suffixes.get("ffn_gate_exps.weight")
                up = suffixes.get("ffn_up_exps.weight")
                down = suffixes.get("ffn_down_exps.weight")
                if gate is not None and up is not None:
                    gate_up = _torch.cat([_read_and_dequant(gate), _read_and_dequant(up)], dim=1)
                    yield f"{prefix}.mlp.experts.gate_up_proj", gate_up.to("cpu")
                if down is not None:
                    yield f"{prefix}.mlp.experts.down_proj", _read_and_dequant(down).to("cpu")


class GgufModel:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(self, *args, **kwargs):
        unimplemented("GgufModel.forward", "models-gguf-iter-and-loader-wiring")
