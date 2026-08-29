"""Model adapter: Qwen3.5 / Qwen3.6 hybrid linear-attention + MoE (``qwen3_5_moe``).

The hero model for the Intel Arc Pro B70 target (issue #18). Qwen3.5/3.6 is a
*hybrid-attention, multimodal MoE*: most layers are **linear attention**
(Gated-DeltaNet -- a cheap, recurrent, O(1)-per-token state) with a few
**full-attention** (gated, GQA) layers interleaved, on top of a 256-way
top-8 MoE with an always-on shared expert. The language tower lives under
``text_config`` / ``model.language_model.*`` (a vision tower ``model.visual.*``
sits alongside it and is out of scope for text serving).

This module owns the two *checkpoint adapters* the loader calls:

* :func:`parse_config` -- torch-free; reads the (nested, multimodal) HF
  ``config.json`` and produces a :class:`freetoken.models.config.ModelConfig`
  with the MoE plumbing the bank fabricator reads (``is_moe``, ``num_experts``,
  ``moe_intermediate_size``, ``num_moe_layers``), stowing the full raw
  ``text_config`` in ``config.attrs`` for the forward pass.
* :func:`iter_weights` -- torch-gated; drops the vision tower, remaps the
  ``model.language_model.*`` prefix to ``model.*`` (so the loader's MoE-bank
  plumbing resolves the keys), and routes the routed experts to host (offload
  banks) and everything else to the dense device.

The forward pass (linear-attention recurrent state + full attention + MoE) is a
later issue; :class:`Qwen3_5MoEForCausalLM` is a forward-only stub for now.
Instantiating it is torch-gated (the constructor rebinds the instance to a real
``nn.Module``), but *importing* this module never requires torch.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from freetoken._stub import NotYetImplemented
from freetoken.models.config import ModelConfig

__all__ = ["parse_config", "iter_weights", "Qwen3_5MoEForCausalLM"]

# The text tower's prefix inside the multimodal checkpoint; remapped to
# ``model.*`` so the loader's MoE-bank plumbing (and the forward pass) sees
# plain ``model.*`` keys, exactly like the dense qwen3_moe model.
_LANGUAGE_PREFIX = "model.language_model."
_LANGUAGE_TARGET = "model."
_VISUAL_PREFIX = "model.visual."

# The one checkpoint weight that the routed-expert bank fabricator expects to be
# an *untransposed* [n_experts, hidden, inter] (see qwen3_moe._transpose_down):
# the down proj of a routed expert. The shared expert's down proj is a *dense*
# weight (it stays on the device, not in the host banks), so it keeps the
# transposed [hidden, inter] layout and must NOT be caught by the key heuristic.
# Matching on the key (not the shape) keeps the dense shared expert out of the
# bank transpose even when its shape would otherwise look like an expert.
_MOE_ROUTED_DOWN_KEY = "mlp.experts.down_proj"


def _expert_source_names(cfg: ModelConfig) -> set:
    """The (remapped) key suffixes that are routed-expert weights: these go to
    the host offload banks, everything else stays dense on the device.

    Mirrors qwen3_moe's ``_is_expert_key`` so the loader's MoE-bank plumbing
    (``load_moe_expert_sources`` / ``stream_moe_expert_sources``) can resolve the
    experts out of the remapped ``model.*`` keys. A routed-expert weight carries
    a ``.experts.`` segment; the dense shared expert (``mlp.shared_expert.*``)
    does not, so it stays on the device.
    """
    names = set()
    if cfg.is_moe:
        for i in range(cfg.first_k_dense_replace, cfg.num_moe_layers + cfg.first_k_dense_replace):
            mlp = f"model.layers.{i}.mlp.experts"
            for suffix in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
                names.add(f"{mlp}.{suffix}")
            # The real (packed) Qwen3.5/3.6 layout also uses the fused / un-suffixed
            # names; accept both so a checkpoint in either spelling routes correctly.
            names.add(f"{mlp}.gate_up_proj")
            names.add(f"{mlp}.down_proj")
    return names


def _first(d: dict, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return default


def _rope_theta(text_config: dict) -> float:
    """Qwen's RoPE theta: it may live under ``rope_parameters.rope_theta`` (the
    Qwen3.6 spelling) or the flat ``rope_theta``. Returns the resolved scalar
    (the nested ``rope_parameters`` -- partial_rotary_factor etc. -- stays
    reachable through ``attrs['text_config']`` for the partial-rotary
    full-attention projection in the forward pass)."""
    theta = _first(text_config, "rope_theta", default=None)
    if theta is None:
        params = text_config.get("rope_parameters")
        if isinstance(params, dict):
            theta = params.get("rope_theta")
    return float(theta) if theta is not None else 10000.0


def parse_config(hf_config: Any, *, use_offload_moe: bool = False) -> ModelConfig:
    """Build a :class:`ModelConfig` from a (multimodal) Qwen3.5/3.6 HF config.

    Torch-free. The language tower is nested under ``text_config`` (the config is
    multimodal); when the given config is already the flat text object (no
    ``text_config`` sub-dict), its own fields are used instead. The MoE fields
    the bank fabricator reads (``num_experts`` -- stored under ``num_experts``
    here, sometimes ``num_local_experts`` -- plus ``moe_intermediate_size``)
    are set first-class, and ``is_moe``/``num_moe_layers`` are derived so the
    loader routes the routed experts to host. The full raw ``text_config`` is
    stowed in ``config.attrs`` for the forward pass (it needs ``layer_types``,
    ``partial_rotary_factor``, the output-gate flag, and the linear-attention
    head dims, none of which are first-class ``ModelConfig`` fields).
    """
    raw = hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(hf_config)
    # Multimodal config: the language tower is nested under text_config. If the
    # config handed in is already the flat text object, use its own fields.
    text_config = raw.get("text_config")
    if not isinstance(text_config, dict):
        text_config = raw

    hidden = int(_first(text_config, "hidden_size", default=0))
    num_layers = int(_first(text_config, "num_hidden_layers", default=0))
    num_attention_heads = int(_first(text_config, "num_attention_heads", default=0))
    num_key_value_heads = int(
        _first(text_config, "num_key_value_heads", "num_kv_heads", default=0)
    )
    vocab_size = int(_first(text_config, "vocab_size", default=0))
    max_position_embeddings = int(
        _first(text_config, "max_position_embeddings", default=0)
    )
    # Qwen3.5/3.6 stores the expert count under num_experts; some transformers
    # configs spell it num_local_experts. Prefer the explicit one, fall back to
    # the other. (The bank fabricator reads cfg.num_experts.)
    num_experts = int(
        _first(text_config, "num_local_experts", "num_experts", default=0)
    )
    moe_intermediate_size = int(
        _first(
            text_config,
            "moe_intermediate_size",
            "intermediate_size",
            default=0,
        )
    )
    first_k_dense_replace = int(_first(text_config, "first_k_dense_replace", default=0))
    head_dim = int(_first(text_config, "head_dim", default=0))

    cfg = ModelConfig(
        architectures=list(_first(raw, "architectures", default=[]) or []),
        hidden_size=hidden,
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        moe_intermediate_size=moe_intermediate_size,
        first_k_dense_replace=first_k_dense_replace,
    )
    # The MoE plumbing: set is_moe explicitly (the text_config is a separate
    # dict from the top-level raw, so ModelConfig.__post_init__ can't infer it),
    # and re-derive num_moe_layers (post_init only sets it when is_moe was true
    # at construction). All Qwen3.5/3.6 layers are MoE (no dense prefix).
    cfg.is_moe = bool(num_experts > 0)
    if cfg.is_moe:
        cfg.num_experts = num_experts
        cfg.num_experts_per_tok = int(_first(text_config, "num_experts_per_tok", default=0))
        cfg.num_moe_layers = num_layers - first_k_dense_replace
    cfg.use_offload_moe = bool(use_offload_moe)

    # rope_theta is a first-class ModelConfig field (the rope builder reads it);
    # resolve it (it may live under rope_parameters.rope_theta) and set the field.
    cfg.rope_theta = _rope_theta(text_config)

    # The forward pass reads the full raw text tower (layer_types,
    # partial_rotary_factor, attn_output_gate, the linear-attention dims, ...),
    # none of which are first-class ModelConfig fields.
    attrs: Dict[str, Any] = {"text_config": dict(text_config), "head_dim": head_dim}
    cfg.attrs.update(attrs)
    return cfg


def iter_weights(
    model_path: str,
    device: Any,
    *,
    include_moe_experts: bool = True,
    include_non_moe: bool = True,
    dtype: Any = None,
) -> Any:
    """Stream Qwen3.5/3.6 weights from ``model_path`` as ``(name, tensor)``.

    torch-gated (imports ``torch`` + the safetensors streamer lazily inside the
    generator body, so importing this module stays torch-free). The checkpoint
    is multimodal:

    * ``model.visual.*`` (the vision tower) is **dropped** -- text serving does
      not run the image encoder.
    * ``model.language_model.*`` is **remapped** to ``model.*`` so the loader's
      MoE-bank plumbing (and the forward pass) see the same key shape as the
      dense qwen3_moe model.
    * Routed experts (``.experts.*``) go to **host** (offload banks); everything
      else -- including the always-on shared expert and the linear-attention
      weights -- goes to the dense ``device``.

    ``include_moe_experts`` / ``include_non_moe`` follow the loader contract:
    the MoE-bank path passes ``include_non_moe=False`` (experts only); the dense
    placement passes ``include_moe_experts=False`` (everything else, including
    the shared expert).
    """
    # Lazy torch import: keeps this module importable on a torch-free box.
    import torch

    from freetoken.models.weight import iter_safetensors

    # Routed-expert key set (remapped names) for the routing decision. The loader
    # always has a resolved config (it parses the checkpoint before calling
    # iter_weights), but fall back to the substring heuristic if not.
    cfg = _cfg_for_path(model_path)
    routed = _expert_source_names(cfg) if cfg is not None else None

    for raw_name, tensor in iter_safetensors(model_path, device=device):
        # Drop the vision tower outright (text serving does not run the
        # image encoder).
        if raw_name.startswith(_VISUAL_PREFIX):
            continue
        # Remap the language tower to the plain model.* prefix so the loader's
        # MoE-bank plumbing (and the forward pass) see the same key shape as the
        # dense qwen3_moe model.
        name = (
            _LANGUAGE_TARGET + raw_name[len(_LANGUAGE_PREFIX):]
            if raw_name.startswith(_LANGUAGE_PREFIX)
            else raw_name
        )
        if routed is not None:
            expert = name in routed
        else:
            expert = ".experts." in name
        # The loader's contract: include_moe_experts=False (dense placement)
        # drops the routed experts; include_non_moe=False (bank fabricator)
        # keeps only the routed experts.
        if not include_moe_experts and expert:
            continue
        if not include_non_moe and not expert:
            continue
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        # Routed experts stream to host (offload banks); the rest (dense: the
        # shared expert, linear-attention weights, embeddings, norms, lm_head)
        # goes to the dense device. The loader banks the experts and moves them
        # to the xpu per-token at decode time.
        if expert and tensor.device.type != "cpu":
            tensor = tensor.to("cpu")
        yield name, tensor


def _cfg_for_path(model_path: str) -> Optional[ModelConfig]:
    """Best-effort resolve of the ModelConfig for ``model_path`` (to build the
    exact routed-expert key set). Returns None when the path has no readable
    config (the iterator then falls back to the substring heuristic)."""
    try:
        from freetoken.utils.hf import cached_load_hf_config

        return parse_config(cached_load_hf_config(model_path))
    except Exception:
        return None


class Qwen3_5MoEForCausalLM:
    """Hybrid linear-attention + MoE text model.

    **Phase 1 (this issue):** a forward-only stub. The constructor rebinds the
    *instance* to a real ``torch.nn.Module`` (so the loader's
    ``named_parameters()`` / bookkeeping runs) while keeping the class object a
    plain object -- that way this module imports without torch and only the
    instantiation (the torch loader/engine path) needs it. ``forward`` fails
    loud (NotImplementedError) until the hybrid forward (linear-attention
    recurrent state + gated full attention + MoE with shared expert + the
    engine's recurrent-state decode path) lands in a later issue.
    """

    def __init__(self, config, device=None) -> None:
        import torch

        # Make *this instance* a real nn.Module: rebind the instance's
        # __class__ to nn.Module FIRST, then run the module bookkeeping. torch's
        # Module.__init__ only executes the init body when the instance's class
        # resolves to nn.Module, so the rebind has to precede the call. The
        # class *object* keeps its plain base, so importing this module never
        # needs torch -- only instantiating it (the torch loader/engine path)
        # does.
        self.__class__ = torch.nn.Module
        torch.nn.Module.__init__(self)
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "device", device)

    def forward(self, *args, **kwargs):
        raise NotYetImplemented(
            "Qwen3.5/3.6 (qwen3_5_moe) forward pass",
            "models-qwen35",
            "The hybrid linear-attention + MoE forward (Gated-DeltaNet recurrent "
            "state + gated full attention + 256-way shared-expert MoE + the "
            "engine's recurrent-state decode path) is a later issue; this adapter "
            "currently wires config/weights only (issue #18, phase 1).",
        )
