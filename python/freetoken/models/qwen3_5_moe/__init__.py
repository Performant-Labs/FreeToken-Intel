"""Model adapter: Qwen3.5 / Qwen3.6 hybrid linear-attention + MoE (``qwen3_5_moe``).

The hero model for the Intel Arc Pro B70 target (issue #18). Qwen3.5/3.6 is a
*hybrid-attention, multimodal MoE*: most layers are **linear attention**
(Gated-DeltaNet -- a cheap, recurrent, O(1)-per-token state) with a few
**full-attention** (gated, GQA) layers interleaved, on top of a 256-way
top-8 MoE with an always-on shared expert. The language tower lives under
``text_config`` / ``model.language_model.*`` (a vision tower ``model.visual.*``
sits alongside it and is out of scope for text serving).

This module owns the two *checkpoint adapters* the loader calls
(:func:`parse_config`, :func:`iter_weights`) **and** the real forward pass
(:class:`Qwen3_5MoEForCausalLM`). The forward is a genuine ``nn.Module`` with
four building blocks:

* :class:`_GatedDeltaNet` -- the linear-attention layer. A causal depthwise
  conv plus separate ``in_proj_qkv`` / ``in_proj_z`` / ``in_proj_b`` /
  ``in_proj_a`` projections feed a *recurrent* Gated-Delta-Net update whose
  per-request state (the ``[S, num_v, key_dim, value_dim]`` matrix plus a conv
  ring buffer) lives in the linear-state pool the model owns. Both the prefill
  and decode phases run the recurrent update (it is exact, and its chunked
  sibling is bit-identical to it), so one code path serves the whole step.
* :class:`_Qwen35Attention` -- the full-attention layer. ``q_proj`` projects to
  ``num_heads * head_dim * 2`` (query + an output *gate*); ``q_norm`` /
  ``k_norm`` RMS-norm the head, partial RoPE rotates a configurable fraction of
  the head, the keys/values are appended to the paged KV pool, and the reference
  attention backend reads the full history back. The output is gated by
  ``sigmoid(gate)`` before ``o_proj``.
* :class:`_Qwen35MoE` -- the 256-way router (top-8) plus the always-on shared
  expert, with the in-VRAM / host-offload split from the qwen3_moe adapter.
* :class:`Qwen3_5MoEForCausalLM` -- embeddings + the 40 layers + final norm +
  ``lm_head``. It owns the linear-state pool (recurrent + conv states, lazily
  grown per request slot) and, per step, runs each request's token slice through
  all layers and keeps the last position's logits.

Importing this module never requires torch (the adapter half is torch-free); only
instantiating :class:`Qwen3_5MoEForCausalLM` does.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Dict, Optional

from freetoken.models.config import ModelConfig

__all__ = ["parse_config", "iter_weights", "Qwen3_5MoEForCausalLM"]

# The text tower's prefix inside the multimodal checkpoint; remapped to
# ``model.*`` so the loader's MoE-bank plumbing (and the forward pass) sees
# plain ``model.*`` keys, exactly like the dense qwen3_moe model.
_LANGUAGE_PREFIX = "model.language_model."
_LANGUAGE_TARGET = "model."
_VISUAL_PREFIX = "model.visual."
_MTP_PREFIX = "mtp."

# GPTQ's four per-projection component tensors (AutoGPTQ layout, see
# freetoken.kernel.triton.gptq_linear). A checkpoint with no GPTQ tensors at
# all (the common case) never populates the buffer these key off of, so
# _dequantize_gptq_stream is a no-op passthrough for a plain bf16 checkpoint.
_GPTQ_SUFFIXES = (".qweight", ".qzeros", ".scales", ".g_idx")

# Component suffixes unique to the other three packed quant formats' own
# per-expert component tensors (block-FP8's weight_scale_inv, issue #152;
# compressed-tensors INT8's weight_packed/weight_scale/weight_shape, issue
# #154) -- unlike GPTQ's four, these formats' own ".weight" component is
# NOT included here (it correctly matches _expert_source_names' plain
# ".weight"-suffixed routed set already, see is_quant_component's own
# comment below for why only the suffixes _expert_source_names has no
# entry for need this bypass).
_OTHER_QUANT_SUFFIXES = (".weight_scale_inv", ".weight_packed", ".weight_scale", ".weight_shape")
# MXFP4's own component "suffix" is fused onto the projection name with an
# underscore, not a dot (e.g. "...gate_up_proj_blocks"), issue #153's real
# checkpoint spelling -- see _parse_mxfp4_expert_key's own docstring.
_MXFP4_SUFFIXES = ("_blocks", "_scales")

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

    A routed-expert weight carries a ``.experts.`` segment; the dense shared
    expert (``mlp.shared_expert.*``) does not, so it stays on the device.

    Real checkpoint files always store one tensor **per expert** (``model.
    layers.{i}.mlp.experts.{e}.gate_proj.weight``, not a single fused
    ``experts.gate_proj.weight`` with no expert index) -- this previously
    generated only the un-indexed spelling, which never matches any real
    checkpoint's keys, so every routed expert silently fell through to "not
    an expert" and got dropped. Found via a real (if tiny) non-GPTQ
    checkpoint: the GPTQ bank-only path never exercises this function (it
    classifies experts by a ``.experts.`` substring check instead, see
    ``iter_weights``), so this bug was invisible on every GPTQ checkpoint
    tested so far.
    """
    names = set()
    if cfg.is_moe:
        for i in range(cfg.first_k_dense_replace, cfg.num_moe_layers + cfg.first_k_dense_replace):
            mlp = f"model.layers.{i}.mlp.experts"
            for e in range(cfg.num_experts or 0):
                for suffix in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
                    names.add(f"{mlp}.{e}.{suffix}")
            # The fused / un-suffixed names (a single [E, ...] tensor per layer,
            # no per-expert index) also route correctly if a checkpoint ever
            # ships that spelling instead.
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


def parse_config(
    hf_config: Any,
    *,
    use_offload_moe: bool = False,
    use_cpu_moe: bool = False,
    use_hybrid: bool = False,
    moe_cpu_layers: str | None = None,
    model_path: str | None = None,
) -> ModelConfig:
    """Build a :class:`ModelConfig` from a (multimodal) Qwen3.5/3.6 HF config.

    ``model_path`` is accepted (and unused) to match ``models/loader.py``'s
    ``load_model``, which calls every architecture's ``parse_config`` with
    ``model_path=model_path`` unconditionally (``qwen3_moe``'s sibling uses
    it to probe ``head_dim`` from the checkpoint when the config omits it;
    this family's config always carries ``head_dim`` explicitly, so there is
    nothing to probe) -- without this parameter every qwen3_5/3.6 checkpoint
    failed to load at all with ``TypeError: parse_config() got an unexpected
    keyword argument 'model_path'``.

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
        head_dim=head_dim,
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
    # The CPU backend (issue #8) also keeps experts non-device-resident, so it
    # flags use_offload_moe as well (the model's moe_offload gate + the loader's
    # host-bank build key off it); use_cpu_moe is the *distinct* flag the block
    # reads to run the routed-expert GEMM on the host instead of the LRU slots.
    cfg.use_cpu_moe = bool(use_cpu_moe)
    # Issue #9 (moe-hybrid): the per-step split also keeps experts non-device-
    # resident (so use_offload_moe, set above, is the gate); use_hybrid is the
    # distinct flag the block reads to split each decode step's misses between
    # the PCIe-fetch (XPU) and host-CPU halves by the profile's fetch fraction.
    cfg.use_hybrid = bool(use_hybrid)
    # Issue #8: store the --moe-cpu-layers spec verbatim (resolved at build time).
    cfg.moe_cpu_layers = moe_cpu_layers

    # rope_theta is a first-class ModelConfig field (the rope builder reads it);
    # resolve it (it may live under rope_parameters.rope_theta) and set the field.
    rope_theta = _rope_theta(text_config)
    cfg.rope_theta = rope_theta

    # The forward pass reads the full raw text tower (layer_types,
    # partial_rotary_factor, attn_output_gate, the linear-attention dims, ...),
    # none of which are first-class ModelConfig fields. The raw text_config
    # usually has NO flat rope_theta (it is nested under rope_parameters), so the
    # resolved value is also stashed here for the forward's RoPE builder.
    attrs: Dict[str, Any] = {
        "text_config": dict(text_config),
        "head_dim": head_dim,
        "rope_theta": rope_theta,
    }
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
    * ``mtp.*`` (the multi-token-prediction head, a separate top-level
      namespace not nested under ``model.language_model.*``) is **dropped** --
      this port's engine does not run MTP.
    * ``model.language_model.*`` is **remapped** to ``model.*`` so the loader's
      MoE-bank plumbing (and the forward pass) see the same key shape as the
      dense qwen3_moe model.
    * A **GPTQ-quantized** checkpoint (e.g. the official
      ``Qwen/Qwen3.5-35B-A3B-GPTQ-Int4``) stores each quantized projection as
      four tensors -- ``.qweight`` / ``.qzeros`` / ``.scales`` / ``.g_idx`` --
      instead of one plain ``.weight``. In **bank-only mode**
      (``include_moe_experts=True, include_non_moe=False`` -- the offload
      MoE-bank fabricator's call, ``load_moe_expert_sources``) these four
      pass through **raw and unmodified**, by design (issue `moe-quant-
      banks-pack`, #135): dequantizing every expert to bf16 at load time is
      a 4x expansion that blows the host RAM budget for a real-scale
      checkpoint (issue #134). A caller that wants packed GPTQ tensors
      turned into per-layer banks streams this generator's output through
      :func:`freetoken.models.weight.stream_moe_expert_sources_gptq`, which
      keeps them packed the whole way through -- dequantization happens
      lazily, per-expert, at compute time (issue #137), not here. In every
      **other** call shape (bank-only mode is off), the four components are
      still dequantized here via
      :func:`freetoken.kernel.triton.gptq_linear.dequantize_gptq_int4` into
      a single dense ``.weight`` tensor, matching a plain bf16 checkpoint's
      shape -- the dense-placement call never sees expert tensors at all
      (they are filtered out below), so this only actually matters for a
      hypothetical future non-offload (fully in-VRAM) consumer of this
      iterator; it is not exercised by the offload path once #136/#137 land.
      Per the real checkpoint's ``quantization_config.dynamic`` exclusions,
      only the routed experts' ``gate_proj``/``up_proj``/``down_proj`` are
      ever quantized this way; attention, the shared expert, embeddings, and
      ``lm_head`` stay plain bf16/fp16 tensors and pass through unchanged
      regardless of mode.
    * Routed experts (``.experts.*``) go to **host** (offload banks); everything
      else -- including the always-on shared expert and the linear-attention
      weights -- goes to the dense ``device``.

    ``include_moe_experts`` / ``include_non_moe`` follow the loader contract:
    the MoE-bank path passes ``include_non_moe=False`` (experts only); the dense
    placement passes ``include_moe_experts=False`` (everything else, including
    the shared expert).
    """
    # Lazy torch import: to keep this module importable on a torch-free box.
    _cfg = _cfg_for_path(model_path)
    routed = _expert_source_names(_cfg) if _cfg is not None else None

    # Bank-only mode (the offload-cache path, #135): do NOT eagerly dequantize
    # GPTQ-packed expert tensors -- pass them through raw so a packed-bank
    # builder can keep them packed in host RAM.
    #
    # The dense-placement call (include_moe_experts=False) never *yields* an
    # expert tensor (the `expert` filter below drops it) -- but a naive
    # `_dequantize_gptq_stream(raw_stream)` here would still eagerly
    # dequantize every routed expert's GPTQ tensors before that filter ever
    # runs, reintroducing the exact whole-checkpoint eager-dequant RAM blowup
    # issue #135 eliminated for the bank-fabricator call, just on this call
    # shape instead (found by issue moe-quant-banks-e2e (#138)'s end-to-end
    # test: a real load_model() call crashed trying to dequantize experts the
    # dense placement was always going to discard). Fix: drop routed-expert
    # GPTQ components from the stream *before* they ever reach the dequant
    # buffer, so they are never assembled/dequantized at all on this path.
    bank_only = include_moe_experts and not include_non_moe
    raw_stream = _iter_safetensors(model_path, device=device)
    if bank_only:
        stream = raw_stream
    elif not include_moe_experts:
        stream = _dequantize_gptq_stream(_drop_routed_expert_gptq_components(raw_stream))
    else:
        stream = _dequantize_gptq_stream(raw_stream)

    for raw_name, tensor in stream:
        # Drop the vision tower and the MTP head outright -- neither is part
        # of the text-serving forward pass this port runs.
        if raw_name.startswith(_VISUAL_PREFIX) or raw_name.startswith(_MTP_PREFIX):
            continue
        # Remap the language tower to the plain model.* prefix so the loader's
        # MoE-bank plumbing (and the forward pass) see the same key shape as the
        # dense qwen3_moe model.
        name = (
            _LANGUAGE_TARGET + raw_name[len(_LANGUAGE_PREFIX):]
            if raw_name.startswith(_LANGUAGE_PREFIX)
            else raw_name
        )
        is_gptq_component = bank_only and name.endswith(_GPTQ_SUFFIXES)
        # A raw packed-quant component's name -- ANY of the four formats'
        # own component suffixes, not just GPTQ's -- is never in `routed`
        # (:func:`_expert_source_names` only ever generates the plain
        # ".weight"-suffixed dense spelling, one entry per expert/proj; it
        # has no idea block-FP8 ships a second ".weight_scale_inv" tensor,
        # INT8 ships three differently-named ones, or MXFP4 fuses its
        # component onto the projection name with an underscore instead of
        # a dot). Found the same way the GPTQ case originally was: a real
        # (if tiny) non-GPTQ quantized checkpoint's forward silently lost
        # its scale tensors here, producing incomplete banks with an opaque
        # "missing" error several layers away from this actual cause. Each
        # format's own ".weight" component (present for GPTQ-less formats
        # too) is NOT in this bypass -- it already matches `routed`'s plain
        # spelling correctly, so only the suffixes `routed` has no entry
        # for at all need the same ".experts."-substring classification
        # GPTQ's own components already use.
        is_quant_component = bank_only and (
            is_gptq_component
            or name.endswith(_OTHER_QUANT_SUFFIXES)
            or name.endswith(_MXFP4_SUFFIXES)
        )
        if is_quant_component:
            # Real routed-expert keys always carry ".experts." (the shared
            # expert does not: it is "mlp.shared_expert.*", a different
            # substring), so this is exact, not a heuristic weakening --
            # the same reasoning the no-config fallback below already
            # relies on.
            expert = ".experts." in name
        elif routed is not None:
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
        # Never dtype-cast a raw packed-quant component: GPTQ's
        # qweight/qzeros/g_idx and INT8's weight_packed/weight_shape are
        # int32/int64, MXFP4's blocks/scales are uint8 -- casting any of
        # these to a requested bf16/fp16 dense dtype would silently
        # corrupt the packed bits, not "convert" them (a real numeric
        # dtype like FP8's own weight/weight_scale_inv would merely be
        # reinterpreted, not corrupted, by such a cast, but is still
        # excluded here for the same "never touch a packed component's
        # dtype before the format-specific streamer parses it" discipline).
        if dtype is not None and not is_quant_component and tensor.dtype != dtype:
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


def _iter_safetensors(model_path, device):
    from freetoken.models.weight import iter_safetensors

    yield from iter_safetensors(model_path, device=device)


def _drop_routed_expert_gptq_components(pairs):
    """Filter a raw ``(name, tensor)`` stream, dropping routed-expert GPTQ
    components (``.qweight``/``.qzeros``/``.scales``/``.g_idx``) outright --
    for the dense-placement call, these are discarded by the main loop's
    ``expert`` filter anyway, so this stops them from ever reaching
    :func:`_dequantize_gptq_stream`'s buffer-and-dequantize logic (see the
    call site's comment, issue #138). Everything else (including a
    GPTQ-quantized *non*-expert tensor, if a future checkpoint ever has one --
    none does today, per the real checkpoint's ``quantization_config.dynamic``
    exclusions) passes through unchanged, still eligible for the normal
    eager-dequant path.
    """
    for name, tensor in pairs:
        if name.endswith(_GPTQ_SUFFIXES) and ".experts." in name:
            continue
        yield name, tensor


def _dequantize_gptq_stream(pairs):
    """Wrap a raw ``(name, tensor)`` stream, buffering and dequantizing any
    GPTQ-packed projections (four tensors -- ``.qweight``/``.qzeros``/
    ``.scales``/``.g_idx`` -- sharing one name prefix) into a single dense
    ``<prefix>.weight`` tensor, matching the shape a plain bf16 checkpoint's
    stream already has. Every other tensor passes through unchanged, so this
    is a no-op for a checkpoint with no GPTQ tensors at all.

    Buffers by prefix rather than assuming the four components arrive
    consecutively (safetensors preserves each shard's own key order, but
    that order is not a documented guarantee, and different components of
    one projection could in principle live in different shards).
    """
    import torch

    from freetoken.kernel.triton.gptq_linear import dequantize_gptq_int4

    pending: Dict[str, Dict[str, Any]] = {}
    for name, tensor in pairs:
        matched_suffix = next((s for s in _GPTQ_SUFFIXES if name.endswith(s)), None)
        if matched_suffix is None:
            yield name, tensor
            continue
        prefix = name[: -len(matched_suffix)]
        parts = pending.setdefault(prefix, {})
        parts[matched_suffix] = tensor
        if len(parts) < len(_GPTQ_SUFFIXES):
            continue
        del pending[prefix]
        # dequantize_gptq_int4 returns [in_features, out_features]; nn.Linear's
        # weight convention (what every non-quantized tensor in this stream
        # already is) is [out_features, in_features].
        dense = dequantize_gptq_int4(
            parts[".qweight"], parts[".qzeros"], parts[".scales"], parts[".g_idx"], out_dtype=torch.bfloat16
        ).T.contiguous()
        yield prefix + ".weight", dense
    if pending:
        incomplete = {prefix: sorted(parts) for prefix, parts in pending.items()}
        raise ValueError(f"GPTQ tensor stream ended with incomplete projection(s): {incomplete}")


# Forward side (the real hybrid model the engine runs, #18 / #14)
# --------------------------------------------------------------------------- #
#
# Torch is imported LAZILY here (not at module scope) so the torch-free CPU
# venv can still import this module and call ``parse_config`` / ``iter_weights``
# without torch installed. When the model is *instantiated* the constructor
# calls ``_ensure_torch()`` (see the ``Qwen3_5MoEForCausalLM.__init__`` below),
# which imports torch and rebinds the forward-side classes to real ``nn.Module``
# subclasses.
#
# The forward-side classes are declared as *plain* (baseless) classes in this
# module so the module imports without torch. ``_ensure_torch`` then, for each,
# creates a real ``nn.Module`` subclass by **subclassing** the plain class
# (``type(name, (nn.Module, Plain), {"__module__": ...})``) -- NOT by re-exec'ing
# the class source. Subclassing preserves the original method objects, whose
# zero-arg ``super()`` / ``__class__`` cells already refer to the plain class;
# because the new class is ``type(nn.Module, Plain)`` and the plain class is a
# direct child of ``object``, the MRO is [New, nn.Module, Plain, object], so
# ``super().__init__()`` in the inherited ``__init__`` resolves to
# ``nn.Module.__init__`` and the instance is a genuine ``nn.Module``. (The
# earlier ``exec``-and-splice approach failed for exactly this reason: re-exec'ing
# the source re-bound the ``__class__`` cell to whatever the exec namespace held
# at that instant, so the top-level class's ``__init__`` could bind to the plain
# class and the instance never got ``nn.Module.__init__`` run.)

def _ensure_torch() -> None:
    """Import torch (once) and rebind the forward-side classes to ``nn.Module``
    subclasses. Idempotent -- a no-op after the first call, so repeated
    instantiation is cheap. Must be called from the public constructor (or
    ``load_model``), never at module scope (that would require torch on a
    torch-free box).
    """
    if "torch" in globals():
        return

    import torch  # noqa: F811
    import torch.nn as nn  # noqa: F811
    import torch.nn.functional as F  # noqa: F811

    # Bind into the module globals so the (plain) class bodies -- which reference
    # ``torch`` / ``nn`` / ``F`` at call time via this module's globals -- resolve
    # to the real torch modules once instances are created.
    g = globals()
    g["torch"] = torch
    g["nn"] = nn
    g["F"] = F

    # The forward-side module-level helpers the class bodies call at runtime.
    # They are defined at module scope (torch-free to *define*; torch is only
    # needed when they are *called*), so they are already in ``g`` -- no rebinding
    # needed, they simply resolve through the module globals once torch is bound.
    # Only the *classes* need rebinding (their base must be nn.Module).

    # The forward-side classes, in dependency order (a class that instantiates an
    # earlier one -- e.g. _Qwen35MoE instantiating _Qwen35Expert, the decoder
    # layer instantiating the attention / MoE blocks -- resolves the torch
    # version because each is rebound before the next is constructed).
    order = (
        "_RMSNorm",
        "_RMSNormGated",
        "_GatedDeltaNet",
        "_Qwen35Attention",
        "_Qwen35MoE",
        "_Qwen35Expert",
        "_Qwen35MxfpExpert",
        "_Qwen35Fp8Expert",
        "_Qwen35Int8Expert",
        "_Qwen35DecoderLayer",
        "Qwen3_5MoEForCausalLM",
    )
    for name in order:
        plain = g.get(name)
        if plain is None or plain.__mro__[1] is not object:
            # Already an nn.Module (idempotent re-call) or absent.
            continue

        # Capture the ORIGINAL plain __init__ function object now. The wrapper
        # closure must call THIS function, not re-dereference the (rebound)
        # module global, or a later instantiation of an earlier class would
        # invoke the NEW class's wrapper __init__ -> infinite recursion.
        plain_init = plain.__dict__["__init__"]

        def make_init(plain_init_func):
            def __init__(self, *args, _device=None, **kwargs):
                # Always initialise nn.Module state before the plain body: the
                # plain __init__'s zero-arg super() has a __class__ cell bound
                # to the PLAIN class (MRO [plain, object]), so its
                # super().__init__() resolves to object.__init__ -- NOT to
                # nn.Module.__init__ -- even though the instance's actual class
                # is the new one. Running nn.Module.__init__ here (idempotent:
                # it just re-creates the _parameters / _modules dicts) guarantees
                # the state exists before the plain body assigns nn.Parameters.
                # The `device` kwarg (passed by the loader to the top-level model
                # only) is consumed here; nn.Module.__init__ takes no kwargs and
                # the building-block plain __init__s don't have a `device` kwarg,
                # so it is dropped.
                del _device
                nn.Module.__init__(self)
                plain_init_func(self, *args, **kwargs)

            return __init__

        # nn.Module is listed FIRST in the new MRO, so its own ``forward``
        # (and any other method nn.Module defines) would SHADOW a same-named
        # method inherited from the plain class: attribute lookup walks
        # ``type(self).__dict__ -> nn.Module.__dict__ (has forward) -> ...``
        # and stops at nn.Module, never reaching the plain class. Copy every
        # plain-class method (besides the wrapped ``__init__``) -- in
        # particular the top-level ``forward`` -- into the new class's own
        # ``__dict__`` so the rebind is identity-preserving and the real
        # forward is the one that runs.
        # Every method the plain class defines (``forward`` and the private
        # helpers), except ``__init__`` (already wrapped above) and class
        # descriptors (``__qualname__``, ``__doc__`` -- not callable here).
        # Copying them into the new class's ``__dict__`` is what makes the
        # rebind identity-preserving (see the note above).
        extra = {
            attr: val
            for attr, val in plain.__dict__.items()
            if attr != "__init__" and callable(val)
        }
        extra.update({"__module__": __name__, "__init__": make_init(plain_init)})
        torch_cls = type(name, (nn.Module, plain), extra)
        g[name] = torch_cls


def _xpu_available() -> bool:
    try:
        return bool(torch.xpu.is_available())
    except Exception:
        return False


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2-norm along ``dim`` (FLA's convention, matching the reference)."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dim by 90 degrees in the *reference* (cat(freqs, freqs))
    layout: the first half is negated, the second half is kept -- ``[-x2, x1]``.
    (The Qwen3.5/3.6 ``apply_rotary_pos_emb`` uses exactly this; it is NOT the
    GLM ``[-x2, x1]``-vs-``[x2, -x1]`` variant, and the cos/sin it pairs with are
    ``cat(freqs, freqs)`` so the per-pair (x_i, x_{i+n}) rotation matches HF.)"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    """Partial-RoPE: rotate only the first ``cos.shape[-1]`` head dims (the
    ``partial_rotary_factor`` fraction), passing the rest through unchanged.
    ``cos``/``sin`` are ``[*, rotary_dim]`` (already indexed by position, built
    as ``cat(freqs, freqs)`` so ``rotary_dim == 2 * len(inv_freq)``). The
    rotation is done in float32 (cos/sin are fp32) and the result is cast back
    to the input dtype, so a bf16/fp16 ``q``/``k`` stays in its compute dtype.
    """
    out_dtype = q.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    q_embed = torch.cat([q_embed, q_pass.to(q_embed.dtype)], dim=-1).to(out_dtype)
    k_embed = torch.cat([k_embed, k_pass.to(k_embed.dtype)], dim=-1).to(out_dtype)
    return q_embed, k_embed


# Mirrors the reference ``Qwen3_5MoeRMSNorm`` exactly: the weight is
# *zero-initialised* and applied as ``(x * (1 + w))`` in float32 (NOT torch's
# ``nn.RMSNorm`` ones-init / ``x * w``), so the loaded weight (0) gives an
# identity norm and a trained weight adds a learned per-dim scale.
class _RMSNorm:
    def __init__(self, dim: int, eps: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        # device/dtype (when given) place the weight in the compute dtype so the
        # loaded weights (streamed in that dtype) copy in without a dtype mix.
        self.weight = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whereas Qwen3.5Moe is (x * w).to(float16).
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


# Mirrors the reference ``Qwen3_5MoeRMSNormGated`` (the Gated-Delta-Net output
# norm): the weight is *one*-initialised, the input is normed (float32) then
# gated by ``silu(gate)``.
class _RMSNormGated:
    def __init__(self, hidden_size: int, eps: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.variance_epsilon = eps
        self.activation = "silu"

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Norm before gate (the reference's order).
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


class _LinearStateSlot:
    """The per-request recurrent + conv state the linear layers read/write.

    ``state`` is ``[num_v, key_dim, value_dim]`` (the Gated-Delta-Net recurrent
    matrix); ``conv_state`` is ``[conv_dim, kernel-1]`` (the causal conv ring).
    A pool is one of these per request slot; the linear layers lazily grow the
    pool when a new slot index is seen.
    """

    __slots__ = ("state", "conv_state")

    def __init__(self, state: torch.Tensor, conv_state: torch.Tensor) -> None:
        self.state = state
        self.conv_state = conv_state


class _LinearStatePool:
    """A dict from linear-layer id to a list of per-request :class:`_LinearStateSlot`.

    The model owns this (it knows the layer count + the per-request shapes). The
    engine sets ``ctx.linear_state_pool`` to it and assigns each request a
    ``linear_slot_idx``; the linear layers lazily grow ``entries[slot]`` when a
    new slot is seen so a request admitted after construction is handled.
    """

    def __init__(self) -> None:
        self._layers: dict = {}

    def register(self, layer_id: int, num_slots: int, state_shape, conv_shape, device, dtype) -> None:
        self._layers[layer_id] = [
            _LinearStateSlot(
                torch.zeros(state_shape, device=device, dtype=dtype),
                torch.zeros(conv_shape, device=device, dtype=dtype),
            )
            for _ in range(num_slots)
        ]

    def get(self, layer_id: int, slot: int) -> "_LinearStateSlot":
        entries = self._layers.get(layer_id)
        if entries is None:
            raise KeyError(f"linear-state pool: layer {layer_id} not registered")
        while slot >= len(entries):
            # Lazily grow for a request admitted after the pool was sized.
            prev = entries[-1]
            entries.append(
                _LinearStateSlot(
                    torch.zeros_like(prev.state),
                    torch.zeros_like(prev.conv_state),
                )
            )
        return entries[slot]

    def __contains__(self, layer_id) -> bool:
        return layer_id in self._layers

    def __getitem__(self, layer_id):
        return self._layers[layer_id]

    def num_slots(self, layer_id: int) -> int:
        return len(self._layers[layer_id])


def _pool_slot(pool, layer_id: int, slot_idx: int) -> "_LinearStateSlot | None":
    """Duck-type a per-request state slot out of either pool shape.

    ``pool`` is either this model's own :class:`_LinearStatePool` (a plain
    dict of pre-allocated ``_LinearStateSlot``s, the default -- unchanged
    behavior for every non-hybrid-cache-managed run) or a real
    :class:`freetoken.kvcache.linear_state_pool.LinearStatePool` (the
    ping-pong/COW-capable stacked-tensor pool, issue `semantic-cache-e2e`
    #172 -- the engine assigns this to ``ctx.linear_state_pool`` only when
    prefix caching + a hybrid model are both on). Wrapping the new pool's
    per-slot tensor VIEWS (not copies) in a ``_LinearStateSlot`` lets
    ``_conv``/``_delta_rule`` read/write either pool identically -- their
    ``.copy_()`` calls land directly on the new pool's own backing storage.
    """
    if pool is None:
        return None
    if hasattr(pool, "get"):  # this model's own _LinearStatePool
        if layer_id not in pool:
            return None
        return pool.get(layer_id, slot_idx)
    if pool.is_linear_layer(layer_id):  # freetoken.kvcache.linear_state_pool.LinearStatePool
        return _LinearStateSlot(pool.recurrent_state(layer_id, slot_idx), pool.conv_state(layer_id, slot_idx))
    return None


class _GatedDeltaNet:
    """A Qwen3.5/3.6 linear-attention (Gated-Delta-Net) layer.

    Separate projections (the qwen3_5_moe spelling -- NOT qwen3_next's fused
    ``in_proj_qkvz``/``in_proj_ba``): ``in_proj_qkv`` -> [key_dim*2 + value_dim]
    (q, k, v), ``in_proj_z`` -> value_dim (the RMSNorm gate), ``in_proj_b`` /
    ``in_proj_a`` -> num_v_heads (the delta-rule beta and decay-rate inputs). A
    causal depthwise ``conv1d`` (grouped, kernel ``conv_kernel``) runs on the
    mixed qkv before the q/k/v split. The recurrent Gated-Delta-Net update keeps
    a per-request state (``[num_v, key_dim, value_dim]``) plus a conv ring buffer
    (``[conv_dim, kernel-1]``); both live in the linear-state pool the model owns
    (``ctx.linear_state_pool``), indexed by this request's ``linear_slot_idx``.
    """

    def __init__(
        self,
        config: ModelConfig,
        device,
        dtype,
        layer_id: int,
        linear_num_key_heads: int,
        linear_num_value_heads: int,
        linear_key_head_dim: int,
        linear_value_head_dim: int,
        linear_conv_kernel_dim: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.num_k_heads = linear_num_key_heads
        self.num_v_heads = linear_num_value_heads
        self.head_k_dim = linear_key_head_dim
        self.head_v_dim = linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel = linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.group_ratio = self.num_v_heads // self.num_k_heads if self.num_k_heads else 1

        hidden = config.hidden_size
        # Every module takes the compute dtype (the loader streams weights in that
        # dtype and copies into these params); the reference math then runs in
        # float32 regardless, so the forward is exact in bf16/fp16/fp32 alike.
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel,
            groups=self.conv_dim,
            padding=self.conv_kernel - 1,
            device=device,
            dtype=dtype,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads, device=device, dtype=dtype))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads, device=device, dtype=dtype))
        self.norm = _RMSNormGated(self.head_v_dim, eps=eps, device=device, dtype=dtype)
        # Pure-torch tensor-parallel Linear port (issue-24 WP6), not ``nn.Linear``.
        # On the B70 (TP=1) each reduces to a plain ``x @ w.T`` -- the identical
        # math ``nn.Linear`` runs -- while as ``nn.Parameter``-backed ``BaseOP``s
        # they stay drop-in for the checkpoint loader (``named_parameters()`` +
        # ``param.copy_()``) and drop the CUDA JIT / NCCL deps upstream carries.
        # Weights are allocated CPU-side here and moved by the model's ``.to(device)``
        # (see Qwen3_5MoEForCausalLM.__init__), matching every other dense module.
        # ``out_proj`` folds the recurrent core output back to the hidden dim -- the
        # row-parallel output projection -- so it maps to LinearOProj; the input-side
        # projections (qkv / z gate / b beta / a decay) are replicated.
        from freetoken.layers import LinearOProj, LinearReplicated

        self.out_proj = LinearOProj(self.value_dim, hidden, has_bias=False, dtype=dtype)
        self.in_proj_qkv = LinearReplicated(hidden, self.key_dim * 2 + self.value_dim, has_bias=False, dtype=dtype)
        self.in_proj_z = LinearReplicated(hidden, self.value_dim, has_bias=False, dtype=dtype)
        self.in_proj_b = LinearReplicated(hidden, self.num_v_heads, has_bias=False, dtype=dtype)
        self.in_proj_a = LinearReplicated(hidden, self.num_v_heads, has_bias=False, dtype=dtype)

    def _conv(self, mixed_qkv: torch.Tensor, slot, hidden_dtype) -> torch.Tensor:
        """Causal depthwise conv over the mixed qkv; updates the conv ring.

        ``mixed_qkv`` is ``[B, conv_dim, T]``. ``slot`` is this request's
        :class:`_LinearStateSlot` (or None: a self-contained ring per call, used
        by the standalone math path with no pool). Returns ``[B, conv_dim, T]``.
        The ring holds the trailing (kernel-1) positions; the new ring is the
        trailing (kernel-1) of the concatenated (old ring + new tokens) sequence,
        and it is written back in place so the next step reads it.

        The reference's ``causal_conv1d_fn``/``causal_conv1d_update`` apply the
        model's activation (``config.hidden_act``, ``"silu"`` for this
        checkpoint) elementwise to the WHOLE conv output (query, key, and value
        channels alike) before the q/k/v split -- this was missing here (issue
        #147), a real bug affecting every linear-attention layer, every token.
        """
        B, C, T = mixed_qkv.shape
        w = self.conv1d.weight  # [C, 1, K] (groups=C -> each channel is 1 input)
        ring_len = self.conv_kernel - 1
        if slot is None:
            padded = torch.cat(
                [torch.zeros(B, C, ring_len, device=mixed_qkv.device, dtype=mixed_qkv.dtype), mixed_qkv],
                dim=-1,
            )
            out = F.conv1d(padded, w, padding=0, groups=C)[:, :, :T]
            return F.silu(out)
        # Old ring (K-1) + new tokens -> conv output for the new tokens only; the
        # new ring is the trailing (K-1) of that concatenation.
        seq = torch.cat([slot.conv_state.unsqueeze(0), mixed_qkv], dim=-1)  # [B, C, K-1+T]
        out = F.conv1d(seq, w, padding=0, groups=C)[:, :, -T:]
        if T > 0:
            slot.conv_state.copy_(seq[:, :, -ring_len:][0])
        return F.silu(out)

    def _delta_rule(self, q, k, v, g, beta, slot, out_dtype=None):
        """Recurrent Gated-Delta-Net over the (post-conv) tokens.

        ``q``/``k``/``v`` are ``[B, T, num_v, D]`` (the head dim after the
        repeat_interleave to num_v heads), ``g``/``beta`` are ``[B, T, num_v]``.
        ``slot`` is the request's state (:class:`_LinearStateSlot`, ``.state`` the
        ``[num_v, key_dim, value_dim]`` matrix) or None (fresh zero state). The
        state is kept in float32 (the reference upcasts it); the per-token output
        is cast to ``out_dtype`` (the model's compute dtype) so a float32
        recurrent core never leaks into the bf16/fp16 residual stream. Returns
        ``[B, T, num_v, D]`` in ``out_dtype``.
        """
        B, T, H, KD = q.shape
        VD = v.shape[-1]
        # Cast the output to the model's compute dtype, NOT q.dtype: the decay
        # rate ``g`` is computed in float32, so q enters this in float32 and
        # q.dtype would be float32 -- leaking into the downstream (bf16) norm.
        out_dtype = out_dtype if out_dtype is not None else q.dtype
        q = _l2norm(q, dim=-1, eps=1e-6)
        k = _l2norm(k, dim=-1, eps=1e-6)
        q, k, v, beta, g = [
            x.transpose(1, 2).contiguous().to(torch.float32) for x in (q, k, v, beta, g)
        ]  # [B, H, T, *]
        scale = 1.0 / (KD**0.5)
        q = q * scale
        out = torch.zeros(B, H, T, VD, dtype=torch.float32, device=q.device)
        if slot is None:
            s = torch.zeros(B, H, KD, VD, dtype=torch.float32, device=q.device)
        else:
            s = slot.state.to(torch.float32)  # [B, H, KD, VD]
        for i in range(T):
            q_t = q[:, :, i]
            k_t = k[:, :, i]
            v_t = v[:, :, i]
            g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, :, i].unsqueeze(-1)
            s = s * g_t
            kv_mem = (s * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            s = s + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            out[:, :, i] = (s * q_t.unsqueeze(-1)).sum(dim=-2)
        if slot is not None:
            # s is batched [B, H, KD, VD]; the per-request slot.state is [H, KD, VD].
            slot.state.copy_(s[0])
        return out.transpose(1, 2).contiguous().to(out_dtype)

    def forward(self, hidden_states, positions, table_idx, ctx, batch, linear_slot_idx=None) -> torch.Tensor:
        # ``hidden_states`` is this request's token slice [T, H] (the decoder
        # layer hands it a 2-D per-request slice, mirroring qwen3_moe); positions
        # is that slice's [T]. T is this request's new-token count.
        # ``linear_slot_idx`` (issue `semantic-cache-e2e`, #172) is this
        # request's OWN GDN-state pool slot -- distinct from ``table_idx``
        # (the KV page-table row) once a hybrid engine with prefix caching
        # assigns one (Req.linear_slot_idx). Falls back to table_idx (the
        # pre-#172 1:1 behavior) when not set, so a non-hybrid-managed run
        # is unaffected.
        T = hidden_states.shape[0]
        slot_idx = linear_slot_idx if linear_slot_idx is not None else table_idx
        slot = _pool_slot(ctx.linear_state_pool, self.layer_id, slot_idx)
        # Mixed qkv projection + causal conv (the conv ring is per-request).
        mixed_qkv = self.in_proj_qkv(hidden_states).unsqueeze(0).transpose(1, 2)  # [1, conv_dim, T]
        z = self.in_proj_z(hidden_states)  # [T, value_dim]
        b = self.in_proj_b(hidden_states)  # [T, num_v]
        a = self.in_proj_a(hidden_states)  # [T, num_v]
        mixed_qkv = self._conv(mixed_qkv, slot, hidden_states.dtype)  # [1, conv_dim, T]
        query, key, value = torch.split(
            mixed_qkv.squeeze(0).transpose(0, 1),  # [T, conv_dim]
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.reshape(T, self.num_k_heads, self.head_k_dim)
        key = key.reshape(T, self.num_k_heads, self.head_k_dim)
        value = value.reshape(T, self.num_v_heads, self.head_v_dim)
        if self.group_ratio > 1:
            query = query.repeat_interleave(self.group_ratio, dim=1)
            key = key.repeat_interleave(self.group_ratio, dim=1)
        beta = b.sigmoid()
        # The decay rate is computed in float32 (the reference does): upcast the
        # decay-log (A_log) and the per-token decay input (a) + its bias so a
        # bf16 compute dtype never mixes with the fp32 A_log.
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        # _delta_rule returned batched [1, T, num_v, D] (the head dim after the
        # repeat_interleave to num_v heads); squeeze the leading batch back to [T, num_v, D].
        core_attn_out = self._delta_rule(query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0), g.unsqueeze(0), beta.unsqueeze(0), slot, out_dtype=hidden_states.dtype)
        core_attn_out = core_attn_out.squeeze(0)  # [T, num_v, D]
        # RMSNormGated is applied PER VALUE HEAD (weight is [D] = head_v_dim and
        # z is the per-head gate [T, num_v, D]); only flatten after the gate.
        z = z.view(T, self.num_v_heads, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(T, -1)  # [T, value_dim]
        return self.out_proj(core_attn_out)


class _Qwen35Attention:
    """A Qwen3.5/3.6 full-attention (gated GQA) layer.

    ``q_proj`` projects to ``num_heads * head_dim * 2``: the first half is the
    query, the second half the output *gate*. ``q_norm`` / ``k_norm`` RMS-norm
    the head (the gate is NOT normed). Partial RoPE rotates the first
    ``partial_rotary_factor * head_dim`` of the head. The (normed, partially
    rotated) keys/values are appended to the paged KV pool at this request's
    positions; the attention backend then reads the full history back. The output
    is gated by ``sigmoid(gate)`` before ``o_proj``.

    The forward mirrors the qwen3_moe attention contract exactly: ``positions``
    is this request's [B, T] (token-major), K/V are written head-major
    ``[heads, T, head_dim]`` to the pool, and ``table_idx`` selects this request.
    """

    def __init__(
        self,
        config: ModelConfig,
        device,
        dtype,
        layer_id: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        partial_rotary_factor: float,
        rope_theta: float,
        eps: float,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden = config.hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        # Only this fraction of the head is rotated (Qwen3.6 partial rotary).
        self.head_dim_rot = int(head_dim * (partial_rotary_factor or 1.0))
        theta = float(rope_theta)
        # Pure-torch tensor-parallel Linear port (issue-24 WP6), not ``nn.Linear``
        # (CPU-allocated here, moved by the model's ``.to(device)``; see above).
        # Qwen3.5's full-attention q_proj fuses the query and the output gate into
        # one [num_heads, 2*head_dim] projection (split in forward), so its output
        # size is num_heads*head_dim*2 -- a single replicated weight the checkpoint
        # stores as ``q_proj.weight``. (It is NOT the q+k+v merged projection, so it
        # maps to LinearReplicated, not LinearQKVMerged, whose output size would be
        # (num_heads + 2*num_kv_heads)*head_dim.) o_proj is the row-parallel output
        # projection; k/v are plain replicated projections.
        from freetoken.layers import LinearOProj, LinearReplicated

        self.q_proj = LinearReplicated(self.hidden, num_heads * head_dim * 2, has_bias=False, dtype=dtype)
        self.k_proj = LinearReplicated(self.hidden, num_kv_heads * head_dim, has_bias=False, dtype=dtype)
        self.v_proj = LinearReplicated(self.hidden, num_kv_heads * head_dim, has_bias=False, dtype=dtype)
        self.o_proj = LinearOProj(num_heads * head_dim, self.hidden, has_bias=False, dtype=dtype)
        self.q_norm = _RMSNorm(head_dim, eps=eps, device=device, dtype=dtype)
        self.k_norm = _RMSNorm(head_dim, eps=eps, device=device, dtype=dtype)
        # Partial-RoPE inverse frequencies: theta ** (-2i / rotary_dim).
        inv_freq = theta ** (
            -torch.arange(0, self.head_dim_rot, 2, device=device, dtype=torch.float32)
            / self.head_dim_rot
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, hidden_states, positions, table_idx, ctx, batch) -> torch.Tensor:
        # ``hidden_states`` is this request's hidden slice [B, T, H]; ``positions``
        # is this request's [B, T] absolute positions (the decoder layer sliced it
        # off the global per-step positions tensor, exactly like qwen3_moe).
        T = hidden_states.shape[0]
        # The decoder layer hands us a 2-D per-request slice [T, H]; the head-major
        # projections below want a leading batch dim, so add one (bsz=1).
        hs = hidden_states.unsqueeze(0)
        # q_proj -> [1, T, heads, 2*head_dim]; split the query (first half) from
        # the output gate (second half).
        q_g = self.q_proj(hs).view(1, T, self.num_heads, self.head_dim * 2)
        q, gate = q_g.split(self.head_dim, dim=-1)
        q = self.q_norm(q)
        k = self.k_norm(self.k_proj(hs).view(1, T, self.num_kv_heads, self.head_dim))
        v = self.v_proj(hs).view(1, T, self.num_kv_heads, self.head_dim)
        # Partial-RoPE over this request's absolute positions. _rope_for_positions
        # returns [1, T, rotary_dim]; _apply_rotary_pos_emb expects a 2-D
        # [T, rotary] (it unsqueezes the head dim itself, dim=1 -> [1, 1, T,
        # rotary]), so drop the leading 1 to match the qwen3_moe convention.
        cos, sin = _rope_for_positions(self.inv_freq, positions, self.head_dim_rot)
        cos = cos.reshape(T, -1)
        sin = sin.reshape(T, -1)
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)
        # Lay out head-major [heads, T, head_dim] (the backend's expected order).
        # Drop the leading batch dim (bsz=1) so q/k/v are 3-D -- the attention
        # backend and the paged-KV pool both read the head count from dim 0, so a
        # stray batch dim would be mistaken for a head (as qwen3_moe does).
        q = q.transpose(1, 2)[0]
        k = k.transpose(1, 2)[0]
        v = v.transpose(1, 2)[0]
        # write_kv's third argument is out_loc -- PHYSICAL pool slots, not
        # logical token positions. These only coincide under an identity page
        # table; MHAKVCache's real per-request free-list allocator is NOT
        # identity (slot 0 is reserved as dummy/padding, so a real request's
        # first token never lands on slot 0). See qwen3/qwen3_moe's identical
        # fix (#234) -- this call had the same wrong "identity table" assumption.
        out_loc = ctx.page_table[table_idx, positions.long()]
        ctx.kv_cache.write_kv(k, v, out_loc, self.layer_id)
        out = ctx.attn_backend.forward(q, k, v, self.layer_id, batch, table_idx=table_idx)
        # Gated output: sigmoid(gate) elementwise over the attention output.
        # out is head-major [heads, T, head_dim]; make the gate head-major to match.
        gate = gate.transpose(1, 2)[0]
        out = out * torch.sigmoid(gate)
        # Fold the heads back into the hidden dim: [T, heads*head_dim] -> [T, H].
        return self.o_proj(out.transpose(0, 1).reshape(T, -1))


def _rope_for_positions(
    inv_freq: torch.Tensor, positions: torch.Tensor, rotary_dim: int
) -> tuple:
    """cos/sin ``[1, N, rotary_dim]`` for absolute token ``positions`` (partial
    RoPE: only the first ``rotary_dim`` of the head are rotated)."""
    # inv_freq has rotary_dim//2 entries (one per (x, y) pair); doubling matches
    # the full-RoPE convention (the rotate_half split expects cat(freqs, freqs)).
    freqs = torch.outer(positions.to(torch.float32), inv_freq)  # [N, rotary_dim//2]
    freqs_full = torch.cat((freqs, freqs), dim=-1)  # [N, rotary_dim]
    cos = freqs_full.cos()[None, :, :]  # [1, N, rotary_dim]
    sin = freqs_full.sin()[None, :, :]  # [1, N, rotary_dim]
    return cos, sin


class _Qwen35MoE:
    """A Qwen3.5/3.6 MoE block: a 256-way top-8 router + an always-on shared
    expert. The router + shared expert are dense (on the device); the routed
    experts are served through the engine's ``OffloadMoeCache`` (host banks ->
    small device LRU slot pool) when offload is on, or from in-VRAM expert
    modules when it is not. The forward mirrors qwen3_moe's MoE block: it
    consumes the *token-major* [num_tokens, H] slice and reaches the cache /
    layer map through ``ctx.model``.
    """

    def __init__(self, config: ModelConfig, device, dtype, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.use_offload = bool(getattr(config, "use_offload_moe", False))
        # The CPU backend (issue #8) keeps the experts non-device-resident like
        # the XPU offload backend, so it also sets use_offload_moe; use_cpu_moe is
        # the *distinct* flag that switches the routed-expert GEMM to the host.
        # The forward dispatches off these block-local flags (set from the
        # loader-resolved config at construction) rather than reading
        # ``ctx.model.moe_backend`` -- the test harness (and any non-engine caller)
        # builds a ``Context`` that never sets ``.model``, so reading it would
        # crash the forward on every path.
        self.use_cpu = bool(getattr(config, "use_cpu_moe", False))
        # Issue #9 (moe-hybrid): the distinct flag for the per-step split. Set
        # from the loader-resolved config at construction (the forward dispatches
        # off block-local flags, not ctx.model, so a test-harness Context that
        # never sets .model still behaves correctly).
        self.use_hybrid = bool(getattr(config, "use_hybrid", False))
        # Issue #8: this block's decoder-layer index (== self.layer_id). The
        # per-layer CPU/offload partition (--moe-cpu-layers) is keyed by
        # MoE-layer index, so the forward resolves the index the same way the
        # forwards resolve the bank rows: model.moe_layer_id[self.layer_id].
        # (The block keeps a copy here so a test-harness Context -- which never
        # sets .model -- can still be distinguished from the no-CPU-layers case.)
        self.moe_layer_id = layer_id
        self.expert_inter = config.moe_intermediate_size
        self.shared_inter = int(config.attrs.get("text_config", {}).get("shared_expert_intermediate_size", config.moe_intermediate_size))
        # Router: pure-torch replicated Linear (issue-24 WP6) -- ``num_experts``
        # is small and fully replicated.
        from freetoken.layers import LinearReplicated

        self.gate = LinearReplicated(self.hidden, self.num_experts, has_bias=False, dtype=dtype)
        if self.use_offload or self.use_cpu or self.use_hybrid:
            self.experts = None
        else:
            self.experts = nn.ModuleList(_Qwen35Expert(config, device, dtype) for _ in range(self.num_experts))
        self.shared_expert = _Qwen35Expert(
            _make_shared_config(config, self.shared_inter), device, dtype
        )
        # Shared-expert router: a single scalar gate (the shared expert is always
        # on, so the router is a 1-way linear), on the pure-torch Linear port.
        self.shared_expert_gate = LinearReplicated(self.hidden, 1, has_bias=False, dtype=dtype)

    def forward(self, hidden_states, table_idx, ctx, batch) -> torch.Tensor:
        in_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, in_shape[-1])  # [n_tok, H]
        gate_logits = self.gate(flat)  # [n_tok, num_experts]
        router_probs = F.softmax(gate_logits, dtype=torch.float32, dim=-1)
        top_w, top_idx = torch.topk(router_probs, self.top_k, dim=-1)  # [n, k]
        top_w = (top_w / top_w.sum(dim=-1, keepdim=True)).to(flat.dtype)
        # The always-on shared expert (dense, on the device), gated per-token.
        shared_out = F.sigmoid(self.shared_expert_gate(flat)) * self.shared_expert(flat)
        # Routed experts: the host-side expert GEMM (CPU backend, issue #8), the
        # host-offload LRU slot pool (XPU offload, ADR 0002), or the in-VRAM
        # modules. When the model carries a ``moe_cpu_moe_layers`` partition
        # (issue #8, --moe-cpu-layers) the per-layer decision reads it through
        # ``ctx.model`` (a cpu/hybrid backend can mix CPU + XPU-offload MoE
        # layers); when it does not (a test harness's Context never sets .model)
        # the block falls back to its block-local use_cpu flag, so existing
        # harnesses are unaffected.
        if self._is_cpu_layer(ctx):
            routed_out = self._forward_cpu(flat, top_idx, top_w, ctx, batch)
        elif self.use_hybrid:
            # Issue #9: the per-step split. A layer the partition steers to the
            # CPU computes fully there (as the cpu backend does); the rest split
            # each step's misses between the PCIe-fetch/XPU half and the host-CPU
            # half by the profile's fetch fraction. The split is *intra-layer*
            # (per decode step), which is where the parent's q* policy lives --
            # not the whole-layer partition the cpu backend uses.
            routed_out = self._forward_hybrid(flat, top_idx, top_w, ctx, batch)
        elif self.use_offload:
            routed_out = self._forward_offload(flat, top_idx, top_w, ctx, batch)
        else:
            routed_out = self._forward_inram(flat, top_idx, top_w, ctx)
        return (routed_out + shared_out).view(in_shape)

    def _is_cpu_layer(self, ctx) -> bool:
        """Whether this block's routed experts compute on the CPU (issue #8).

        True when the resolved MoE backend is a CPU variant (``cpu`` / ``hybrid``)
        AND the model's ``moe_cpu_moe_layers`` partition names this block. The
        partition (resolved from ``--moe-cpu-layers`` by the loader) is a concrete
        list of MoE-layer indices: ``[]`` means no layer is steered to the CPU
        (the serve default ``--moe-backend auto`` -> ``offload``, so this block
        stays on the XPU slot pool), and a full list means every MoE layer on the
        CPU (the ``--moe-backend cpu`` / explicit ``"auto"`` spec). The MoE index
        is resolved the *same way* the forwards resolve the bank rows
        (``model.moe_layer_id[self.layer_id]``), so the partition decision can
        never diverge from the expert weights this block actually reads. When
        ``ctx.model`` is unset (a test harness), fall back to the block-local
        ``use_cpu`` flag so non-engine callers keep their existing behavior.
        """
        model = getattr(ctx, "model", None)
        if model is None:
            # A test-harness Context never sets .model. For the cpu backend that
            # means "compute on the CPU" (use_cpu); for hybrid it means "no CPU
            # layers" -> the block's use_hybrid flag routes to _forward_hybrid
            # with the (default 0.0) fetch fraction, i.e. pure offload.
            return self.use_cpu
        moe_backend = getattr(model, "moe_backend", None)
        if moe_backend not in ("cpu", "hybrid"):
            # In-VRAM (fused/None) or pure offload: no CPU layers.
            return False
        cpu_layers = getattr(model, "moe_cpu_moe_layers", None)
        if cpu_layers is None:
            # For the cpu backend a None partition means "every MoE layer on the
            # CPU" (the --moe-backend cpu default). For the hybrid backend a None
            # partition means "no whole layer is carved out to the CPU" -- the
            # per-step q* split in _forward_hybrid is the hybrid mechanism, so the
            # CPU half is governed by the fetch fraction, not a layer partition.
            return moe_backend == "cpu"
        moe_layer_id_map = getattr(model, "moe_layer_id", None)
        if moe_layer_id_map is None:
            return bool(cpu_layers)
        # Resolve the MoE-layer index exactly the way the forwards resolve the
        # bank rows (moe_layer_id[self.layer_id]) so the partition decision and
        # the expert weights this block reads can never diverge.
        moe_idx = moe_layer_id_map[self.layer_id]
        return moe_idx in cpu_layers

    def _forward_cpu(self, flat, top_idx, top_w, ctx, batch):
        """Run the routed experts on the host (issue #8, ADR 0002).

        Same contract as ``_Qwen3MoE._forward_cpu``: read this layer's pinned
        host banks straight off the loader-built ``cpu_expert_sources`` (no device
        round-trip -- the host is the source of truth) and run the expert GEMM on
        the CPU, shipping only the resulting activations back. The accumulated
        order is expert-major-then-top-k-column, so the CPU path matches the
        in-VRAM reference's greedy tokens exactly.
        """
        model = ctx.model
        moe_idx = model.moe_layer_id[self.layer_id]
        # The host banks are the source of truth (ADR 0002) and live in the moe
        # cache the loader attached (``set_bank_sources`` keeps the raw per-layer
        # [E, ...] host tensors, already unwrapped from any _PlainBank). Read
        # this layer's gate_up / down straight off the host -- no device
        # round-trip, no PCIe stream, no LRU slot juggling.
        sources = model.moe_cache.bank_sources
        gate_up = sources["gate_up"][moe_idx]
        down = sources["down"][moe_idx]
        from freetoken.moe.cpu_executor import CpuMoeExecutor

        executor = getattr(model, "_moe_cpu_executor", None)
        if executor is None:
            threads = int(getattr(model.config, "moe_cpu_threads", 0) or 0)
            executor = CpuMoeExecutor(
                num_experts=int(model.config.num_experts),
                intermediate=int(model.config.moe_intermediate_size),
                threads=threads,
            )
            model._moe_cpu_executor = executor
        return executor.forward(flat, top_idx, top_w, gate_up, down)

    def _forward_inram(self, flat, top_idx, top_w, ctx=None):
        # Indices are built on the HOST, not with device-side boolean-mask
        # indexing (`flat[sel]`) or `torch.nonzero` on an XPU tensor: on this
        # torch/XPU build, `nonzero()` (which boolean-mask indexing calls
        # internally) silently returns an EMPTY result for an XPU bool tensor
        # regardless of its actual content (`sel.sum()` / `sel.tolist()` are
        # correct; `sel.nonzero()` / `flat[sel]` are not) -- a real
        # correctness bug, not just the "implicit D2H sync" performance
        # concern the offload path already routes around this same way (see
        # qwen3_moe's in-VRAM forward for the same fix).
        #
        # EXCEPT while a graph capture is in flight (issue
        # moe-fused-graph-capture, #123): the CPU round-trip below
        # (`top_idx.to("cpu")`) is itself a device->host sync, a hard capture
        # error. Route densely instead -- every expert on every token,
        # weight-summed with a mask built from `==`/`torch.where` (plain
        # elementwise ops: no `nonzero()`, no sync, so neither the broken
        # on-device nonzero() nor the capture restriction applies). Strictly
        # more compute, so opt-in via `_capturing` only (mirrors #118's
        # fixed-KV-buffer attention, the same trade-compute-for-capturability
        # shape).
        model = getattr(ctx, "model", None)
        if getattr(model, "_capturing", False):
            out = torch.zeros_like(flat)
            for e in range(self.num_experts):
                w = torch.zeros(flat.shape[0], device=flat.device, dtype=flat.dtype)
                for slot in range(self.top_k):
                    w = torch.where(top_idx[:, slot] == e, top_w[:, slot], w)
                out = out + w[:, None] * self.experts[e](flat)
            return out

        out = torch.zeros_like(flat)
        top_idx_cpu = top_idx.to("cpu")
        for e in range(self.num_experts):
            for slot in range(self.top_k):
                sel_cpu = top_idx_cpu[:, slot] == e
                if not bool(sel_cpu.any()):
                    continue
                idx = sel_cpu.nonzero(as_tuple=True)[0].to(flat.device)
                w = top_w.index_select(0, idx)[:, slot, None]
                y = self.experts[e](flat.index_select(0, idx))
                out.index_add_(0, idx, w * y)
        return out

    def _forward_offload(self, flat, top_idx, top_w, ctx, batch):
        """Serve the routed experts through the host-offload LRU slot pool.

        All slot-routing bookkeeping is done on the *host* (the cache's
        ``slot_for_id`` map is already Python-side), and the forward pushes only
        a clean, in-bounds ``top_idx`` plus tiny CPU-built index/mask tensors to
        the XPU. This is deliberate: rewriting ``top_idx`` in place on the XPU
        (the old path) left per-element writes that raced the async topk/softmax
        kernels, so the forward could read a stale id and index the slot pool
        out of bounds -- an XPU kernel abort (Indexing.h "index out of bounds")
        that takes down the whole GPU. Host-driven routing makes the ids
        deterministic and in-bounds by construction.
        """
        return self._forward_offload_core(
            flat, top_idx, top_w, ctx, batch,
            exclude=set(),
        )

    def _forward_hybrid(self, flat, top_idx, top_w, ctx, batch):
        """Serve the routed experts by splitting each step's misses (issue #9).

        The host-offload and host-CPU halves share the same pinned expert banks
        (ADR 0002), so a decode step's routed-expert misses can be *partitioned*:
        a fraction f PCIe-fetched into the XPU LRU slot pool and computed there,
        the rest (1 - f) computed on the host CPU from the same host banks. f is
        the ``ft bench bw`` profile's fetch fraction for this expert format
        (``q*``: pcie/(pcie+cpu) -- of the two halves' combined bandwidth, the
        share carried by PCIe), and is what balances the two halves' completion
        times.

        Correctness: the two halves are *disjoint* expert sets (the CPU set and
        the XPU set never overlap and together cover exactly the routed experts),
        and each half uses the *same* math + accumulation order as the pure
        backend it mirrors -- so hybrid's output is numerically identical to
        offload's (the q* split changes *which* experts ride each transport, not
        the arithmetic). The XPU side gathers only the fetched experts' slots
        (``exclude`` = the CPU-computed experts); the CPU side computes only the
        rest. The two contributions are summed per-row.

        Prefill (a whole layer, or a layer the partition steers to the CPU) has
        no miss-split -- every routed expert is made resident -- so it degrades
        cleanly to the offload path there. The split applies to the routed-expert
        decode step, where the q* balance lives.
        """
        model = ctx.model
        # Per-layer CPU carve-out (issue #8 --moe-cpu-layers): a layer named
        # there computes its routed experts fully on the CPU, even under the
        # hybrid backend (the partition is the coarse whole-layer split; the q*
        # fraction is the fine per-step split for the layers it does not name).
        if self._is_cpu_layer(ctx):
            return self._forward_cpu(flat, top_idx, top_w, ctx, batch)
        # Issue moe-quant-banks-compute (#137): the CPU half's math
        # (_cpu_subset_math) reads model.moe_cache.bank_sources["gate_up"] /
        # ["down"] directly -- the "bf16" schema's bank names -- and runs
        # plain-float matmuls on them. A "gptq_int4" cache's bank_sources use
        # different names entirely (qweight_gate_up, ...), so that lookup
        # would KeyError. Rather than teach the CPU half to dequantize too
        # (real, separable follow-up work -- the CPU path has none of the
        # slot-cache/copy_missing machinery SlotWeightAccessor hooks into, so
        # it would need its own dequant-and-cache logic), this format is
        # excluded from the hybrid split for now: force fetch_frac to 1.0 so
        # every miss rides PCIe through the (already gptq_int4-aware)
        # offload path below. A documented, deliberate tradeoff (this
        # issue's own accept criteria explicitly allows it as a first cut),
        # not a silent gap -- gptq_int4 previously would have crashed loudly
        # here (KeyError) rather than produced wrong numbers, and now simply
        # never reaches that code path. "fp8_block" (issue
        # moe-quant-banks-fp8, #152), "mxfp4" (issue moe-quant-banks-mxfp4,
        # #153), and "int8_channel" (issue moe-quant-banks-int8, #154) hit
        # the exact same bank-name mismatch (weight_gate_up/scale_gate_up/...
        # or blocks_.../scales_... instead of gate_up/down) -- excluded for
        # the same reason, not a separate decision.
        cache_quant_format = getattr(getattr(model, "moe_cache", None), "quant_format", "bf16")
        # The fetch fraction f (share of misses PCIe-fetched, the rest on CPU).
        # Read through ctx.model (the loader stores it there); a test-harness
        # Context that sets none falls back to the block-local 0.0 (pure offload),
        # which is the correct no-split default.
        fetch_frac = float(getattr(model, "moe_hybrid_fetch_fraction", 0.0) or 0.0)
        if cache_quant_format in ("gptq_int4", "fp8_block", "mxfp4", "int8_channel"):
            fetch_frac = 1.0
        if fetch_frac <= 0.0:
            # No usable profile -> every miss rides PCIe (pure offload).
            return self._forward_offload(flat, top_idx, top_w, ctx, batch)
        if fetch_frac >= 1.0:
            # 100% fetch -> no CPU misses (pure offload); avoid a degenerate CPU
            # call with an empty expert set.
            return self._forward_offload(flat, top_idx, top_w, ctx, batch)
        # Split the routed-expert *ids* into the two disjoint halves. The XPU half
        # (fetched) gets the top round(n*f) ids; the CPU half the rest. The split
        # is over the *unique* routed ids of this step (a miss is per-expert, not
        # per-token-column), computed host-side from the topk snapshot so it is
        # deterministic and never triggers a device->host sync.
        expert_ids_cpu = top_idx.to("cpu")
        seen: list[int] = []
        for eid in expert_ids_cpu.reshape(-1).tolist():
            if eid not in seen:
                seen.append(eid)
        n = len(seen)
        n_fetch = int(round(n * fetch_frac))
        # --moe-hybrid-max-fetch (issue #9): a non-negative int caps the per-step
        # PCIe-fetched expert count -- the operator's override of the profile's
        # q* fraction (a hard ceiling, not a ratio). -1 / unset = fully
        # profile-driven (no cap). The cap only ever *shrinks* the XPU half (and
        # thus grows the CPU half): it is a floor on the CPU share, so the
        # disjoint-cover invariant (XPU set + CPU set == routed set) still holds
        # and the output stays numerically identical to pure offload.
        max_fetch = int(getattr(model, "moe_hybrid_max_fetch", -1) or -1)
        if 0 <= max_fetch < n_fetch:
            n_fetch = max_fetch
        # Clamp the split to a sane range (never empty the XPU or CPU half unless
        # the fraction truly is 0 / 1, handled above).
        n_fetch = max(1, min(n - 1, n_fetch))
        seen_sorted = sorted(seen)
        cpu_experts = set(seen_sorted[: n - n_fetch])  # (1 - f) share -> CPU

        # The two halves run CONCURRENTLY (issue moe-hybrid-overlap): the
        # host-CPU half's pure-CPU matmuls (no XPU tensor touched) run on a
        # persistent single-worker background thread while the XPU half's
        # PCIe fetch + gather runs on *this* (main) thread -- the same regime
        # benchbw._bench_overlap measures. A decode step then costs
        # max(cpu_half, pcie_half), not their sum, matching the q* fetch
        # fraction's bandwidth-matched assumption. The pool is reused across
        # every layer / every step (cached on the model) rather than spawning
        # a fresh thread per call: thread-creation overhead alone was large
        # enough relative to a small model's per-expert matmul cost to erase
        # most of the overlap's benefit when measured with a fresh Thread
        # each time.
        #
        # The device<->host transfers (flat -> CPU in, the CPU result -> device
        # out) must stay on this thread: the XPU runtime faults that sync when
        # issued off the thread that built the engine (see
        # test_serve_live_engine_xpu.py's docstring for the same constraint).
        # So the CPU half's input is prepared here before the submit, and its
        # output is moved back to the device here after the result is
        # collected.
        x_cpu = flat.to("cpu", non_blocking=True)

        future = self._hybrid_cpu_pool(model).submit(
            self._cpu_subset_math, x_cpu, expert_ids_cpu, ctx, cpu_experts
        )

        out = self._forward_offload_core(
            flat, top_idx, top_w, ctx, batch,
            exclude=cpu_experts,
        )

        cpu_out = future.result()
        # The host-CPU half's (disjoint) share, so the per-row sum matches the
        # sequential version exactly regardless of which half finishes first.
        out += cpu_out.to(flat.device, non_blocking=True)
        return out

    @staticmethod
    def _hybrid_cpu_pool(model):
        """A single-worker thread pool for the hybrid CPU half, cached on the
        model so it survives across decode steps / layers instead of paying
        thread-creation cost on every call (see ``_forward_hybrid``)."""
        pool = getattr(model, "_moe_hybrid_cpu_pool", None)
        if pool is None:
            from concurrent.futures import ThreadPoolExecutor

            pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="moe-hybrid-cpu")
            model._moe_hybrid_cpu_pool = pool
        return pool

    def _cpu_subset_math(self, x_cpu, expert_ids_cpu, ctx, cpu_experts):
        """Pure-CPU math for the hybrid split's host-CPU half.

        Computes *only* the routed experts in ``cpu_experts`` (from the pinned
        host banks) and returns their per-row contribution, on the CPU -- so
        this is safe to run on a background thread (no XPU tensor is read or
        written anywhere in this method; the device<->host transfers around it
        are the caller's job, on the main thread). A row that routes to an
        expert not in ``cpu_experts`` contributes nothing (that row is served
        by the XPU half).

        Accumulation is expert-major then top-k-column (matching
        ``_forward_cpu`` / the in-VRAM reference), so the per-row result the
        hybrid path sums is numerically identical to what the pure CPU backend
        would produce for that subset. The upstream per-row weight for this
        half (``cpu_top_w`` in the pre-overlap code) was always identically
        1.0 -- applying it was a no-op multiply -- so it is omitted here
        rather than reintroduced as a real weight.
        """
        model = ctx.model
        moe_idx = model.moe_layer_id[self.layer_id]
        sources = model.moe_cache.bank_sources
        gate_up = sources["gate_up"][moe_idx]
        down = sources["down"][moe_idx]
        B, k = expert_ids_cpu.shape
        num_experts = self.num_experts
        out = torch.zeros_like(x_cpu)
        for e in range(num_experts):
            if e not in cpu_experts:
                continue
            for j in range(k):
                sel = expert_ids_cpu[:, j] == e
                if not bool(sel.any()):
                    continue
                idx = torch.nonzero(sel, as_tuple=False).view(-1)
                x_sel = x_cpu.index_select(0, idx)
                y = self._expert_compute_cpu(gate_up, down, e, x_sel)
                out.index_add_(0, idx, y)
        return out

    @staticmethod
    def _expert_compute_cpu(gate_up, down, e, x_cpu):
        """One expert on the host from the bank row (the CPU half's GEMM)."""
        I = gate_up.shape[1] // 2
        gate = x_cpu @ gate_up[e, 0:I].t()
        up = x_cpu @ gate_up[e, I : 2 * I].t()
        return (F.silu(gate) * up) @ down[e].t()

    def _forward_offload_core(self, flat, top_idx, top_w, ctx, batch, *, exclude):
        model = ctx.model
        layer_id = model.moe_layer_id[self.layer_id]
        cache = model.moe_cache
        # is_prefill must come from the batch's *phase flag*, not from flat.shape[0]:
        # in a mixed step (decode reqs with 1 token each, or a decode req alongside a
        # prefill req) flat.shape[0] > 1 even though every token is a 1-token decode.
        # The offload path then calls materialize_layer (whole layer) instead of
        # ensure_experts (routed experts only). Routing itself is host-side and
        # phase-independent: we snapshot the routed *expert* ids and map them to
        # slots from the cache's slot_for_id, so the phase only decides *which*
        # experts to make resident (the whole layer, or just the routed ones).
        is_prefill = bool(batch is not None and batch.is_prefill)
        is_xpu = bool(getattr(cache, "is_xpu", False))
        B, k = top_idx.shape
        # 1) Snapshot the routed *expert* ids on the host BEFORE the LRU call.
        #    top_idx holds *expert* ids (the topk output) and the cache never
        #    rewrites it (issue #7): the old code rewrote top_idx in place to
        #    *slot* ids, so a repeat routing that skipped the rewrite left the
        #    previous step's slot ids in place and the next step read them as if
        #    they were expert ids -> out-of-bounds (XPU IndexError) / a wrong
        #    expert gather on CPU. Routing is fully determined by this snapshot +
        #    the cache's slot map, so top_idx may keep holding expert ids.
        expert_ids = top_idx.to("cpu")
        # 2) Let the LRU pool decide residency (prefill: whole layer; decode:
        #    only the routed experts), staging the host->XPU copy plan.
        if is_prefill:
            cache.materialize_layer(layer_id)
        else:
            cache.ensure_experts(layer_id, expert_ids)
        cache.copy_missing()
        # 3) Map expert -> slot on the host from the cache's own (Python) map.
        #    An expert the pool evicted maps to -1 -> clamped to 0 with valid=False.
        slots = cache.slot_for_id[layer_id].to("cpu").tolist()
        S = cache.cache_size
        intermediate = int(model.config.moe_intermediate_size)
        # SlotWeightAccessor abstracts gu[s_i, ...]/dn[s_i] bf16 indexing over a
        # quantized bank format too (issue moe-quant-banks-compute, #137): for
        # "bf16" this is the exact same plain-tensor indexing as before (zero
        # behavior change); for "gptq_int4" it dequantizes each distinct
        # resident slot at most once per step, from the packed banks, never
        # the whole checkpoint (the RAM-saving point of the whole epic, #134).
        from freetoken.moe.offload_cache import SlotWeightAccessor

        slot_weights = SlotWeightAccessor(cache, intermediate, flat.dtype)
        dev = flat.device
        out = torch.zeros_like(flat)
        routed_cpu = torch.empty(B, k, dtype=torch.int64)
        valid_cpu = torch.zeros(B, k, dtype=torch.bool)
        for i in range(B):
            for j in range(k):
                slot = slots[int(expert_ids[i, j])]
                ok = 0 <= slot < S
                routed_cpu[i, j] = slot if ok else 0
                valid_cpu[i, j] = ok
        # Loud guard: an out-of-range slot means a stale map entry (should be -1);
        # fail in Python, not with a GPU abort. (The clamp above already makes the
        # push in-bounds, so this is a belt-and-suspenders tripwire.)
        if is_xpu and (~valid_cpu).any():
            stale = int(routed_cpu[~valid_cpu].max()) if (~valid_cpu).any() else -1
            if stale >= S:
                raise IndexError(
                    f"layer {self.layer_id}: offload routed slot id {stale} >= "
                    f"cache_size {S}: stale slot map (ensure_experts desync)"
                )
        # 4) Host-side: (column j, slot s_i) -> the list of row indices that route
        #    to that slot in that column. Built purely on the host from routed_cpu /
        #    valid_cpu, so the loop below never needs to ask the device "which rows
        #    matched?". (A boolean-mask index -- out[sub], with sub = valid &
        #    (routed == s) -- would call nonzero() internally; nonzero's output
        #    shape is data-dependent, so each use forces an implicit device->host
        #    sync once per slot per column, INSIDE the loop. On the shared XPU that mid-block
        #    sync is what faults: no amount of torch.xpu.synchronize() around the
        #    block helps, because the offending sync is in its middle. Integer
        #    index_select / index_add_ have static shapes and never call nonzero.)
        # Build the (expert, slot, rows) groups in EXPERT-MAJOR order: for each
        # expert e (ascending), for each top-k column j (ascending), collect the
        # rows that route to e's slot in column j. This mirrors the in-VRAM
        # _forward_inram loop (for e in range(num_experts): for slot in range(top_k))
        # so the float32 accumulation order into out[i] is identical. The offload
        # transport is a byte-identical weight copy (ADR 0002); the only remaining
        # source of divergence was the accumulation *order* (slot-major here vs
        # expert-major in-VRAM), which produces a ~1e-7 ULP difference that the
        # chaotic Gated-Delta-Net recurrence amplifies over decode steps until the
        # greedy argmax flips (issue #18).
        num_experts = model.config.num_experts
        expert_slots = [int(slots[e]) for e in range(num_experts)]
        expert_to_col: dict[int, list[tuple[int, int]]] = {}
        for j in range(k):
            for i in range(B):
                if bool(valid_cpu[i, j]):
                    e_id = int(expert_ids[i, j])
                    expert_to_col.setdefault(e_id, []).append((j, i))
        groups: list[tuple[int, int, list[int]]] = []
        for e in range(num_experts):
            # Issue #9 hybrid: an expert the host-CPU half serves this step is
            # excluded from the XPU gather (its slot was left at -1 above, so the
            # `0 <= s_i < S` check already drops it; this is the belt-and-
            # suspenders guard against a stale positive slot entry).
            if e in exclude:
                continue
            s_i = expert_slots[e]
            if not (0 <= s_i < S):
                continue
            for j, i in expert_to_col.get(e, []):
                groups.append((j, s_i, [i]))
        # (No device-side push needed: validity is encoded in which rows appear
        # in `groups`, and the gather below uses host-built index tensors.)
        # Gather per (expert, slot, row) on the XPU using host-computed INTEGER
        # indices: static shapes, no nonzero(), no implicit D2H anywhere in the
        # loop. index_add_ accumulates exactly as `out[sel] += ...` did.
        for j, s_i, rows in groups:
            idx = torch.tensor(rows, dtype=torch.long, device=dev)
            y = top_w.index_select(0, idx)[:, j, None] * slot_weights.expert_forward(
                s_i, flat.index_select(0, idx)
            )
            out.index_add_(0, idx, y)
        # NB: we do NOT write the slot ids back into top_idx here (the old code did
        # `top_idx.copy_(routed_dev)`). top_idx is a *fresh* topk tensor each call,
        # and routing is done host-side from the `expert_ids` snapshot + the cache's
        # slot map, so nothing downstream reads top_idx after we return. On the XPU
        # the in-place write raced the next step's fresh topk: the host snapshot of
        # top_idx then read *stale slot ids* (0..S-1, e.g. 5 >= num_experts 4)
        # instead of the just-computed expert ids -- an out-of-bounds slots[] read
        # (XPU IndexError / kernel abort) and, on CPU, a wrong expert gather that
        # silently corrupts the logits (offload no longer matches the in-VRAM
        # reference). The routing is fully determined by the host snapshot, so the
        # device-side top_idx may keep holding expert ids; leave it untouched.
        return out


class _Qwen35Expert:
    """A single MoE expert: gate/up/down projections (SwiGLU). Used by the
    in-VRAM path (and the always-on shared expert, which is dense)."""

    def __init__(self, config: ModelConfig, device, dtype) -> None:
        super().__init__()
        # Pure-torch replicated Linear port (issue-24 WP6); each SwiGLU projection
        # is an independent replicated Linear (the in-VRAM path reads each
        # projection's own checkpoint weight, so we do not fuse them).
        from freetoken.layers import LinearReplicated

        self.gate_proj = LinearReplicated(config.hidden_size, config.moe_intermediate_size, has_bias=False, dtype=dtype)
        self.up_proj = LinearReplicated(config.hidden_size, config.moe_intermediate_size, has_bias=False, dtype=dtype)
        self.down_proj = LinearReplicated(config.moe_intermediate_size, config.hidden_size, has_bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _Qwen35MxfpExpert:
    """A single MXFP4-quantized MoE expert, fully XPU-resident (issue
    `moe-fused-mxfp4`, #180, part of the `quant-xpu` epic, #10).

    Unlike :class:`_Qwen35Expert` (plain bf16 ``gate_proj``/``up_proj``/
    ``down_proj`` ``nn.Linear``s), this holds the checkpoint's packed MXFP4
    ``blocks``/``scales`` (the same per-expert tensors
    :class:`freetoken.models.weight.MxfpExpertBank` streams for the offload
    backend, moved onto the device instead of staying on host) and never
    materializes a dequantized weight: :func:`freetoken.kernel.triton.
    fused_mxfp4_linear.fused_mxfp4_expert_forward` runs the native packed
    GEMM directly, the same kernel the offload backend's dequant-at-compute
    path (#163) uses -- this is its fully-resident sibling, with no host
    round-trip / LRU slot pool at all.
    """

    def __init__(
        self,
        blocks_gate_up: torch.Tensor,
        scales_gate_up: torch.Tensor,
        blocks_down: torch.Tensor,
        scales_down: torch.Tensor,
        intermediate: int,
    ) -> None:
        super().__init__()
        # persistent=False: these are checkpoint-derived, streamed by the
        # loader every load, never part of a state_dict this port saves/
        # restores (matching every other quant bank's own storage -- the
        # offload cache's host banks aren't state_dict members either).
        self.register_buffer("blocks_gate_up", blocks_gate_up, persistent=False)
        self.register_buffer("scales_gate_up", scales_gate_up, persistent=False)
        self.register_buffer("blocks_down", blocks_down, persistent=False)
        self.register_buffer("scales_down", scales_down, persistent=False)
        self.intermediate = intermediate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.fused_mxfp4_linear import fused_mxfp4_expert_forward

        return fused_mxfp4_expert_forward(
            x,
            self.blocks_gate_up,
            self.scales_gate_up,
            self.blocks_down,
            self.scales_down,
            intermediate=self.intermediate,
            out_dtype=x.dtype,
        )


class _Qwen35Fp8Expert:
    """A single block-FP8-quantized MoE expert, fully XPU-resident (issue
    `moe-fused-fp8`, #181, part of the `quant-xpu` epic, #10).

    The block-FP8 sibling of :class:`_Qwen35MxfpExpert`: holds the
    checkpoint's packed ``weight``/``weight_scale_inv`` (the same per-expert
    tensors :class:`freetoken.models.weight.Fp8BlockExpertBank` streams for
    the offload backend, moved onto the device instead of staying on host)
    and never materializes a dequantized weight --
    :func:`freetoken.kernel.triton.fused_fp8_linear.fused_fp8_expert_forward`
    runs the native packed GEMM directly, the same kernel the offload
    backend's dequant-at-compute path (#163) uses.
    """

    def __init__(
        self,
        weight_gate_up: torch.Tensor,
        scale_gate_up: torch.Tensor,
        weight_down: torch.Tensor,
        scale_down: torch.Tensor,
        intermediate: int,
    ) -> None:
        super().__init__()
        self.register_buffer("weight_gate_up", weight_gate_up, persistent=False)
        self.register_buffer("scale_gate_up", scale_gate_up, persistent=False)
        self.register_buffer("weight_down", weight_down, persistent=False)
        self.register_buffer("scale_down", scale_down, persistent=False)
        self.intermediate = intermediate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.fused_fp8_linear import fused_fp8_expert_forward

        return fused_fp8_expert_forward(
            x,
            self.weight_gate_up,
            self.scale_gate_up,
            self.weight_down,
            self.scale_down,
            intermediate=self.intermediate,
            out_dtype=x.dtype,
        )


class _Qwen35Int8Expert:
    """A single compressed-tensors pack-quantized INT8 MoE expert, fully
    XPU-resident (issue `moe-fused-int8`, #182, part of the `quant-xpu`
    epic, #10).

    The INT8 sibling of :class:`_Qwen35MxfpExpert` / :class:`_Qwen35Fp8Expert`:
    holds the checkpoint's packed ``weight_packed``/``weight_scale`` (the
    same per-expert tensors :class:`freetoken.models.weight.Int8ExpertBank`
    streams for the offload backend, moved onto the device instead of
    staying on host) and never materializes a dequantized weight --
    :func:`freetoken.kernel.triton.fused_int8_linear.fused_int8_expert_forward`
    runs the native packed GEMM directly, the same kernel the offload
    backend's dequant-at-compute path (#163) uses. ``k_gate_up``/``k_down``
    (the real logical in-features per projection, an architecture constant
    shared by every expert -- see :class:`Int8ExpertBank`'s own docstring)
    are plain ints, not tensors, so no buffer is needed for them.
    """

    def __init__(
        self,
        weight_packed_gate_up: torch.Tensor,
        weight_scale_gate_up: torch.Tensor,
        weight_packed_down: torch.Tensor,
        weight_scale_down: torch.Tensor,
        intermediate: int,
        k_gate_up: int,
        k_down: int,
    ) -> None:
        super().__init__()
        self.register_buffer("weight_packed_gate_up", weight_packed_gate_up, persistent=False)
        self.register_buffer("weight_scale_gate_up", weight_scale_gate_up, persistent=False)
        self.register_buffer("weight_packed_down", weight_packed_down, persistent=False)
        self.register_buffer("weight_scale_down", weight_scale_down, persistent=False)
        self.intermediate = intermediate
        self.k_gate_up = k_gate_up
        self.k_down = k_down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.fused_int8_linear import fused_int8_expert_forward

        return fused_int8_expert_forward(
            x,
            self.weight_packed_gate_up,
            self.weight_scale_gate_up,
            self.weight_packed_down,
            self.weight_scale_down,
            intermediate=self.intermediate,
            k_gate_up=self.k_gate_up,
            k_down=self.k_down,
            out_dtype=x.dtype,
        )


def _make_shared_config(config: ModelConfig, shared_inter: int) -> ModelConfig:
    """A shallow config view with ``moe_intermediate_size`` = the shared expert's
    (possibly smaller) intermediate size, so the shared expert builds with the
    right width while sharing the model's hidden_size."""
    import copy as _copy

    view = _copy.copy(config)
    view.moe_intermediate_size = shared_inter
    return view


class _Qwen35DecoderLayer:
    """One hybrid decoder layer: a pre-norm + (linear OR full attention) + a
    post-norm + MoE block, with the usual residual connections. A 'linear' layer
    runs the Gated-Delta-Net (recurrent, O(1)-per-token state); a 'full' layer
    runs the gated GQA attention (paged-KV, full history)."""

    def __init__(self, config: ModelConfig, device, dtype, layer_id: int, layer_type: str) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.layer_type = layer_type
        # The raw text config carries the per-layer eps (the shared ModelConfig
        # has no rms_norm_eps field -- parse_config stashes it in config.attrs).
        text_cfg = config.attrs.get("text_config", {})
        eps = float(text_cfg.get("rms_norm_eps", 1e-6))
        self.input_layernorm = _RMSNorm(config.hidden_size, eps=eps, device=device, dtype=dtype)
        self.post_attention_layernorm = _RMSNorm(config.hidden_size, eps=eps, device=device, dtype=dtype)
        if layer_type == "linear_attention":
            self.linear_attn = _GatedDeltaNet(
                config,
                device,
                dtype,
                layer_id,
                linear_num_key_heads=config.attrs["text_config"]["linear_num_key_heads"],
                linear_num_value_heads=config.attrs["text_config"]["linear_num_value_heads"],
                linear_key_head_dim=config.attrs["text_config"]["linear_key_head_dim"],
                linear_value_head_dim=config.attrs["text_config"]["linear_value_head_dim"],
                linear_conv_kernel_dim=config.attrs["text_config"]["linear_conv_kernel_dim"],
                eps=eps,
            )
            self.self_attn = None
        else:
            self.linear_attn = None
            self.self_attn = _Qwen35Attention(
                config,
                device,
                dtype,
                layer_id,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.attrs.get("head_dim", config.hidden_size // config.num_attention_heads),
                partial_rotary_factor=config.attrs["text_config"].get("partial_rotary_factor", 1.0),
                rope_theta=config.attrs.get("rope_theta", 10000.0),
                eps=eps,
            )
        self.mlp = _Qwen35MoE(config, device, dtype, layer_id)

    def forward(self, hidden_states, positions, table_idx, ctx, batch, linear_slot_idx=None) -> torch.Tensor:
        residual = hidden_states
        if self.linear_attn is not None:
            hidden_states = self.linear_attn(
                self.input_layernorm(hidden_states), positions, table_idx, ctx, batch,
                linear_slot_idx=linear_slot_idx,
            )
        else:
            hidden_states = self.self_attn(
                self.input_layernorm(hidden_states), positions, table_idx, ctx, batch
            )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = residual + self.mlp(
            self.post_attention_layernorm(hidden_states), table_idx, ctx, batch
        )
        return hidden_states


class Qwen3_5MoEForCausalLM:
    """The Qwen3.5/3.6 hybrid MoE model: embeddings + 40 hybrid layers (30
    linear Gated-Delta-Net + 10 full gated GQA, every layer MoE) + final norm +
    lm_head. It owns the linear-state pool (recurrent + conv states, lazily
    grown per request slot) and, per step, runs each request's token slice
    through all layers, keeping the last position's logits.

    A genuine ``nn.Module`` so its parameters are real registered nn.Parameters
    the loader resolves via ``model.named_parameters()``.
    """

    def __init__(self, config: ModelConfig, device=None) -> None:
        # Plain (baseless) constructor. The runtime rebind (_ensure_torch) wraps
        # this class in a real nn.Module subclass whose wrapper __init__ runs
        # nn.Module.__init__(self) (initialising _parameters / _modules) and
        # THEN calls this method. We must NOT call super().__init__() here:
        # super() binds to the plain class (MRO [plain, object]), so it would
        # re-enter this method (RecursionError) instead of reaching
        # nn.Module.__init__ (which the wrapper already did).
        self.config = config
        if device is None:
            device = torch.device("xpu") if _xpu_available() else torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        dtype = getattr(config, "dtype", None) or torch.bfloat16
        self.dtype = dtype
        vocab_size = getattr(config, "vocab_size", 256)
        hidden_size = getattr(config, "hidden_size", 256)
        num_layers = getattr(config, "num_layers", 0)
        text_cfg = config.attrs.get("text_config", {})
        # The raw text config carries the per-layer hybrid layout and the norm eps.
        self.rms_norm_eps = float(text_cfg.get("rms_norm_eps", 1e-6))
        # layer_types: 'linear_attention' or 'full_attention' per layer.
        layer_types = text_cfg.get("layer_types")
        if layer_types is None:
            interval = int(text_cfg.get("full_attention_interval", 4))
            layer_types = [
                "full_attention" if (i + 1) % interval == 0 else "linear_attention"
                for i in range(num_layers)
            ]
        self.layer_types = layer_types
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            _Qwen35DecoderLayer(config, device, dtype, layer_id=i, layer_type=layer_types[i])
            for i in range(num_layers)
        )
        self.norm = _RMSNorm(hidden_size, eps=self.rms_norm_eps, device=device, dtype=dtype)
        # Pure-torch replicated Linear (issue-24 WP6); weight [vocab, hidden],
        # the same shape the checkpoint's ``lm_head.weight`` carries.
        from freetoken.layers import LinearReplicated

        self.lm_head = LinearReplicated(hidden_size, vocab_size, has_bias=False, dtype=dtype)

        # The MoE offload wiring (ADR 0002, issue #8): when offload OR the CPU
        # backend is on, the routed experts are never device-resident and are read
        # from the host banks the loader attaches (self.moe_cache / self.moe_layer_id);
        # the linear layers read their per-request state from the linear-state pool
        # (self.linear_state_pool). The CPU backend (use_cpu_moe) is additionally
        # flagged so the block's forward runs the expert GEMM on the host instead of
        # streaming experts to the XPU.
        self.moe_offload = (
            (
                bool(getattr(config, "use_offload_moe", False))
                or bool(getattr(config, "use_cpu_moe", False))
                or bool(getattr(config, "use_hybrid", False))
            )
            and bool(getattr(config, "is_moe", False))
        )
        self.moe_cache = None
        self.moe_layer_id = None
        # The linear-state pool: one recurrent + conv state per request slot, per
        # linear layer. Sized for the max running requests; the model (not the
        # engine) owns it because it knows the layer count + per-request shapes.
        # The engine assigns each request a linear_slot_idx at admission and points
        # ctx.linear_state_pool at this pool (see forward()); the pool lazily grows
        # for any slot index beyond the initial size. max_running_req is an engine
        # (not checkpoint) knob, so it is stashed in config.attrs by the engine.
        max_slots = int(config.attrs.get("max_running_req") or 8)
        self._num_slots = max_slots
        self.linear_state_pool = _LinearStatePool()
        self._register_linear_pool(max_slots)
        if self.device.type != "cpu":
            self.to(self.device)

    def _register_linear_pool(self, num_slots: int) -> None:
        """(Re)size the linear-state pool for ``num_slots`` requests."""
        self.linear_state_pool = _LinearStatePool()
        for layer in self.layers:
            if layer.linear_attn is not None:
                ln = layer.linear_attn
                self.linear_state_pool.register(
                    ln.layer_id,
                    num_slots,
                    (ln.num_v_heads, ln.head_k_dim, ln.head_v_dim),
                    (ln.conv_dim, ln.conv_kernel - 1),
                    self.device,
                    self.dtype,
                )

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, out_loc: torch.Tensor) -> torch.Tensor:
        """Run one engine step; return the **last-position** logits ``[bs, V]``.

        ``input_ids`` / ``positions`` / ``out_loc`` are ``[num_tokens]`` device
        tensors (set by the engine on the global ``Batch``). For decode
        ``num_tokens == bs`` so the last row of each request is its next-token
        logits. Returns ``[bs, vocab_size]``.
        """
        from freetoken.core import get_global_ctx

        ctx = get_global_ctx()
        batch = ctx.batch
        reqs = batch.reqs
        num_tokens = input_ids.shape[0]

        # The linear layers read per-request recurrent state from the pool.
        # Default: the model's own pool, one slot per table_idx (1:1, no
        # ping-pong). A hybrid engine running with prefix caching on
        # (issue `semantic-cache-e2e`, #172) instead assigns a real
        # ping-pong-capable LinearStatePool (freetoken.kvcache.
        # linear_state_pool) to ctx.linear_state_pool, and a real
        # free-list-allocated req.linear_slot_idx, BEFORE the first
        # forward -- both are left alone here when already set, so that
        # path's own slot lifetime (allocated at admission, freed at
        # completion) is authoritative rather than being reset every step.
        if ctx.linear_state_pool is None:
            ctx.linear_state_pool = self.linear_state_pool
        for req in reqs:
            if req.linear_slot_idx is None:
                req.linear_slot_idx = req.table_idx

        hidden = self.embed_tokens(input_ids)  # [num_tokens, hidden]
        out = torch.empty((batch.size, self.config.hidden_size), device=hidden.device, dtype=hidden.dtype)

        offset = 0
        extend_lens = batch.extend_lens
        if extend_lens is None:
            prefill = batch.is_prefill or (num_tokens > batch.size)
            extend_lens = torch.tensor([req.extend_len if prefill else 1 for req in reqs], device=hidden.device)
        # A decode batch is uniform (the scheduler never mixes phases within
        # one batch -- see qwen3_moe's forward() for the same fix and its
        # rationale, issue #15's XpuGraphRunner work), so skip the
        # extend_lens[i] device->host sync per request when decoding: every
        # request contributes exactly one new token.
        is_decode_batch = batch.phase == "decode"
        for i, req in enumerate(reqs):
            ext = 1 if is_decode_batch else int(extend_lens[i])
            token_slice = slice(offset, offset + ext)
            h = hidden[token_slice]
            pos = positions[token_slice]
            for layer in self.layers:
                h = layer(h, pos, req.table_idx, ctx, batch, linear_slot_idx=req.linear_slot_idx)
            # Keep only the last position of this request (next-token logits).
            out[i] = self.norm(h)[-1]
            offset += ext

        return self.lm_head(out)
