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
    # Lazy torch import: to keep this module importable on a torch-free box.
    _cfg = _cfg_for_path(model_path)
    routed = _expert_source_names(_cfg) if _cfg is not None else None

    for raw_name, tensor in _iter_safetensors(model_path, device=device):
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


def _iter_safetensors(model_path, device):
    from freetoken.models.weight import iter_safetensors

    yield from iter_safetensors(model_path, device=device)


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
        self.out_proj = nn.Linear(self.value_dim, hidden, bias=False, device=device, dtype=dtype)
        # Separate projections (qwen3_5_moe spelling).
        self.in_proj_qkv = nn.Linear(hidden, self.key_dim * 2 + self.value_dim, bias=False, device=device, dtype=dtype)
        self.in_proj_z = nn.Linear(hidden, self.value_dim, bias=False, device=device, dtype=dtype)
        self.in_proj_b = nn.Linear(hidden, self.num_v_heads, bias=False, device=device, dtype=dtype)
        self.in_proj_a = nn.Linear(hidden, self.num_v_heads, bias=False, device=device, dtype=dtype)

    def _conv(self, mixed_qkv: torch.Tensor, slot, hidden_dtype) -> torch.Tensor:
        """Causal depthwise conv over the mixed qkv; updates the conv ring.

        ``mixed_qkv`` is ``[B, conv_dim, T]``. ``slot`` is this request's
        :class:`_LinearStateSlot` (or None: a self-contained ring per call, used
        by the standalone math path with no pool). Returns ``[B, conv_dim, T]``.
        The ring holds the trailing (kernel-1) positions; the new ring is the
        trailing (kernel-1) of the concatenated (old ring + new tokens) sequence,
        and it is written back in place so the next step reads it.
        """
        B, C, T = mixed_qkv.shape
        w = self.conv1d.weight  # [C, 1, K] (groups=C -> each channel is 1 input)
        ring_len = self.conv_kernel - 1
        if slot is None:
            padded = torch.cat(
                [torch.zeros(B, C, ring_len, device=mixed_qkv.device, dtype=mixed_qkv.dtype), mixed_qkv],
                dim=-1,
            )
            return F.conv1d(padded, w, padding=0, groups=C)[:, :, :T]
        # Old ring (K-1) + new tokens -> conv output for the new tokens only; the
        # new ring is the trailing (K-1) of that concatenation.
        seq = torch.cat([slot.conv_state.unsqueeze(0), mixed_qkv], dim=-1)  # [B, C, K-1+T]
        out = F.conv1d(seq, w, padding=0, groups=C)[:, :, -T:]
        if T > 0:
            slot.conv_state.copy_(seq[:, :, -ring_len:][0])
        return out

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

    def forward(self, hidden_states, positions, table_idx, ctx, batch) -> torch.Tensor:
        # ``hidden_states`` is this request's token slice [T, H] (the decoder
        # layer hands it a 2-D per-request slice, mirroring qwen3_moe); positions
        # is that slice's [T]. T is this request's new-token count.
        T = hidden_states.shape[0]
        slot = None
        pool = ctx.linear_state_pool
        if pool is not None and self.layer_id in pool:
            slot = pool.get(self.layer_id, table_idx)
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
        self.q_proj = nn.Linear(self.hidden, num_heads * head_dim * 2, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(self.hidden, num_kv_heads * head_dim, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(num_kv_heads * head_dim, self.hidden, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(num_heads * head_dim, self.hidden, bias=False, device=device, dtype=dtype)
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
        # Append this request's K/V to the pool at its positions (identity table:
        # out_loc == position under the identity page table, as in qwen3_moe).
        ctx.kv_cache.write_kv(k, v, positions)
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


def _expert_compute(gate_w, up_w, down_w, x):
    """Run one expert on a [t, H] input using *detached* projection weights
    (the offload bank rows, in [out, in] weight orientation): gate/up are
    [I, H], down is [H, I], so each projection is ``x @ w.t()``. Returns
    ``down(silu(gate(x)) * up(x))``."""
    inter = gate_w.shape[0]
    return (F.silu(x @ gate_w.t()) * (x @ up_w.t())) @ down_w.t()


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
        self.expert_inter = config.moe_intermediate_size
        self.shared_inter = int(config.attrs.get("text_config", {}).get("shared_expert_intermediate_size", config.moe_intermediate_size))
        self.gate = nn.Linear(self.hidden, self.num_experts, bias=False, device=device, dtype=dtype)
        if self.use_offload:
            self.experts = None
        else:
            self.experts = nn.ModuleList(_Qwen35Expert(config, device, dtype) for _ in range(self.num_experts))
        self.shared_expert = _Qwen35Expert(
            _make_shared_config(config, self.shared_inter), device, dtype
        )
        self.shared_expert_gate = nn.Linear(self.hidden, 1, bias=False, device=device, dtype=dtype)

    def forward(self, hidden_states, table_idx, ctx, batch) -> torch.Tensor:
        in_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, in_shape[-1])  # [n_tok, H]
        gate_logits = self.gate(flat)  # [n_tok, num_experts]
        router_probs = F.softmax(gate_logits, dtype=torch.float32, dim=-1)
        top_w, top_idx = torch.topk(router_probs, self.top_k, dim=-1)  # [n, k]
        top_w = (top_w / top_w.sum(dim=-1, keepdim=True)).to(flat.dtype)
        # The always-on shared expert (dense, on the device), gated per-token.
        shared_out = F.sigmoid(self.shared_expert_gate(flat)) * self.shared_expert(flat)
        # Routed experts: in-VRAM modules, or the host-offload LRU slot pool.
        if self.use_offload:
            routed_out = self._forward_offload(flat, top_idx, top_w, ctx, batch)
        else:
            routed_out = self._forward_inram(flat, top_idx, top_w)
        return (routed_out + shared_out).view(in_shape)

    def _forward_inram(self, flat, top_idx, top_w):
        out = torch.zeros_like(flat)
        for e in range(self.num_experts):
            for slot in range(self.top_k):
                sel = top_idx[:, slot] == e
                if not sel.any():
                    continue
                out[sel] += top_w[sel, slot, None] * self.experts[e](flat[sel])
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
        model = ctx.model
        layer_id = model.moe_layer_id[self.layer_id]
        cache = model.moe_cache
        is_prefill = (batch is not None and batch.is_prefill) or flat.shape[0] > 1
        is_xpu = bool(getattr(cache, "is_xpu", False))
        B, k = top_idx.shape
        # 1) Snapshot the routed *expert* ids on the host (the topk output is a
        #    fresh tensor; reading it back here is a clean D2H of its final value).
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
        gu, dn = cache.bank_views()  # ([S, 2I, H], [S, H, I])
        intermediate = int(model.config.moe_intermediate_size)
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
        groups: list[tuple[int, int, list[int]]] = []
        for j in range(k):
            by_slot: dict[int, list[int]] = {}
            for i in range(B):
                if bool(valid_cpu[i, j]):
                    by_slot.setdefault(int(routed_cpu[i, j]), []).append(i)
            groups.extend((j, s_i, rows) for s_i, rows in by_slot.items())
        # 5) Push the routing to the XPU (one op). Validity is already encoded in
        #    which rows appear in `groups`, so no separate valid tensor is pushed.
        routed_dev = routed_cpu.to(dev)
        # 6) Gather per slot on the XPU using host-computed INTEGER indices:
        #    static shapes, no nonzero(), no implicit D2H anywhere in the loop.
        #    index_add_ accumulates exactly as `out[sub] += ...` did (and handles
        #    duplicate indices, though rows are unique within a (j, s_i) group).
        for j, s_i, rows in groups:
            idx = torch.tensor(rows, dtype=torch.long, device=dev)
            y = top_w.index_select(0, idx)[:, j, None] * _expert_compute(
                gu[s_i, 0:intermediate],
                gu[s_i, intermediate : 2 * intermediate],
                dn[s_i],
                flat.index_select(0, idx),
            )
            out.index_add_(0, idx, y)
        # 7) Leave top_idx in the slot-id contract (harmless; nothing downstream
        #    reads it, but keeps the "top_idx holds slot ids" invariant true).
        top_idx.copy_(routed_dev)
        return out


class _Qwen35Expert:
    """A single MoE expert: gate/up/down projections (SwiGLU). Used by the
    in-VRAM path (and the always-on shared expert, which is dense)."""

    def __init__(self, config: ModelConfig, device, dtype) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


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

    def forward(self, hidden_states, positions, table_idx, ctx, batch) -> torch.Tensor:
        residual = hidden_states
        if self.linear_attn is not None:
            hidden_states = self.linear_attn(
                self.input_layernorm(hidden_states), positions, table_idx, ctx, batch
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
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False, device=device, dtype=dtype)

        # The MoE offload wiring (ADR 0002): when offload is on the routed experts
        # are never device-resident and are read from the LRU slot pool the loader
        # attaches (self.moe_cache / self.moe_layer_id); the linear layers read
        # their per-request state from the linear-state pool (self.linear_state_pool).
        self.moe_offload = bool(getattr(config, "use_offload_moe", False)) and bool(getattr(config, "is_moe", False))
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

        # The linear layers read per-request recurrent state from the pool; the
        # engine assigns each request a linear_slot_idx (== table_idx here) and
        # points ctx.linear_state_pool at this model's pool.
        ctx.linear_state_pool = self.linear_state_pool
        for req in reqs:
            req.linear_slot_idx = req.table_idx

        hidden = self.embed_tokens(input_ids)  # [num_tokens, hidden]
        out = torch.empty((batch.size, self.config.hidden_size), device=hidden.device, dtype=hidden.dtype)

        offset = 0
        extend_lens = batch.extend_lens
        if extend_lens is None:
            prefill = batch.is_prefill or (num_tokens > batch.size)
            extend_lens = torch.tensor([req.extend_len if prefill else 1 for req in reqs], device=hidden.device)
        for i, req in enumerate(reqs):
            ext = int(extend_lens[i])
            token_slice = slice(offset, offset + ext)
            h = hidden[token_slice]
            pos = positions[token_slice]
            for layer in self.layers:
                h = layer(h, pos, req.table_idx, ctx, batch)
            # Keep only the last position of this request (next-token logits).
            out[i] = self.norm(h)[-1]
            offset += ext

        return self.lm_head(out)
