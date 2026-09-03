"""Checkpoint loader: build a model on the XPU and place its weights.

Upstream NVIDIA path: python/freetoken/models/loader.py
Fill in: GitHub issue `models-loader` (see docs/architecture.md).

``load_model`` is the top-level entry point the ``ft serve`` spine and the
engine call. It resolves the checkpoint's model spec, parses its config, builds
the model on the destination device (the XPU), and loads the dense weights from
the safetensors shards. For MoE checkpoints it also builds the host offload
banks for the expert weights (which do not fit on the XPU) and attaches them so
the engine can serve experts on demand.

The dense weights are consumed through the *model's* ``iter_weights`` and written
back into the model's own parameters. The engine's ``forward`` is a stub
(``#14``), so a freshly-loaded model is not yet runnable -- but the loader's job
is to place weights and build the parameter set, which is what this implements.
"""
from __future__ import annotations

import importlib

import torch

from freetoken.models.register import _load_attr, get_model_class, get_model_spec
from freetoken.models.weight import load_moe_expert_sources, load_weight
from freetoken.utils import cached_load_hf_config
from freetoken.utils.arch import is_xpu_available

# Architectures whose parse_config accepts the ``use_offload_moe`` /
# ``use_cpu_moe`` kwargs (ADR 0002, issue #8). These are the only ones that
# build router-only MoE blocks (non-device-resident experts) when offloaded or
# run the routed-expert GEMM on the host CPU when the backend is ``cpu``. Every
# other architecture's parse_config is a no-op ``(*args, **kwargs)`` stub that
# raises unimplemented() on any kwarg, so load_model must only re-parse for these.
_CPU_MOE_CAPABLE_ARCHS = {
    "Qwen3MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5ForConditionalGeneration",
}


def load_model(
    model_path: str,
    device: torch.device | str | None = None,
    *,
    dtype: torch.dtype | None = None,
    dummy: bool = False,
    moe_backend: str | None = None,
    moe_cpu_layers: str | None = None,
    moe_cache_size: int | None = None,
) -> tuple:
    """Load a checkpoint onto ``device`` (defaults to the XPU when available).

    Returns ``(model, expert_sources)`` where ``model`` is the instantiated model
    with its dense parameters populated on ``device``, and ``expert_sources`` is
    the per-layer MoE bank tuple from :func:`load_moe_expert_sources` (empty for
    a dense model). With ``dummy=True`` the expert banks are fabricated from the
    config (offline / CPU-testable) and no checkpoint is read.

    ``moe_backend`` (ADR 0002): when it names the host-offload backend the
    MoE experts are never XPU-resident -- the model builds router-only MoE
    blocks and the loader attaches the LRU slot pool (``OffloadMoeCache``) wired
    to the host banks, so the forward streams the routed experts on demand.

    ``moe_cache_size`` (issue #16): the number of slots the offload cache's
    device pool should hold. ``None`` (default) keeps the loader's conventional
    sizing (``num_experts + max(2, num_moe)``); a positive int pins the size
    (e.g. a value planned off the VRAM budget by the engine).
    """
    if device is None:
        device = torch.device("xpu") if is_xpu_available() else torch.device("cpu")
    if isinstance(device, str):
        device = torch.device(device)
    if dtype is None:
        dtype = torch.bfloat16

    hf_config = cached_load_hf_config(model_path)
    spec = get_model_spec(hf_config.architectures[0])
    # The offload flag is baked into the model config at parse time, but
    # resolving ``moe_backend="auto"`` (the default) needs to know whether the
    # model is a MoE -- which the config only exposes *after* parsing. So parse
    # once without offload, inspect the result, resolve the backend, and re-parse
    # only if the resolution flips the flag (the common ``auto``-on-XPU path).
    parse_config = _load_attr(spec.module, spec.parse_config)
    model_config = parse_config(hf_config, use_offload_moe=False, model_path=model_path)
    from freetoken.moe import parse_moe_cpu_layers, resolve_moe_backend
    from freetoken.moe.bench_profile import quant_format_for_dtype

    is_moe_early = bool(getattr(model_config, "is_moe", False))
    # Only an explicit "auto" (the EngineConfig default) is resolved here; an
    # explicit name or None is honored as-is so existing call sites that pass
    # None for the in-VRAM path are unaffected.
    if moe_backend == "auto":
        # Issue #9: pass the model's expert format so ``auto`` can consult the
        # ``ft bench bw`` profile and upgrade the offload default to hybrid when
        # the box's measured bandwidths say the CPU beats PCIe. Quantized
        # checkpoints expose ``quant_format``; a bf16 checkpoint has none, so fall
        # back to the dtype-derived format (bfloat16 -> "bf16") -- otherwise a
        # bf16 hero would never resolve the profile and ``auto`` would stay on
        # offload even when hybrid is benched-better. A None result keeps the
        # resolve's safe default (offload).
        moe_backend = resolve_moe_backend(
            moe_backend,
            is_moe=is_moe_early,
            quant_format=getattr(model_config, "quant_format", None)
            or quant_format_for_dtype(getattr(model_config, "dtype", None)),
        )
    # "offload" streams activated experts to the device; "cpu" runs the expert
    # GEMM on the host (issue #8); "hybrid" splits each decode step's misses
    # between the two halves by the bench profile's fetch fraction (issue #9).
    # All three keep the experts non-device-resident, so all flag use_offload_moe
    # (the block's "no resident experts" gate); the CPU path is additionally
    # flagged use_cpu_moe (whole-layer CPU) and the hybrid path use_hybrid, so
    # the block dispatches each to the right forward.
    use_offload = moe_backend in ("offload", "cpu", "hybrid")
    use_cpu = moe_backend == "cpu"
    use_hybrid = moe_backend == "hybrid"
    # Issue #8: the --moe-cpu-layers spec is resolved to the concrete MoE-layer
    # indices that compute on the CPU (None == all MoE layers on the CPU, the
    # --moe-backend=cpu default; an explicit list/carve-out is a subset). It is
    # stored on the model (model.moe_cpu_moe_layers) and read per-layer by the
    # block's forward, so a cpu/offload/hybrid backend can mix CPU + XPU-offload
    # MoE layers in one model. (The spec is parsed here -- torch-free -- so the
    # CPU venv resolves it without importing torch.) num_moe_layers is derived in
    # ModelConfig.__post_init__ only for is_moe configs; the first parse (before
    # is_moe is set) can leave it None, in which case the partition is "no CPU
    # layers" (a dense model has none).
    moe_cpu_moe_layers = parse_moe_cpu_layers(
        moe_cpu_layers, int(getattr(model_config, "num_moe_layers", 0) or 0)
    )
    # Re-parse only for architectures whose parse_config takes the offload / cpu
    # MoE flags (the Qwen3 MoE family). Every other architecture's parse_config is
    # a no-op (*args, **kwargs) stub that raises unimplemented() the moment it is
    # handed kwargs, and non-MoE models never flip these flags anyway -- so the
    # re-parse must not fire for them, or load_model crashes on every dense model.
    _arch = hf_config.architectures[0] if getattr(hf_config, "architectures", None) else None
    if _arch in _CPU_MOE_CAPABLE_ARCHS and (
        use_offload != bool(getattr(model_config, "use_offload_moe", False))
        or use_cpu != bool(getattr(model_config, "use_cpu_moe", False))
        or use_hybrid != bool(getattr(model_config, "use_hybrid", False))
        or moe_cpu_layers != getattr(model_config, "moe_cpu_layers", None)
    ):
        model_config = parse_config(
            hf_config,
            use_offload_moe=use_offload,
            use_cpu_moe=use_cpu,
            use_hybrid=use_hybrid,
            moe_cpu_layers=moe_cpu_layers,
            model_path=model_path,
        )
    # Build the model *on this device* (the loader already resolved it): an
    # explicit device wins, and only a None device lets the model default to
    # the XPU. Without this the model would re-default to the XPU and ignore
    # the loader's device (e.g. a CPU test on an XPU box).
    #
    # The model builds its modules in ``model_config.dtype``; stamp the loader's
    # effective dtype onto it so the modules and the streamed weights share a
    # dtype (no bf16-module / fp32-weight mismatch when the engine pins one).
    object.__setattr__(model_config, "dtype", dtype)
    # Qwen3.5/3.6 keeps its forward classes torch-free at import (the CPU venv
    # parses the config without torch); the model rebinds them to real
    # nn.Module subclasses lazily. The rebind must complete BEFORE the model is
    # instantiated -- its __init__ uses an explicit super() that resolves
    # nn.Module only if the class already carries the nn.Module base -- so
    # trigger it here, in the loader, not in the constructor. Only adapters that
    # expose _ensure_torch need this (others import torch at module scope).
    _mod = importlib.import_module(spec.module)
    _ensure = getattr(_mod, "_ensure_torch", None)
    if callable(_ensure):
        _ensure()
    model = get_model_class(hf_config.architectures[0], model_config, device=device)

    is_moe = bool(getattr(model_config, "is_moe", False))
    # Issue #8: record the per-layer CPU/offload partition on the model (read by
    # the block's forward). Set unconditionally -- even the in-VRAM path gets it
    # (None == "no CPU layers"), so the forward's getattr is always safe.
    model.moe_cpu_moe_layers = moe_cpu_moe_layers if is_moe else None
    # The resolved MoE backend, recorded on the model for the block's forward
    # (``_is_cpu_layer`` gates on it) and the engine. Set unconditionally -- the
    # in-VRAM path (moe_backend None / "fused") gets it too -- so a block that
    # never sees _attach_offload_cache still has a valid gate value.
    model.moe_backend = moe_backend
    # Issue #9 (moe-hybrid): the per-step fetch fraction (share of a decode step's
    # expert misses PCIe-fetched vs computed on the host CPU). Read from the
    # ``ft bench bw`` profile for this expert format; 0.0 (pure offload) when no
    # usable profile exists, so a box that never benched degrades cleanly. Set
    # unconditionally so the block's forward getattr is always safe.
    from freetoken.moe.bench_profile import load_hybrid_fetch_fraction, quant_format_for_dtype

    # The expert format to key the bench profile on. Quantized checkpoints expose
    # it as ``model_config.quant_format``; a bf16 checkpoint has none, so derive
    # the expert format from the effective dtype (the bench profile is keyed by
    # dtype, and a bf16 model's experts are bf16). Without this the fraction
    # resolves to ``None`` and a bf16 hybrid model silently fetches 0.0 (never
    # fetches). ``quant_format_for_dtype`` maps the long torch dtype spellings to
    # the short bench keys (bfloat16 -> bf16) and folds in the fp8/mxfp4 aliases.
    _qf = getattr(model_config, "quant_format", None) or quant_format_for_dtype(
        getattr(model_config, "dtype", None)
    )
    model.moe_hybrid_fetch_fraction = (
        float(load_hybrid_fetch_fraction(_qf) or 0.0) if use_hybrid else 0.0
    )
    offload = is_moe and use_offload and moe_backend in ("offload", "cpu", "hybrid")
    if dummy:
        # Offline path: no checkpoint is read, so the model's MoE experts must
        # come from fabricated banks. To make this *reproducible* (the engine's
        # greedy output must be a pure function of the config, not of the
        # process's prior RNG state -- see _seed_dummy_experts), the model's
        # expert modules are zeroed first, the RNG is re-seeded from a hash of
        # the config, the banks are fabricated, and then copied into the model.
        # (The dense weights are not read in this path; the reference test
        # fabricates a tiny checkpoint but only the dummy experts are consumed.)
        if is_moe:
            # Both backends need the dummy path reproducible: the in-VRAM model
            # owns the expert modules (seeded here), while the offload model owns
            # only the dense params (experts live in the host banks, seeded
            # separately by load_moe_expert_sources). _seed_dummy_experts zeroes
            # every non-expert param and re-seeds the RNG from the config hash,
            # so the dense weights -- and hence the greedy output -- are a pure
            # function of the config regardless of the process's prior RNG state.
            # It is safe on the offload model too: it only iterates
            # named_parameters (the dense set there; no expert params exist), so
            # nothing offload-specific is touched.
            _seed_dummy_experts(model)
            gate_up_banks, down_banks = load_moe_expert_sources(model_path, dtype=dtype, dummy=True)
            if offload:
                _attach_offload_cache(
                    model,
                    model_config,
                    device,
                    gate_up_banks,
                    down_banks,
                    moe_backend=moe_backend,
                    moe_cache_size=moe_cache_size,
                    model_path=model_path,
                )
            else:
                _place_expert_weights(model, gate_up_banks, down_banks)
            expert_sources = (gate_up_banks, down_banks)
        else:
            expert_sources = ([], [])
    else:
        # Real path: stream the dense checkpoint weights onto ``device`` and, for
        # a MoE checkpoint, build the per-layer host offload banks for the experts
        # (which do not fit on the XPU and are served to the engine on demand).
        #
        # Both backends read the *same* checkpoint here, so the dense weights the
        # reference and the offload model receive are byte-identical -- the
        # offload forward then differs from the in-VRAM one only in how the expert
        # weights are transported (host banks -> LRU slots), never in the values.
        # (No dummy seeding: a real checkpoint is a stable source of the dense
        # weights, so re-seeding the RNG here would be unnecessary and would make
        # the model depend on process state instead of the checkpoint.)
        for name, tensor in load_weight(model_path, device, include_moe_experts=False):
            _place_dense(model, name, tensor)
        if is_moe:
            # Thread the backend so the banks land on the device the engine
            # wants (host for offload/hybrid/cpu -- see load_moe_expert_sources),
            # not the XPU that parse_config's defaults would pick.
            gate_up_banks, down_banks = load_moe_expert_sources(
                model_path, dtype=dtype, moe_backend=moe_backend
            )
            if offload:
                _attach_offload_cache(
                    model,
                    model_config,
                    device,
                    gate_up_banks,
                    down_banks,
                    moe_backend=moe_backend,
                    moe_cache_size=moe_cache_size,
                    model_path=model_path,
                )
            else:
                _place_expert_weights(model, gate_up_banks, down_banks)
            expert_sources = (gate_up_banks, down_banks)
        else:
            expert_sources = ([], [])
    return model, expert_sources


def _seed_dummy_experts(model) -> None:
    """Make the offline dummy-expert path reproducible regardless of RNG state.

    Building the model's expert ``nn.Linear`` modules consumes the global RNG
    during ``__init__``; the RNG offset of that construction depends on how much
    random state the *process* consumed earlier, so the same dummy seed would
    land the fabricated banks at a different offset from run to run -> different
    weights -> non-deterministic greedy output.

    Fix: zero the expert parameters (so any expert the banks do not cover still
    contributes a deterministic zero) and re-seed the global RNG from a stable
    hash of the model config, so the bank fabricating draw is a pure function of
    the config, not the process.
    """
    import hashlib

    from freetoken.models.weight import _num_moe_layers

    num_experts = int(getattr(model.config, "num_experts", 0) or 0)
    num_moe = _num_moe_layers(model.config)
    hidden = int(getattr(model.config, "hidden_size", 0) or 0)
    moe_inter = int(getattr(model.config, "moe_intermediate_size", 0) or 0)
    seed = int.from_bytes(
        hashlib.md5(
            f"{model.config.architectures}{hidden}{num_moe}{num_experts}{moe_inter}".encode()
        ).digest()[:8],
        "big",
    ) % (2**32)
    if not num_experts:
        return
    # Zero every weight the dummy path does NOT read back from a checkpoint:
    # the MoE expert params (filled from the fabricated banks below) AND the
    # dense params (embeddings / attention / norms / lm_head), which are left at
    # their (process-dependent) random init otherwise. With the dense weights
    # zeroed and the experts seeded from the config hash, the forward becomes a
    # pure function of the config -- reproducible regardless of the process's
    # prior RNG state. (Zero_ does not consume the RNG.)
    with torch.no_grad():
        for name, param in list(model.named_parameters()):
            if ".experts." in name:
                param.zero_()
        dense = {n: p for n, p in model.named_parameters() if ".experts." not in n}
        for param in dense.values():
            param.zero_()
    # Re-seed *after* the zeroing so the fabricating randn draw is a pure
    # function of the config hash (the zeroing above does not consume the RNG).
    torch.manual_seed(seed)


def _moe_layers(config) -> list[int]:
    """Layer ids that carry MoE experts (all but the leading dense ones)."""
    total = int(getattr(config, "num_layers", 0) or 0)
    first_dense = int(getattr(config, "first_k_dense_replace", 0) or 0)
    return list(range(first_dense, total))


def _place_expert_weights(model, gate_up_banks, down_banks) -> None:
    """Copy the per-layer expert banks into the model's per-expert modules.

    The model owns a ``_Qwen3Expert`` per (moe layer, expert) with ``gate_proj``
    / ``up_proj`` / ``down_proj``; the loader's banks are stacked ``[num_experts,
    ...]`` per MoE layer. Writing them in keeps the model's forward using the
    *same* expert weights the banks describe -- which also makes the dummy path
    deterministic (a fixed seed fabricates fixed banks -> fixed model weights).
    Layers with no bank (e.g. the leading dense layers) are left as-is.
    """
    import torch

    # The gate_up bank is [num_experts, 2*intermediate, hidden]: for each expert
    # row the *first* ``intermediate_size`` rows are the gate projection and the
    # next ``intermediate_size`` are the up projection (gate then up, concatenated
    # on dim 1 -- the layout the repacker / dummy fabricator both produce). The
    # down bank is [num_experts, hidden, intermediate].
    intermediate = int(getattr(model.config, "moe_intermediate_size", 0))
    for layer_id in _moe_layers(model.config):
        moe = getattr(getattr(model, "layers", [None] * (layer_id + 1))[layer_id], "mlp", None)
        experts = getattr(moe, "experts", None)
        if experts is None:
            continue
        # The dummy path wraps banks in _PlainBank (exposing .tensor); the real
        # streamed path returns the raw stacked tensors. Normalize both.
        gu = gate_up_banks[layer_id].tensor if hasattr(gate_up_banks[layer_id], "tensor") else gate_up_banks[layer_id]
        dn = down_banks[layer_id].tensor if hasattr(down_banks[layer_id], "tensor") else down_banks[layer_id]
        for e in range(len(experts)):
            with torch.no_grad():
                experts[e].gate_proj.weight.copy_(gu[e, 0:intermediate])
                experts[e].up_proj.weight.copy_(gu[e, intermediate : 2 * intermediate])
                experts[e].down_proj.weight.copy_(dn[e])


def _attach_offload_cache(
    model,
    model_config,
    device: torch.device,
    gate_up_banks,
    down_banks,
    moe_backend: str = "offload",
    moe_cache_size: int | None = None,
    model_path: str | None = None,
) -> None:
    """Build the LRU slot pool and wire it into the offload model (ADR 0002).

    The MoE experts are never XPU-resident on this path. The host expert banks
    (``gate_up_banks`` / ``down_banks``, one ``[num_experts, ...]`` per MoE
    layer, on host memory) are attached to an ``OffloadMoeCache`` that owns a
    small device slot pool; the forward (``_Qwen3MoE._forward_offload``) routes
    each layer's routed experts through it -- ``materialize_layer`` on prefill,
    ``ensure_experts`` (timestamp LRU) + ``copy_missing`` on decode.

    The cache is indexed by *MoE-layer index* (0-based among the MoE layers),
    while the model's blocks are indexed by *absolute layer id*, so we also give
    the model the ``moe_layer_id`` map (layer_id -> MoE index) the forward uses.
    """
    from freetoken.models.weight import Fp8BlockExpertBank, GptqExpertBank, Int8ExpertBank, MxfpExpertBank
    from freetoken.moe.offload_cache import OffloadMoeCache

    num_experts = int(getattr(model_config, "num_experts", 0) or 0)
    moe_layers = _moe_layers(model_config)
    num_moe = len(moe_layers)
    if not num_moe:
        return
    # The pool is a single global LRU shared by all MoE layers (the only place
    # the 61 GB of experts fits): it holds the *current* layer's whole expert set
    # (num_experts slots) plus a small slack of decode slots so a decode miss can
    # be placed without immediately evicting a just-routed expert of the same
    # step. By default the pool is sized off the layer count so it scales with
    # the model (a real Qwen3-30B-A3B has 128 experts / 48 MoE layers); when the
    # engine ran the issue-#16 budget planner it instead passes the VRAM-planned
    # size (``moe_cache_size``), which may be larger (more layers warm at once)
    # but never below ``num_experts`` (a whole MoE layer must still fit).
    if moe_cache_size and moe_cache_size >= num_experts:
        cache_size = moe_cache_size
    else:
        cache_size = num_experts + max(2, num_moe)
    # A GPTQ-quantized checkpoint's banks (issue moe-quant-banks-e2e, #138)
    # are GptqExpertBank, not a plain tensor/_PlainBank -- detected from the
    # bank shape itself (load_moe_expert_sources already dispatched to
    # stream_moe_expert_sources_gptq for these), not re-derived from the
    # checkpoint path here.
    is_gptq = bool(gate_up_banks) and isinstance(gate_up_banks[0], GptqExpertBank)
    # A block-FP8-quantized checkpoint's banks (issue moe-quant-banks-fp8,
    # #152) are Fp8BlockExpertBank, detected the same way -- from the bank
    # shape itself (load_moe_expert_sources already dispatched to
    # stream_moe_expert_sources_fp8 for these), not re-derived from the
    # checkpoint path here.
    is_fp8_block = bool(gate_up_banks) and isinstance(gate_up_banks[0], Fp8BlockExpertBank)
    # An MXFP4-quantized checkpoint's banks (issue moe-quant-banks-mxfp4,
    # #153) are MxfpExpertBank, detected the same way as GPTQ's -- from the
    # bank shape load_moe_expert_sources already dispatched to
    # stream_moe_expert_sources_mxfp4 for.
    is_mxfp4 = bool(gate_up_banks) and isinstance(gate_up_banks[0], MxfpExpertBank)
    # A per-channel-INT8 checkpoint's banks (issue moe-quant-banks-int8, #154)
    # are Int8ExpertBank, detected the same way is_gptq is -- from the bank
    # type itself, not re-derived from the checkpoint path here.
    is_int8 = bool(gate_up_banks) and isinstance(gate_up_banks[0], Int8ExpertBank)
    if is_gptq:
        quant_format = "gptq_int4"
    elif is_fp8_block:
        quant_format = "fp8_block"
    elif is_mxfp4:
        quant_format = "mxfp4"
    elif is_int8:
        quant_format = "int8_channel"
    else:
        quant_format = "bf16"
    cache = OffloadMoeCache(
        num_layers=num_moe,
        num_experts=num_experts,
        cache_size=cache_size,
        device=device,
        quant_format=quant_format,
    )
    if is_mxfp4:
        # Four packed banks (issue moe-quant-banks-mxfp4, #153's
        # _BANK_SCHEMAS["mxfp4"]) -- every bank is one row per expert (no
        # g_idx-equivalent side table; MXFP4's scale is fully local to its
        # own block, never shared across a whole projection), so
        # set_bank_sources' generic per-expert-row contract applies
        # unchanged, same as gptq_int4 below.
        cache.set_bank_sources(
            {
                "blocks_gate_up": [b.blocks for b in gate_up_banks],
                "scales_gate_up": [b.scales for b in gate_up_banks],
                "blocks_down": [b.blocks for b in down_banks],
                "scales_down": [b.scales for b in down_banks],
            }
        )
    elif is_int8:
        # Four packed banks (issue moe-quant-banks-int8, #154's
        # _BANK_SCHEMAS["int8_channel"] -- corrected to compressed-tensors'
        # real pack-quantized format, verified against rj1013/
        # gemma-4-26B-A4B-it_q8; see Int8ExpertBank's own docstring) --
        # every bank is one row per expert, so set_bank_sources' generic
        # per-expert-row contract applies unchanged. K (real logical
        # in-features) is shared across every expert of a projection type
        # (an architecture constant, gate_up's == hidden_size, down's ==
        # moe_intermediate_size) -- set as two plain scalar cache attributes
        # (SlotWeightAccessor refuses to guess them), the same pattern as
        # gptq_group_size below, since K is derivable from the parsed model
        # config rather than needing a per-layer extra_metadata side table.
        cache.set_bank_sources(
            {
                "weight_packed_gate_up": [b.weight_packed for b in gate_up_banks],
                "weight_scale_gate_up": [b.weight_scale for b in gate_up_banks],
                "weight_packed_down": [b.weight_packed for b in down_banks],
                "weight_scale_down": [b.weight_scale for b in down_banks],
            }
        )
        gate_up_ks = {b.k for b in gate_up_banks}
        down_ks = {b.k for b in down_banks}
        if len(gate_up_ks) != 1 or len(down_ks) != 1:
            raise ValueError(
                f"int8_channel: K differs across layers -- gate_up {gate_up_ks}, down {down_ks} "
                "(K is an architecture constant, expected identical across every layer)"
            )
        cache.int8_k_gate_up = next(iter(gate_up_ks))
        cache.int8_k_down = next(iter(down_ks))
    elif is_gptq:
        # Six packed banks (issue moe-quant-banks-schema, #136's schema) --
        # every bank is one row per expert, so set_bank_sources' generic
        # per-expert-row contract applies unchanged. g_idx is NOT a bank (it
        # is shared across every expert of a projection type, see
        # GptqExpertBank's own docstring): it goes through extra_metadata
        # instead, one shared [K] tensor per layer per projection type.
        cache.set_bank_sources(
            {
                "qweight_gate_up": [b.qweight for b in gate_up_banks],
                "qzeros_gate_up": [b.qzeros for b in gate_up_banks],
                "scales_gate_up": [b.scales for b in gate_up_banks],
                "qweight_down": [b.qweight for b in down_banks],
                "qzeros_down": [b.qzeros for b in down_banks],
                "scales_down": [b.scales for b in down_banks],
            }
        )
        cache.set_extra_metadata("g_idx_gate_up", [b.g_idx for b in gate_up_banks])
        cache.set_extra_metadata("g_idx_down", [b.g_idx for b in down_banks])
        # SlotWeightAccessor (#137) refuses to guess this -- the loader (here)
        # is the one place that has both the checkpoint path and the cache.
        if model_path is None:
            raise ValueError("_attach_offload_cache needs model_path to read a GPTQ checkpoint's group_size")
        from freetoken.models.weight import checkpoint_gptq_group_size

        cache.gptq_group_size = checkpoint_gptq_group_size(model_path)
    elif is_fp8_block:
        # Four packed banks (_BANK_SCHEMAS["fp8_block"]) -- every bank is one
        # row per expert, so set_bank_sources' generic per-expert-row contract
        # applies unchanged. Unlike GPTQ there is no shared side tensor to
        # register via set_extra_metadata (see Fp8BlockExpertBank's own
        # docstring): block-FP8's weight_scale_inv is genuinely per-expert.
        cache.set_bank_sources(
            {
                "weight_gate_up": [b.weight for b in gate_up_banks],
                "scale_gate_up": [b.weight_scale_inv for b in gate_up_banks],
                "weight_down": [b.weight for b in down_banks],
                "scale_down": [b.weight_scale_inv for b in down_banks],
            }
        )
    else:
        # The banks are indexed by MoE-layer order (moe_layers), matching the
        # cache's 0-based MoE-layer ids.
        gu = [b.tensor if hasattr(b, "tensor") else b for b in gate_up_banks]
        dn = [b.tensor if hasattr(b, "tensor") else b for b in down_banks]
        cache.set_bank_sources({"gate_up": gu, "down": dn})

    model.moe_cache = cache
    # Record the resolved backend on the model. The block's MoE forward reads
    # this (``_Qwen3MoE.forward``) to route the routed experts through the
    # right path: "cpu" runs the GEMM on the host (issue #8), "offload" streams
    # activated experts to the device. The engine reads the model back, so the
    # loader is the one place that knows which backend it materialized.
    model.moe_backend = moe_backend
    model.moe_layer_id = [0] * len(model.layers)
    for moe_idx, layer_id in enumerate(moe_layers):
        model.moe_layer_id[layer_id] = moe_idx
    model.ctx_moe_cache = cache  # the engine installs this on its Context


def _place_dense(model, name: str, tensor: torch.Tensor) -> None:
    """Place a dense checkpoint tensor into the model's corresponding parameter.

    The checkpoint keys the ``model.`` prefix that the HF layout uses
    (``model.embed_tokens.weight`` / ``model.layers.0.self_attn.q_proj.weight``),
    but the model registers its params *without* that prefix
    (``embed_tokens.weight`` / ``layers.0.self_attn.q_proj.weight``) -- see the
    ``Qwen3MoeForCausalLM`` module layout. A bare exact-match ``_place`` therefore
    silently no-ops on every dense weight (the param name differs by the prefix),
    leaving the model at random init. ``lm_head.weight`` matches either way (no
    prefix on either side), so it is the one dense weight a naive ``_place`` does
    fill -- the asymmetry that masks the rest. This strips a single leading
    ``model.`` segment so the checkpoint key resolves to the model param.
    """
    if name.startswith("model."):
        name = name[len("model.") :]
    _place(model, name, tensor)


def _place(model, name: str, tensor: torch.Tensor) -> None:
    """Write ``tensor`` into the model's parameter named ``name``.

    The model's parameter set is populated as its forward pass is implemented
    (``engine-loop`` / the per-model issues). Until then the base model owns an
    empty set, so this no-ops for names the model does not yet define -- the
    loader's job is to stream the weights and route them, which the test asserts
    via the (empty) parameter set and the MoE banks.
    """
    named = dict(model.named_parameters())
    if name not in named:
        named.update(dict(model.named_buffers()))
    if name in named:
        param = named[name]
        with torch.no_grad():
            param.copy_(tensor.to(param.device, param.dtype))


__all__ = ["load_model"]
