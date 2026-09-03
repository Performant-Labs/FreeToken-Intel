"""Checkpoint weight iteration and MoE expert source banks.

Upstream NVIDIA path: python/freetoken/models/weight.py
Fill in: GitHub issue `models-loader` (see docs/architecture.md).

``load_weight`` routes every tensor of a checkpoint to its destination device:
dense (attention / dense MLP / embeddings) weights go to the XPU, while MoE
expert weights stay on host memory (offloaded into per-layer "banks" and served
to the engine on demand -- the XPU has no room for all experts). The two paths
share one iterator so the engine's load loop is device-agnostic.

``load_moe_expert_sources`` returns per-layer bank *descriptors* for the offloaded
experts (the upstream ``(gate_up_banks, down_banks)`` shape, each entry a bank
exposing ``.tensor`` and ``.pin()``). With ``dummy=True`` the checkpoint is never
touched and the banks are randomly filled from the parsed config, so the whole
load path is exercisable offline on a CPU machine (the CPU test suite uses this).
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple

import torch
from safetensors import safe_open

from freetoken.utils import cached_load_hf_config, div_even, download_hf_weight

from .register import _load_attr, get_model_spec

# safetensors header dtype strings -> torch dtypes (for the parallel reader).
_ST_DTYPE = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "F8_E8M0": torch.float8_e8m0fnu,
}


class _PlainBank:
    """Host-memory expert bank.

    A plain (unpinned) tensor with a no-op ``pin``. On an XPU build ``torch.cuda``
    is absent, so upstream's ``HostBank`` (which pins for a fast H2D copy) has no
    CUDA host-alloc to lean on -- a plain contiguous host tensor is what an XPU
    ``.to("xpu")`` per-step copy sources from, and a no-op pin keeps the bank
    object's interface stable. Attribute access (``.shape``, ``.dtype``, ``.to``
    ...) falls through to the underlying tensor so callers treat the bank
    transparently.
    """

    __slots__ = ("tensor",)

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def pin(self) -> None:  # no-op: no CUDA host-alloc on the Intel build
        pass

    def __getattr__(self, name):
        # Delegate tensor attributes (shape/dtype/to/...) to the stored tensor.
        # __getattr__ only fires for names not found on the bank itself, so
        # ``.tensor`` and ``.pin`` still resolve to the bank's own members.
        return getattr(self.tensor, name)


def iter_safetensors(model_path: str, device: torch.device | str = "cpu"):
    """Yield ``(name, tensor)`` for every tensor in a local safetensors checkpoint.

    Uses the index's ``weight_map`` when present (so a name resolves to the shard
    that holds it, even when a sibling lives in a different file) and otherwise
    falls back to scanning each shard's header. Tensors are materialized on
    ``device``.

    Auto-detects an FTW archive (issue `ftw-checkpoint`, #11) and reads that
    instead when ``model_path`` is one -- every caller of this function (every
    model's ``iter_weights``, hence ``load_model`` / ``ft serve --model``)
    gets FTW support for free, no per-model changes needed.
    """
    folder = download_hf_weight(model_path)
    from freetoken.checkpoint.ftw import FtwArchive, is_ftw_dir

    if is_ftw_dir(folder):
        yield from FtwArchive(folder).read(device)
        return
    index = os.path.join(folder, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        map_ = {name: os.path.join(folder, shard) for name, shard in weight_map.items()}
    else:  # single-file / unindexed checkpoint: build the map from each shard header
        map_ = {}
        for path in sorted(glob.glob(os.path.join(folder, "*.safetensors"))):
            if path.endswith("consolidated.safetensors"):
                continue
            with safe_open(path, framework="pt", device="cpu") as f:
                for name in f.keys():
                    map_[name] = path
    # Group by shard file and open each shard exactly once, yielding every one
    # of its tensors before moving on -- not one open (== one mmap of the
    # whole shard) per tensor. Real bug, not a hypothetical one: found via
    # issue #138's real-checkpoint validation, where a GPTQ MoE checkpoint's
    # per-expert layout puts thousands of tensors in one shard (10,240 in one
    # ~1.4GB shard of the real Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 checkpoint) --
    # reopening (mmap-ing) that same file once per tensor exhausted a 27GB
    # virtual-address ulimit well before physical RAM was ever the
    # constraint. A checkpoint with only a handful of tensors per shard (the
    # CPU test suite's tiny fixtures) never has enough tensors-per-shard for
    # this to matter, which is why it went unnoticed until real scale.
    by_path: Dict[str, list] = {}
    for name, path in map_.items():
        by_path.setdefault(path, []).append(name)
    for path, names in by_path.items():
        with safe_open(path, framework="pt", device=str(device)) as f:
            for name in names:
                yield name, f.get_tensor(name)


def load_weight(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool = True,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Yield the checkpoint's tensors, each on its destination device.

    Dense tensors are yielded on ``device`` (the XPU); MoE expert tensors are
    yielded on host memory (offloaded). Dispatches to the model's
    ``iter_weights`` with the destination device. The FTW / GGUF branches of the
    upstream loader are not part of this port and are out of scope.
    """
    _config, spec = _spec_for_model_path(model_path)
    iter_weights = _load_attr(spec.module, spec.iter_weights)
    yield from iter_weights(
        model_path,
        device,
        include_moe_experts=include_moe_experts,
        include_non_moe=True,
    )


def load_moe_expert_sources(
    model_path: str,
    *,
    dtype: torch.dtype,
    dummy: bool = False,
    layer_sink=None,
    moe_backend: Optional[str] = None,
) -> Tuple[list, list]:
    """Per-layer MoE expert source banks: ``(gate_up_banks, down_banks)``.

    Each bank is a per-layer ``[num_experts, ...]`` tensor exposing ``.tensor`` /
    ``.pin()`` (see :class:`_PlainBank`). ``dummy=True`` fabricates the banks
    from the parsed config without reading the checkpoint.

    ``moe_backend`` is the engine's routing choice; it re-parses the model config
    so the banks land on the device that backend needs (host for offload/hybrid/
    cpu, XPU only for a hypothetical in-VRAM backend). See the body for why the
    XPU placement would otherwise hang engine construction.
    """
    # Re-parse with the engine's MoE routing flags so the banks land on the
    # device the *engine* wants, not parse_config's defaults. Without this the
    # loader would build the per-layer banks on the XPU (offload off) and
    # _finalize_per_expert_banks would fuse them with XPU 3-D cats that the
    # Level Zero driver serializes and hangs on -- the first request then blocks
    # in engine construction forever. The offload / hybrid / cpu backends all
    # keep the banks on host (the cache fetches them to the XPU per step).
    effective_backend = moe_backend if moe_backend else "auto"
    offload = effective_backend in ("offload", "hybrid", "cpu")
    config, spec = _spec_for_model_path(model_path, use_offload_moe=offload)
    if not config.is_moe:
        raise ValueError(f"{config.architectures[0]} does not provide MoE expert source loading")
    if dummy:
        return dummy_moe_expert_sources(config, dtype=dtype)
    iter_weights = _load_attr(spec.module, spec.iter_weights)
    src = iter_weights(
        model_path,
        torch.device("cpu"),
        include_moe_experts=True,
        include_non_moe=False,
    )
    # Issue moe-quant-banks-e2e (#138): a GPTQ-quantized checkpoint's expert
    # tensors arrive from iter_weights RAW and packed (bank-only mode, see
    # its docstring) -- stream_moe_expert_sources_gptq (#135) is the only
    # function that knows how to fold them into per-layer banks without
    # dequantizing. checkpoint_quant_method reads this straight from the
    # checkpoint's own config.json, independent of ModelConfig (which does
    # not carry quantization_config today).
    quant_method = checkpoint_quant_method(model_path)
    if quant_method == "gptq":
        return stream_moe_expert_sources_gptq(src, config)
    # "fp8" (issue moe-quant-banks-fp8, #152): DeepSeek-V3 / sglang / vLLM's
    # own quantization_config.quant_method spelling for the block-FP8
    # convention this streams (verified against deepseek-ai/DeepSeek-V3's
    # real config.json -- see stream_moe_expert_sources_fp8's docstring).
    if quant_method == "fp8":
        return stream_moe_expert_sources_fp8(src, config)
    # openai/gpt-oss-{20b,120b}'s own config.json quantization_config.quant_method
    # (verified: https://huggingface.co/openai/gpt-oss-20b/raw/main/config.json).
    if quant_method == "mxfp4":
        return stream_moe_expert_sources_mxfp4(src, config)
    # "compressed-tensors" (issue moe-quant-banks-int8, #154): this
    # quant_method is shared across multiple bit-widths/strategies -- only
    # dispatch to the INT8 streamer when checkpoint_is_int8_pack_quantized
    # confirms num_bits=8/type="int"/format="pack-quantized" specifically
    # (verified against rj1013/gemma-4-26B-A4B-it_q8's real config.json; see
    # stream_moe_expert_sources_int8's docstring). Any other compressed-
    # tensors scheme (e.g. INT4, a different FP8 variant) falls through to
    # the plain bf16 streamer below and fails loudly there -- a real,
    # not-yet-supported format, not silently mishandled as INT8.
    if quant_method == "compressed-tensors" and checkpoint_is_int8_pack_quantized(model_path):
        return stream_moe_expert_sources_int8(src, config)
    return stream_moe_expert_sources(src, config, dtype=dtype, layer_sink=layer_sink)


def checkpoint_quant_method(model_path: str) -> Optional[str]:
    """``quantization_config.quant_method`` from the checkpoint's own
    ``config.json`` (e.g. ``"gptq"``), or ``None`` for an unquantized
    checkpoint. Read directly from the raw HF config -- independent of
    ``ModelConfig``/``parse_config``, which does not stash this today."""
    hf_config = cached_load_hf_config(model_path)
    raw = hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(hf_config)
    qc = raw.get("quantization_config")
    if not isinstance(qc, dict):
        text_config = raw.get("text_config")
        qc = text_config.get("quantization_config") if isinstance(text_config, dict) else None
    return qc.get("quant_method") if isinstance(qc, dict) else None


def _compressed_tensors_weights_config(model_path: str) -> Optional[dict]:
    """``quantization_config.config_groups.*.weights`` (the first group) from
    a checkpoint using the ``compressed-tensors`` library's format, or
    ``None`` if it has no such config. Shared helper for the ``int8_pack_
    quantized`` checks below (issue `moe-quant-banks-int8`, #154)."""
    hf_config = cached_load_hf_config(model_path)
    raw = hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(hf_config)
    qc = raw.get("quantization_config")
    if not isinstance(qc, dict):
        text_config = raw.get("text_config")
        qc = text_config.get("quantization_config") if isinstance(text_config, dict) else None
    if not isinstance(qc, dict):
        return None
    config_groups = qc.get("config_groups")
    if not isinstance(config_groups, dict) or not config_groups:
        return None
    group = next(iter(config_groups.values()))
    if not isinstance(group, dict) or group.get("format") != "pack-quantized":
        return None
    weights = group.get("weights")
    return weights if isinstance(weights, dict) else None


def checkpoint_is_int8_pack_quantized(model_path: str) -> bool:
    """Whether this checkpoint uses ``compressed-tensors``' ``pack-
    quantized`` INT8 scheme (issue `moe-quant-banks-int8`, #154 -- verified
    against ``rj1013/gemma-4-26B-A4B-it_q8``'s real ``config.json``):
    ``num_bits: 8``, ``type: "int"``, ``format: "pack-quantized"``.

    ``compressed-tensors``' own ``quant_method`` value (``"compressed-
    tensors"``, from :func:`checkpoint_quant_method`) is shared across
    multiple bit-widths/strategies (INT8, INT4, FP8 variants, ...) --
    unlike GPTQ's or block-FP8's own dedicated ``quant_method`` strings
    (``"gptq"``, ``"fp8"``), so a caller must inspect ``config_groups``
    itself to confirm this specific scheme, not just check ``quant_method``
    alone. Call this BEFORE :func:`checkpoint_int8_pack_quantized_group_size`
    to distinguish "not this format" from "this format, per-channel"
    (``group_size: null`` is a valid value of THIS format, not an absence
    of it).
    """
    weights = _compressed_tensors_weights_config(model_path)
    return weights is not None and weights.get("num_bits") == 8 and weights.get("type") == "int"


def checkpoint_int8_pack_quantized_group_size(model_path: str) -> Optional[int]:
    """``quantization_config.config_groups.*.weights.group_size`` for a
    checkpoint :func:`checkpoint_is_int8_pack_quantized` has already
    confirmed is a real ``compressed-tensors`` ``pack-quantized`` INT8
    checkpoint. ``None`` means the checkpoint's own config literally sets
    ``group_size: null`` -- pure per-channel (one group covering the whole
    row), NOT "this function found no config" (call
    :func:`checkpoint_is_int8_pack_quantized` first to rule that case out).
    """
    weights = _compressed_tensors_weights_config(model_path)
    return weights.get("group_size") if weights is not None else None


def checkpoint_gptq_group_size(model_path: str) -> int:
    """``quantization_config.group_size`` from a GPTQ checkpoint's own
    ``config.json``. Raises if the checkpoint has no GPTQ
    ``quantization_config`` -- callers must check :func:`checkpoint_quant_method`
    first (this is not a general-purpose accessor)."""
    hf_config = cached_load_hf_config(model_path)
    raw = hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(hf_config)
    qc = raw.get("quantization_config")
    if not isinstance(qc, dict):
        text_config = raw.get("text_config")
        qc = text_config.get("quantization_config") if isinstance(text_config, dict) else None
    if not isinstance(qc, dict) or "group_size" not in qc:
        raise ValueError(f"{model_path!r} has no quantization_config.group_size (not a GPTQ checkpoint?)")
    return int(qc["group_size"])


def _spec_for_model_path(model_path: str, use_offload_moe: bool = False):
    hf_config = cached_load_hf_config(model_path)
    spec = get_model_spec(hf_config.architectures[0])
    parse_config = _load_attr(spec.module, spec.parse_config)
    return parse_config(hf_config, use_offload_moe=use_offload_moe), spec


def _num_moe_layers(config) -> int:
    value = getattr(config, "num_moe_layers", None)
    if value is not None:
        return int(value)
    return int(config.num_layers) - int(getattr(config, "first_k_dense_replace", 0))


def dummy_moe_expert_sources(config, *, dtype: torch.dtype) -> Tuple[list, list]:
    """Random expert banks shaped exactly like the real loader's output: one
    independently allocated ``[num_experts, ...]`` bank per MoE layer, so the
    downstream repack/offload path runs unchanged with no weights on disk."""
    num_layers = _num_moe_layers(config)
    intermediate_size = div_even(config.moe_intermediate_size, 1)  # TP=1 on the B70
    gate_up = [
        _PlainBank(torch.randn(config.num_experts, 2 * intermediate_size, config.hidden_size, dtype=dtype))
        for _ in range(num_layers)
    ]
    down = [
        _PlainBank(torch.randn(config.num_experts, config.hidden_size, intermediate_size, dtype=dtype))
        for _ in range(num_layers)
    ]
    return gate_up, down


def stream_moe_expert_sources(
    tensors: Iterator[Tuple[str, torch.Tensor]],
    config,
    *,
    dtype: torch.dtype,
    layer_sink=None,
) -> Tuple[list, list]:
    """Stream per-layer MoE expert tensors into final offload banks.

    Two checkpoint layouts are accepted (ADR 0002):

    * **packed** -- ``...experts.{gate_up_proj|down_proj}`` already stacked to
      ``[num_experts, ...]`` (what the model adapter / FTW normalizes to). Each
      arrives whole-layer and is written straight into its per-layer bank.
    * **per-expert** -- the raw HF layout
      ``...experts.{e}.{gate|up|down}_proj`` (one ``[I, H]`` / ``[H, I]`` tensor
      per expert). A real HF checkpoint (e.g. Qwen3-30B-A3B) ships this form, so
      the streamer stacks the per-expert rows into the packed
      ``gate_up [E, 2I, H]`` (gate then up, concatenated on dim 1) and
      ``down [E, H, I]`` banks -- the exact layout the ``dummy=True`` fabricator
      and the offload slot cache expect.

    Either form (or a mix) is fine per layer. Returns ``(gate_up_banks,
    down_banks)`` (per-layer bank objects), each ``[num_experts, ...]``.
    """
    del layer_sink  # converter-facing sink not used on the B70 port
    banks: dict[str, list] = {
        "gate_up": [None] * config.num_layers,
        "down": [None] * config.num_layers,
    }
    row_shape: dict[str, tuple[int, ...]] = {}
    seen: dict[str, set[int]] = {"gate_up": set(), "down": set()}
    # Per-expert rows are buffered as {layer: {expert_id: row}} and fused into the
    # packed bank at the end (see _finalize_per_expert_banks) rather than written
    # in-place row-by-row: the in-place int+slice / narrow / scatter write paths of
    # the torch XPU build are unreliable, whereas torch.cat is not.
    per_expert: dict[str, list[dict[int, torch.Tensor]]] = {
        "gate": [{} for _ in range(config.num_layers)],
        "up": [{} for _ in range(config.num_layers)],
        "down": [{} for _ in range(config.num_layers)],
    }

    for name, tensor in tensors:
        info = _expert_source_info(name)
        if info is None:
            raise ValueError(f"Unexpected expert weight key: {name}")
        layer, packed_name, expert_id = info
        bank_name = "gate_up" if packed_name in {"gate_up_proj", "gate_proj", "up_proj"} else "down"
        if expert_id is None:
            # Packed: the tensor is the whole [num_experts, ...] layer -> copy as-is.
            _copy_expert_layer_into_bank(
                banks,
                row_shape,
                seen,
                bank_name=bank_name,
                tensor=tensor.to(dtype=dtype),
                layer=layer,
                config=config,
                dtype=dtype,
            )
        else:
            # Per-expert: buffer this one row and try to fuse it right away.
            # Finalizing per layer as soon as that layer's rows are complete
            # (rather than only once at the very end of the stream) keeps at
            # most a handful of layers' raw per-expert rows resident at once
            # instead of the whole checkpoint's -- deferring to end-of-stream
            # meant the buffered rows (~the full expert bank) and the finalized
            # banks (~the full expert bank again) were resident simultaneously,
            # roughly doubling peak host RAM during load.
            _buffer_expert_row(per_expert, row_shape, bank_name, packed_name, tensor.to(dtype=dtype), layer, expert_id, config)
            seen[bank_name].add(layer)
            _maybe_finalize_layer(banks, per_expert, config, layer)

    missing = {
        name: sorted(set(range(config.num_layers)) - seen)
        for name, seen in seen.items()
        if seen != set(range(config.num_layers))
    }
    if missing:
        raise ValueError(f"Missing MoE expert source layers: {missing}")
    _finalize_per_expert_banks(banks, per_expert, config)
    return (
        [bank.tensor for bank in banks["gate_up"]],
        [bank.tensor for bank in banks["down"]],
    )


def _expert_source_info(key: str) -> Tuple[int, str, Optional[int]] | None:
    """Map a HF MoE expert key to ``(layer, name, expert_id)`` or None.

    ``name`` is the trailing projection token and ``expert_id`` is the row index for
    per-expert keys (``None`` when the key is already the packed,
    ``[num_experts, ...]`` tensor). Recognized forms (anchored on ``mlp`` -> ``experts``):

    * packed -- ``model.layers.{L}.mlp.experts.{gate_up_proj|down_proj}`` (the
      experts key is terminal, nothing follows it) -> ``(L, name, None)``
    * per-expert -- ``model.layers.{L}.mlp.experts.{e}.{gate|up|down}_proj`` ->
      ``(L, name, e)`` (``gate_proj``/``up_proj`` both feed the ``gate_up`` bank,
      ``down_proj`` feeds ``down``). A non-projection trailing token (e.g. an
      unknown ``...experts.{e}.{other}``) is not an expert source and yields ``None``.
    """
    parts = key.split(".")
    # A MoE expert key is anchored on the trailing ``.experts.`` group. The layer
    # id is the integer token immediately before the ``mlp`` token that precedes
    # ``experts`` (``...layers.{L}.mlp.experts...``), which is not a fixed offset
    # from the end (the per-expert form appends ``.{e}.{proj}...``), so locate
    # ``mlp`` first and take the integer that directly precedes it.
    if "mlp" not in parts or "experts" not in parts:
        return None
    try:
        mlp_pos = parts.index("mlp")
        layer = int(parts[mlp_pos - 1])
    except (ValueError, IndexError):
        return None
    if parts[mlp_pos + 1] != "experts":
        return None
    e_pos = parts.index("experts")
    tail = parts[e_pos + 1 :]
    # A GPTQ-packed component (.qweight/.qzeros/.scales/.g_idx, see
    # stream_moe_expert_sources_gptq, issue #135) is NOT a plain dense
    # weight tensor -- silently parsing e.g. "...gate_proj.qweight" as if it
    # were "...gate_proj"'s actual weight would misinterpret a raw
    # int32-packed tensor as the dense weight itself (garbage, not an
    # error). Reject loudly instead of guessing: a GPTQ checkpoint's expert
    # tensors must go through stream_moe_expert_sources_gptq, not this
    # (bf16-oriented) function.
    if tail and tail[-1] in _GPTQ_COMPONENTS:
        raise ValueError(
            f"GPTQ-packed expert weight key {key!r} passed to the bf16 expert-source "
            "streamer; use stream_moe_expert_sources_gptq for a GPTQ checkpoint instead"
        )
    # Same defense for block-FP8's per-block scale component (see
    # stream_moe_expert_sources_fp8, issue #152). Unlike GPTQ's qweight/qzeros/
    # g_idx, block-FP8's OTHER component is spelled ``.weight`` -- identical to
    # a plain dense tensor's own trailing token -- so it cannot be rejected the
    # same way and instead relies on load_moe_expert_sources' checkpoint_quant_method
    # dispatch (the primary safety net for both quantized formats) to never reach
    # this function at all. ``weight_scale_inv`` has no such collision, so it is
    # rejected here too, as defense in depth.
    if tail and tail[-1] == "weight_scale_inv":
        raise ValueError(
            f"block-FP8-packed expert weight key {key!r} passed to the bf16 expert-source "
            "streamer; use stream_moe_expert_sources_fp8 for a block-FP8 checkpoint instead"
        )
    # Real HF safetensors keys carry a trailing ``.weight`` (``...experts.{e}.{
    # gate|up|down}_proj.weight``); the FTW-normalized packed form (ADR 0002) may
    # omit it (``...experts.{gate_up_proj|down_proj}``). A ``.bias`` trailing token
    # likewise exists on some checkpoints. Strip one terminal ``.weight``/``.bias``
    # (and nothing else) so both spellings resolve to the same projection token.
    if tail and tail[-1] in {"weight", "bias"}:
        tail = tail[:-1]
    # Packed form: the experts group holds exactly one (non-numeric) token -- the
    # projection name. A packed ``gate_up_proj``/``down_proj`` key is terminal
    # (``len(tail) == 1``); a per-expert key always appends ``.{e}.{proj}``
    # (``len(tail) >= 2``). The two are distinguished by that length, not by a
    # specific name (``down_proj`` exists in both forms).
    if len(tail) == 1:
        if tail[0] in {"gate_up_proj", "down_proj"}:
            return layer, tail[0], None
        return None
    # Per-expert form: ``...experts.{e}.{proj}`` -- tail[0] is the expert index and
    # the projection is the next token (parts[-1] after the optional weight strip
    # would be ``weight``; parts[-2] is the real projection name).
    proj = tail[1] if len(tail) >= 2 else None
    if proj in {"gate_proj", "up_proj", "down_proj"}:
        try:
            expert_id = int(tail[0])
        except ValueError:
            return None
        return layer, proj, expert_id
    return None


def _copy_expert_layer_into_bank(
    banks: dict[str, list],
    row_shape: dict[str, tuple[int, ...]],
    seen: dict[str, set[int]],
    *,
    bank_name: str,
    tensor: torch.Tensor,
    layer: int,
    config,
    dtype: torch.dtype,
) -> None:
    if layer < 0 or layer >= config.num_layers:
        raise ValueError(f"Unexpected MoE expert layer {layer}; expected [0, {config.num_layers})")
    if tensor.size(0) != config.num_experts:
        raise ValueError(
            f"Unexpected {bank_name} expert count {tensor.size(0)}; expected {config.num_experts}"
        )
    expected_shape = row_shape.setdefault(bank_name, tuple(tensor.shape[1:]))
    if tuple(tensor.shape[1:]) != expected_shape:
        raise ValueError(
            f"Inconsistent {bank_name} expert shape {tuple(tensor.shape[1:])}; expected {expected_shape}"
        )
    bank = banks[bank_name][layer]
    if bank is None:
        banks[bank_name][layer] = bank = _PlainBank(torch.empty((config.num_experts, *tensor.shape[1:]), dtype=dtype))
    bank.tensor.copy_(tensor)
    seen[bank_name].add(layer)


def _buffer_expert_row(
    per_expert: dict[str, list[dict[int, torch.Tensor]]],
    row_shape: dict[str, tuple[int, ...]],
    bank_name: str,
    packed_name: str,
    tensor: torch.Tensor,
    layer: int,
    expert_id: int,
    config,
) -> None:
    """Buffer one per-expert row; the packed bank is fused at the end of the stream.

    ``gate_proj`` / ``up_proj`` are keyed under the ``gate_up`` bank (as the ``gate``
    and ``up`` halves respectively); ``down_proj`` under ``down``. Rows are not
    written in-place -- see :func:`_finalize_per_expert_banks`.
    """
    bucket = "gate" if packed_name == "gate_proj" else ("up" if packed_name == "up_proj" else "down")
    bucket_by_layer = per_expert[bucket]
    if not (0 <= layer < config.num_layers):
        raise ValueError(f"Unexpected MoE expert layer {layer}")
    if not (0 <= expert_id < config.num_experts):
        raise ValueError(f"Unexpected MoE expert id {expert_id} in layer {layer}")
    expected_shape = row_shape.setdefault(bank_name, tuple(tensor.shape))
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"Inconsistent {bank_name} per-expert shape {tuple(tensor.shape)}; expected {expected_shape}"
        )
    if expert_id in bucket_by_layer[layer]:
        raise ValueError(f"Duplicate per-expert source for layer {layer} expert {expert_id} ({packed_name})")
    bucket_by_layer[layer][expert_id] = tensor


def _maybe_finalize_layer(banks: dict[str, list], per_expert, config, layer: int) -> None:
    """Fuse one layer's buffered per-expert rows into its packed banks, if complete.

    ``gate_up`` is ``[E, 2I, H]`` (gate then up, matching the ``dummy=True``
    fabricator and the upstream ``_BANK_SCHEMAS``); ``down`` is ``[E, H, I]``.
    A layer is only finalized once *all* of its experts are present (gate + up +
    down), so a checkpoint that ships a bank packed and the rest per-expert (or
    vice versa) still completes.

    The per-expert rows (``[I, H]`` / ``[H, I]``) are stacked into the ``[E, ...]``
    bank with ``cat`` on the *middle* dim + a ``permute`` (``_stack_expert_rows``),
    and the gate/up fusion is a ``cat`` on dim 1. These 3-D ops are the only
    reliable path in the torch XPU build: in-place row writes (``__setitem__`` with
    an int + slice, ``narrow`` + ``copy_``, ``scatter``) and ``cat``/``reshape`` on
    the 2-D per-expert rows are all mishandled by that build (they drop or
    misplace the expert dimension), whereas ``cat(dim=1)`` + ``permute`` on 3-D
    tensors are correct.

    Called as soon as a layer's rows complete (per-tensor, during the stream) so
    at most a few in-flight layers' raw rows are resident at once, rather than
    the whole checkpoint's -- see the call site in :func:`stream_moe_expert_sources`.
    The buffered rows for a finalized layer are dropped immediately so their
    memory doesn't coexist with the packed bank it was fused into.
    """
    if banks["gate_up"][layer] is not None and banks["down"][layer] is not None:
        return  # already produced (packed source, or finalized earlier)
    gate = per_expert["gate"][layer]
    up = per_expert["up"][layer]
    down = per_expert["down"][layer]
    num_experts = config.num_experts
    # Only finalize a layer whose expert coverage is complete.
    if len(gate) < num_experts or len(up) < num_experts or len(down) < num_experts:
        return
    gate_bank = _stack_expert_rows([gate[e] for e in range(num_experts)])  # [E, I, H]
    up_bank = _stack_expert_rows([up[e] for e in range(num_experts)])  # [E, I, H]
    down_bank = _stack_expert_rows([down[e] for e in range(num_experts)])  # [E, H, I]
    # Fuse gate + up on the inner (dim 1) axis -> [E, 2I, H].
    banks["gate_up"][layer] = _PlainBank(torch.cat([gate_bank, up_bank], dim=1))
    banks["down"][layer] = _PlainBank(down_bank)
    # Drop this layer's raw rows now that the packed bank owns the data.
    per_expert["gate"][layer] = {}
    per_expert["up"][layer] = {}
    per_expert["down"][layer] = {}


def _finalize_per_expert_banks(banks: dict[str, list], per_expert, config) -> None:
    """End-of-stream safety net: finalize any layer :func:`_maybe_finalize_layer`
    didn't already catch during streaming (e.g. an unusual checkpoint ordering).
    A no-op for every layer the incremental path already finalized.
    """
    for layer in range(config.num_layers):
        _maybe_finalize_layer(banks, per_expert, config, layer)


def _stack_expert_rows(rows: list) -> torch.Tensor:
    """Stack ``[E]`` per-expert rows of shape ``[D0, D1]`` into one ``[E, D0, D1]``
    tensor, using only ops the torch XPU build handles correctly (see
    :func:`_finalize_per_expert_banks`).

    Each 2-D row is promoted to ``[1, D0, D1]`` (``unsqueeze(0)``) and the results
    are ``cat``-ed on dim 0 -> ``[E, D0, D1]``. A direct ``cat`` on the 2-D rows
    (dim 0 or 1) drops/misplaces the expert dimension in this build, so the rows
    are first made 3-D: ``cat`` on a 3-D tensor along dim 0 is reliable.
    """
    if not rows:
        raise ValueError("Cannot stack an empty list of expert rows")
    first = rows[0]
    if first.dim() != 2:
        raise ValueError(f"Expected 2-D per-expert rows; got {first.dim()}-D")
    if any(r.dim() != 2 or r.shape != first.shape for r in rows[1:]):
        raise ValueError("Per-expert rows must share a single shape")
    return torch.cat([row.unsqueeze(0) for row in rows], dim=0)


@dataclass(frozen=True)
class GptqExpertBank:
    """One MoE layer's packed GPTQ-quantized bank for one projection slot
    (``gate_up`` or ``down``) -- the quantized-format sibling of the plain
    ``[E, ...]`` bf16 bank tensor ``stream_moe_expert_sources`` produces
    (issue `moe-quant-banks-pack`, #135).

    ``qweight`` / ``qzeros`` / ``scales`` hold one row per expert (``[E,
    ...]``, AutoGPTQ's packed-int32 layout -- see
    :mod:`freetoken.kernel.triton.gptq_linear`). ``g_idx`` does NOT: it is
    shared across every expert of this projection type in this layer
    (``g_idx[k] = k // group_size`` depends only on ``K``/``group_size``,
    both architecture constants, never on which expert), so stacking it per
    expert would be pure, avoidable duplication.

    Deliberately packed, never dequantized here: dequantizing every expert
    to bf16 at load time is a 4x expansion that blows the host RAM budget
    for a real-scale checkpoint (issue #134). Dequantization happens lazily,
    per-expert, at compute time (issue #137) against whichever of these rows
    the offload cache's device slot pool has fetched for the current step.
    """

    qweight: torch.Tensor  # [E, K // 8, N] int32
    qzeros: torch.Tensor  # [E, ceil(K/group_size), N // 8] int32
    scales: torch.Tensor  # [E, ceil(K/group_size), N]
    g_idx: torch.Tensor  # [K] int32 -- shared across every expert


_GPTQ_COMPONENTS = ("qweight", "qzeros", "scales", "g_idx")


def _parse_gptq_expert_key(key: str) -> Tuple[int, str, int, str] | None:
    """Parse a GPTQ-packed per-expert weight key into ``(layer, proj,
    expert_id, component)``, or ``None`` if ``key`` is not one.

    Recognizes ``...layers.{L}.mlp.experts.{e}.{gate_proj|up_proj|down_proj}
    .{qweight|qzeros|scales|g_idx}`` -- the real checkpoint's raw per-expert
    GPTQ layout (confirmed against ``Qwen/Qwen3.5-35B-A3B-GPTQ-Int4``'s own
    ``model.safetensors.index.json``; there is no "packed" GPTQ spelling to
    also accept, unlike :func:`_expert_source_info`'s bf16 case -- GPTQ
    checkpoints only ever ship the raw per-expert form).
    """
    parts = key.split(".")
    if len(parts) < 2 or parts[-1] not in _GPTQ_COMPONENTS:
        return None
    component = parts[-1]
    body = parts[:-1]
    if "mlp" not in body or "experts" not in body:
        return None
    try:
        mlp_pos = body.index("mlp")
        layer = int(body[mlp_pos - 1])
    except (ValueError, IndexError):
        return None
    if body[mlp_pos + 1] != "experts":
        return None
    e_pos = body.index("experts")
    tail = body[e_pos + 1 :]
    if len(tail) != 2:
        return None
    try:
        expert_id = int(tail[0])
    except ValueError:
        return None
    proj = tail[1]
    if proj not in {"gate_proj", "up_proj", "down_proj"}:
        return None
    return layer, proj, expert_id, component


def stream_moe_expert_sources_gptq(
    tensors: Iterator[Tuple[str, torch.Tensor]],
    config,
) -> Tuple[list, list]:
    """Stream GPTQ-packed per-expert weight tensors into packed per-layer
    banks (issue `moe-quant-banks-pack`, #135) -- the quantized-format
    sibling of :func:`stream_moe_expert_sources`.

    Kept as a separate function rather than folded into
    ``stream_moe_expert_sources``: GPTQ's four-tensor-per-projection shape
    does not fit that function's single-tensor-per-projection contract, and
    that bf16 path is well-tested and used by every other model/checkpoint
    -- not worth risking a regression there to shoehorn GPTQ's shape in.

    ``gate_up`` fuses ``gate_proj`` + ``up_proj`` (the checkpoint stores
    them as separate quantized tensors, not pre-fused):
    ``qweight``/``qzeros``/``scales`` concatenate along their ``N``
    (output-channel) axis; ``g_idx`` does not change (same ``K``/
    ``group_size`` for both halves -- asserted here, not assumed, and
    likewise asserted equal across every expert of a bank, since it is
    architecturally shared, never per-expert -- see :class:`GptqExpertBank`).

    Returns ``(gate_up_banks, down_banks)``, each a list of ``num_layers``
    :class:`GptqExpertBank`. Every tensor stays in its packed, quantized
    form the whole way through -- never dequantized here (see
    :class:`GptqExpertBank`'s docstring for why).

    Finalizes each ``(layer, bank_name)`` as soon as its last component
    arrives -- immediately concatenating/stacking it into a
    :class:`GptqExpertBank` and releasing its raw per-expert tensors --
    rather than buffering the *entire checkpoint's* raw tensors until the
    stream ends. This was a real bug, not a hypothetical one: found by
    issue #138's real-checkpoint validation, an earlier version of this
    function buffered every layer's raw components until after the whole
    stream was consumed, so peak RAM briefly held (nearly) the whole raw
    checkpoint *and* the packed banks being built from it at once --
    exactly the RAM blowup issue #135 (this function's own issue) was
    supposed to eliminate, just moved to a different phase of the same
    call. The real ``Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`` checkpoint's loader
    call exhausted a 27GB virtual-address ceiling on shard 6 of 14 before
    this fix.
    """
    buf: Dict[Tuple[int, str], Dict[int, Dict[str, Dict[str, torch.Tensor]]]] = {}
    finalized: Dict[Tuple[int, str], GptqExpertBank] = {}

    for name, tensor in tensors:
        info = _parse_gptq_expert_key(name)
        if info is None:
            raise ValueError(f"Unexpected GPTQ expert weight key: {name}")
        layer, proj, expert_id, component = info
        bank_name = "gate_up" if proj in ("gate_proj", "up_proj") else "down"
        if not (0 <= layer < config.num_layers):
            raise ValueError(f"Unexpected MoE expert layer {layer}; expected [0, {config.num_layers})")
        if not (0 <= expert_id < config.num_experts):
            raise ValueError(f"Unexpected MoE expert id {expert_id} in layer {layer}")
        key = (layer, bank_name)
        if key in finalized:
            raise ValueError(
                f"Layer {layer} {bank_name!r}: tensor {name!r} arrived after this bank was already "
                "finalized -- duplicate key or an out-of-order/re-streamed source?"
            )
        by_expert = buf.setdefault(key, {})
        by_proj = by_expert.setdefault(expert_id, {})
        by_component = by_proj.setdefault(proj, {})
        if component in by_component:
            raise ValueError(f"Duplicate GPTQ component {component!r} for layer {layer} expert {expert_id} {proj}")
        by_component[component] = tensor

        fuse = bank_name == "gate_up"
        if _gptq_bank_is_complete(by_expert, config.num_experts, fuse=fuse):
            finalized[key] = _finalize_gptq_bank(by_expert, config.num_experts, fuse=fuse, layer=layer)
            del buf[key]  # release the raw per-expert tensors now that the packed bank owns the data

    missing = [
        (layer, bank_name)
        for layer in range(config.num_layers)
        for bank_name in ("gate_up", "down")
        if (layer, bank_name) not in finalized
    ]
    if missing:
        raise ValueError(f"Missing/incomplete GPTQ MoE expert bank(s): {missing}")

    gate_up_banks = [finalized[(layer, "gate_up")] for layer in range(config.num_layers)]
    down_banks = [finalized[(layer, "down")] for layer in range(config.num_layers)]
    return gate_up_banks, down_banks


def _gptq_bank_is_complete(
    by_expert: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
    num_experts: int,
    *,
    fuse: bool,
) -> bool:
    """True once every expert 0..num_experts-1 has all required projections
    (``gate_proj``+``up_proj`` for ``fuse=True``, else ``down_proj``), each
    with all four GPTQ components -- i.e. this ``(layer, bank_name)`` is
    ready for :func:`_finalize_gptq_bank`."""
    if set(by_expert) != set(range(num_experts)):
        return False
    required_projs = ("gate_proj", "up_proj") if fuse else ("down_proj",)
    for e in range(num_experts):
        by_proj = by_expert[e]
        for proj in required_projs:
            components = by_proj.get(proj)
            if components is None or set(components) != set(_GPTQ_COMPONENTS):
                return False
    return True


def _finalize_gptq_bank(
    by_expert: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
    num_experts: int,
    *,
    fuse: bool,
    layer: int,
) -> GptqExpertBank:
    if set(by_expert) != set(range(num_experts)):
        missing = sorted(set(range(num_experts)) - set(by_expert))
        raise ValueError(f"Layer {layer}: missing GPTQ experts {missing}")

    per_expert_rows = []
    for e in range(num_experts):
        by_proj = by_expert[e]
        if fuse:
            gate = by_proj.get("gate_proj")
            up = by_proj.get("up_proj")
            if gate is None or up is None:
                raise ValueError(f"Layer {layer} expert {e}: missing gate_proj/up_proj GPTQ components")
            if not torch.equal(gate["g_idx"], up["g_idx"]):
                raise ValueError(
                    f"Layer {layer} expert {e}: gate_proj/up_proj g_idx mismatch (different group_size/K?)"
                )
            qweight = torch.cat([gate["qweight"], up["qweight"]], dim=1)
            qzeros = torch.cat([gate["qzeros"], up["qzeros"]], dim=1)
            scales = torch.cat([gate["scales"], up["scales"]], dim=1)
            g_idx = gate["g_idx"]
        else:
            down = by_proj.get("down_proj")
            if down is None:
                raise ValueError(f"Layer {layer} expert {e}: missing down_proj GPTQ components")
            qweight, qzeros, scales, g_idx = down["qweight"], down["qzeros"], down["scales"], down["g_idx"]
        per_expert_rows.append((qweight, qzeros, scales, g_idx))

    # g_idx is shared, not per-expert (see GptqExpertBank docstring): assert
    # every expert's copy agrees, then keep only one -- stacking it into an
    # [E, K] bank would be real, avoidable duplication of identical data.
    g_idx0 = per_expert_rows[0][3]
    for e, row in enumerate(per_expert_rows[1:], start=1):
        if not torch.equal(row[3], g_idx0):
            raise ValueError(
                f"Layer {layer}: g_idx differs between expert 0 and expert {e} "
                "(unexpected -- group_size/K should be architecture-constant, identical for every expert)"
            )

    # _stack_expert_rows (not torch.stack): the torch XPU build mishandles a
    # direct cat/stack of 2-D per-expert rows along a new leading dim --
    # unsqueeze-then-cat is the reliable path this codebase already
    # standardizes on (see _stack_expert_rows's own docstring).
    return GptqExpertBank(
        qweight=_stack_expert_rows([row[0] for row in per_expert_rows]),
        qzeros=_stack_expert_rows([row[1] for row in per_expert_rows]),
        scales=_stack_expert_rows([row[2] for row in per_expert_rows]),
        g_idx=g_idx0,
    )


@dataclass(frozen=True)
class Fp8BlockExpertBank:
    """One MoE layer's packed block-FP8 bank for one projection slot
    (``gate_up`` or ``down``) -- the block-FP8 sibling of
    :class:`GptqExpertBank` (issue `moe-quant-banks-fp8`, #152, part of the
    same #134/#140 epic).

    ``weight`` / ``weight_scale_inv`` hold one row per expert (``[E, ...]``,
    the DeepSeek-V3 / sglang / vLLM ``w8a8_block_fp8`` layout -- see
    :mod:`freetoken.kernel.triton.fp8_block_linear`). Unlike GPTQ's ``g_idx``,
    block-FP8 has no side tensor shared across experts: every expert's scale
    table is a function of that expert's own weight magnitudes alone, so both
    tensors fit the plain per-expert ``[E, ...]`` bank shape and there is
    nothing left over for ``OffloadMoeCache.extra_metadata``.

    Deliberately packed, never dequantized here: dequantizing every expert to
    the activation dtype at load time is the same real-scale RAM blowup issue
    #134 built ADR 0002 to avoid. Dequantization happens lazily, per-expert,
    at compute time (:class:`freetoken.moe.offload_cache.SlotWeightAccessor`)
    against whichever row the offload cache's device slot pool has fetched
    for the current step.
    """

    weight: torch.Tensor  # [E, N, K] torch.float8_e4m3fn
    weight_scale_inv: torch.Tensor  # [E, ceil(N/block), ceil(K/block)]


_FP8_COMPONENTS = ("weight", "weight_scale_inv")
_FP8_DEFAULT_BLOCK = 128  # DeepSeek-V3 / sglang / vLLM's own default; see fp8_block_linear.py


def _parse_fp8_expert_key(key: str) -> Tuple[int, str, int, str] | None:
    """Parse a block-FP8-packed per-expert weight key into ``(layer, proj,
    expert_id, component)``, or ``None`` if ``key`` is not one.

    Recognizes ``...layers.{L}.mlp.experts.{e}.{gate_proj|up_proj|down_proj}
    .{weight|weight_scale_inv}`` -- verified against the real
    ``deepseek-ai/DeepSeek-V3`` checkpoint's own ``model.safetensors.index.json``
    on HuggingFace (its ``config.json`` reads ``quantization_config ==
    {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
    "weight_block_size": [128, 128]}``, and ``model.layers.3.mlp.experts.0.
    gate_proj.{weight,weight_scale_inv}`` are real keys in its index) -- the
    same per-expert key shape GPTQ uses (:func:`_parse_gptq_expert_key`), just
    with block-FP8's two components instead of GPTQ's four, and no shared
    ``g_idx``-like side tensor.
    """
    parts = key.split(".")
    if len(parts) < 2 or parts[-1] not in _FP8_COMPONENTS:
        return None
    component = parts[-1]
    body = parts[:-1]
    if "mlp" not in body or "experts" not in body:
        return None
    try:
        mlp_pos = body.index("mlp")
        layer = int(body[mlp_pos - 1])
    except (ValueError, IndexError):
        return None
    if body[mlp_pos + 1] != "experts":
        return None
    e_pos = body.index("experts")
    tail = body[e_pos + 1 :]
    if len(tail) != 2:
        return None
    try:
        expert_id = int(tail[0])
    except ValueError:
        return None
    proj = tail[1]
    if proj not in {"gate_proj", "up_proj", "down_proj"}:
        return None
    return layer, proj, expert_id, component


def stream_moe_expert_sources_fp8(
    tensors: Iterator[Tuple[str, torch.Tensor]],
    config,
    *,
    block: int = _FP8_DEFAULT_BLOCK,
) -> Tuple[list, list]:
    """Stream block-FP8-packed per-expert weight tensors into packed per-layer
    banks (issue `moe-quant-banks-fp8`, #152) -- the block-FP8 sibling of
    :func:`stream_moe_expert_sources_gptq`.

    ``gate_up`` fuses ``gate_proj`` + ``up_proj`` (the checkpoint stores them
    as separate quantized tensors, not pre-fused): both ``weight`` and
    ``weight_scale_inv`` concatenate along dim 0, the ``N`` (output-channel)
    axis -- matching every other bank's ``[E, 2I, H]`` gate-then-up
    convention. Unlike GPTQ's ``K``-axis groups (architecture-constant and
    shared across the whole checkpoint, so concatenating on ``N`` never
    crosses a group boundary), block-FP8 blocks the ``N`` axis too:
    concatenating two independently-quantized ``[N, K]`` tensors only
    produces a tensor whose blocks still exactly cover ``block`` rows each if
    ``gate_proj``'s own ``N`` (the MoE intermediate size) is itself a multiple
    of ``block`` -- true of every real block-FP8 checkpoint found so far
    (DeepSeek-V3's ``moe_intermediate_size`` is 2048, evenly divisible by its
    own ``weight_block_size`` of 128) -- asserted in :func:`_finalize_fp8_bank`
    rather than assumed, raising loudly instead of silently misaligning the
    fused scale table.

    Finalizes each ``(layer, bank_name)`` as soon as its last component
    arrives, mirroring :func:`stream_moe_expert_sources_gptq`'s incremental
    per-layer finalization -- never buffering more than one layer's worth of
    raw per-expert tensors at once (issue #145 found and fixed a real
    whole-checkpoint-buffering RAM bug in the GPTQ version's first draft; the
    same bug class is avoided here from the start, not retrofitted).
    """
    buf: Dict[Tuple[int, str], Dict[int, Dict[str, Dict[str, torch.Tensor]]]] = {}
    finalized: Dict[Tuple[int, str], Fp8BlockExpertBank] = {}

    for name, tensor in tensors:
        info = _parse_fp8_expert_key(name)
        if info is None:
            raise ValueError(f"Unexpected block-FP8 expert weight key: {name}")
        layer, proj, expert_id, component = info
        bank_name = "gate_up" if proj in ("gate_proj", "up_proj") else "down"
        if not (0 <= layer < config.num_layers):
            raise ValueError(f"Unexpected MoE expert layer {layer}; expected [0, {config.num_layers})")
        if not (0 <= expert_id < config.num_experts):
            raise ValueError(f"Unexpected MoE expert id {expert_id} in layer {layer}")
        key = (layer, bank_name)
        if key in finalized:
            raise ValueError(
                f"Layer {layer} {bank_name!r}: tensor {name!r} arrived after this bank was already "
                "finalized -- duplicate key or an out-of-order/re-streamed source?"
            )
        by_expert = buf.setdefault(key, {})
        by_proj = by_expert.setdefault(expert_id, {})
        by_component = by_proj.setdefault(proj, {})
        if component in by_component:
            raise ValueError(f"Duplicate block-FP8 component {component!r} for layer {layer} expert {expert_id} {proj}")
        by_component[component] = tensor

        fuse = bank_name == "gate_up"
        if _fp8_bank_is_complete(by_expert, config.num_experts, fuse=fuse):
            finalized[key] = _finalize_fp8_bank(by_expert, config.num_experts, fuse=fuse, layer=layer, block=block)
            del buf[key]  # release the raw per-expert tensors now that the packed bank owns the data

    missing = [
        (layer, bank_name)
        for layer in range(config.num_layers)
        for bank_name in ("gate_up", "down")
        if (layer, bank_name) not in finalized
    ]
    if missing:
        raise ValueError(f"Missing/incomplete block-FP8 MoE expert bank(s): {missing}")

    gate_up_banks = [finalized[(layer, "gate_up")] for layer in range(config.num_layers)]
    down_banks = [finalized[(layer, "down")] for layer in range(config.num_layers)]
    return gate_up_banks, down_banks


def _fp8_bank_is_complete(
    by_expert: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
    num_experts: int,
    *,
    fuse: bool,
) -> bool:
    """True once every expert 0..num_experts-1 has all required projections
    (``gate_proj``+``up_proj`` for ``fuse=True``, else ``down_proj``), each
    with both block-FP8 components -- i.e. this ``(layer, bank_name)`` is
    ready for :func:`_finalize_fp8_bank`."""
    if set(by_expert) != set(range(num_experts)):
        return False
    required_projs = ("gate_proj", "up_proj") if fuse else ("down_proj",)
    for e in range(num_experts):
        by_proj = by_expert[e]
        for proj in required_projs:
            components = by_proj.get(proj)
            if components is None or set(components) != set(_FP8_COMPONENTS):
                return False
    return True


def _finalize_fp8_bank(
    by_expert: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
    num_experts: int,
    *,
    fuse: bool,
    layer: int,
    block: int,
) -> Fp8BlockExpertBank:
    if set(by_expert) != set(range(num_experts)):
        missing = sorted(set(range(num_experts)) - set(by_expert))
        raise ValueError(f"Layer {layer}: missing block-FP8 experts {missing}")

    per_expert_rows = []
    for e in range(num_experts):
        by_proj = by_expert[e]
        if fuse:
            gate = by_proj.get("gate_proj")
            up = by_proj.get("up_proj")
            if gate is None or up is None:
                raise ValueError(f"Layer {layer} expert {e}: missing gate_proj/up_proj block-FP8 components")
            # See stream_moe_expert_sources_fp8's docstring for why this
            # alignment must hold before the N-axis concat below is safe.
            n_gate = gate["weight"].shape[0]
            if n_gate % block != 0:
                raise ValueError(
                    f"Layer {layer} expert {e}: gate_proj N={n_gate} is not a multiple of block={block}; "
                    "fusing gate_proj+up_proj on the N axis would misalign the fused weight_scale_inv"
                )
            weight = torch.cat([gate["weight"], up["weight"]], dim=0)
            weight_scale_inv = torch.cat([gate["weight_scale_inv"], up["weight_scale_inv"]], dim=0)
        else:
            down = by_proj.get("down_proj")
            if down is None:
                raise ValueError(f"Layer {layer} expert {e}: missing down_proj block-FP8 components")
            weight, weight_scale_inv = down["weight"], down["weight_scale_inv"]
        per_expert_rows.append((weight, weight_scale_inv))

    # _stack_expert_rows (not torch.stack): the torch XPU build mishandles a
    # direct cat/stack of 2-D per-expert rows along a new leading dim --
    # unsqueeze-then-cat is the reliable path this codebase already
    # standardizes on (see _stack_expert_rows's own docstring).
    return Fp8BlockExpertBank(
        weight=_stack_expert_rows([row[0] for row in per_expert_rows]),
        weight_scale_inv=_stack_expert_rows([row[1] for row in per_expert_rows]),
    )


@dataclass(frozen=True)
class MxfpExpertBank:
    """One MoE layer's packed MXFP4-quantized bank for one projection slot
    (``gate_up`` or ``down``) -- the MXFP4 sibling of :class:`GptqExpertBank`
    (issue `moe-quant-banks-mxfp4`, #153, part of epic #140).

    ``blocks`` / ``scales`` hold one row per expert (``[E, ...]``, OCP
    Microscaling's packed layout -- see
    :func:`freetoken.kernel.triton.mxfp4_linear.dequantize_mxfp4_blocks`):
    ``blocks`` is ``[E, out_features, K // 32, 16]`` ``uint8`` (16 packed
    bytes = 32 fp4 elements per block) and ``scales`` is ``[E, out_features,
    K // 32]`` ``uint8`` (one shared E8M0 exponent per block). Unlike GPTQ
    there is no fourth, non-per-expert component (no ``g_idx`` equivalent):
    MXFP4's block scale is fully local to its own block, not shared across a
    whole projection, so every component here is a genuine per-expert row.

    Deliberately packed, never dequantized here -- same RAM-budget rationale
    as :class:`GptqExpertBank`. Dequantization happens lazily, per-expert, at
    compute time (:class:`freetoken.moe.offload_cache.SlotWeightAccessor`)
    against whichever of these rows the offload cache's device slot pool has
    fetched for the current step.
    """

    blocks: torch.Tensor  # [E, out_features, K // 32, 16] uint8
    scales: torch.Tensor  # [E, out_features, K // 32] uint8


_MXFP4_COMPONENTS = ("blocks", "scales")


def _parse_mxfp4_expert_key(key: str) -> Tuple[int, str, str] | None:
    """Parse an MXFP4-packed per-layer weight key into ``(layer, proj,
    component)``, or ``None`` if ``key`` is not one.

    Recognizes ``...layers.{L}.mlp.experts.{gate_up_proj|down_proj}_{blocks|
    scales}`` -- confirmed against the real ``openai/gpt-oss-20b`` checkpoint's
    own ``model.safetensors.index.json``
    (``https://huggingface.co/openai/gpt-oss-20b/raw/main/model.safetensors.index.json``):
    e.g. ``model.layers.0.mlp.experts.gate_up_proj_blocks``. Two things this
    real spelling gets right that the issue's own guess (``.blocks`` /
    ``.scales``, a dot-separated trailing component like GPTQ's) got wrong:
    the separator is an underscore fused onto the projection name, not a
    dot-separated trailing key; and unlike GPTQ, the checkpoint ships each
    projection's experts as ONE already-packed ``[num_experts, ...]`` tensor
    per (layer, projection, component) -- never split per-expert-index, and
    ``gate_proj``/``up_proj`` are already fused into a single
    ``gate_up_proj`` tensor upstream (no separate halves to concatenate, per
    HF's own MXFP4-quantized GPT-OSS ``quantization_config`` convention).
    There is also a real ``..._bias`` sibling key in that checkpoint
    (per-expert MXFP4-unrelated bf16 bias) -- out of scope for this issue
    (packed-weight banks only) and not recognized here.
    """
    parts = key.split(".")
    if "mlp" not in parts or "experts" not in parts:
        return None
    try:
        mlp_pos = parts.index("mlp")
        layer = int(parts[mlp_pos - 1])
    except (ValueError, IndexError):
        return None
    if parts[mlp_pos + 1] != "experts":
        return None
    e_pos = parts.index("experts")
    tail = parts[e_pos + 1 :]
    if len(tail) != 1:
        return None
    token = tail[0]
    for proj, bank_name in (("gate_up_proj", "gate_up"), ("down_proj", "down")):
        for component in _MXFP4_COMPONENTS:
            suffix = f"{proj}_{component}"
            if token == suffix:
                return layer, bank_name, component
    return None


def stream_moe_expert_sources_mxfp4(
    tensors: Iterator[Tuple[str, torch.Tensor]],
    config,
) -> Tuple[list, list]:
    """Stream MXFP4-packed per-layer weight tensors into packed per-layer
    banks (issue `moe-quant-banks-mxfp4`, #153) -- the MXFP4 sibling of
    :func:`stream_moe_expert_sources_gptq`.

    Unlike GPTQ's raw checkpoint layout (one tensor per expert per
    component), a real MXFP4 checkpoint (verified against
    ``openai/gpt-oss-20b``) ships each ``(layer, projection, component)`` as
    ONE already-packed ``[num_experts, ...]`` tensor -- so there is no
    per-expert stacking to do here, only pairing each projection's
    ``blocks`` with its ``scales`` (the two components of one
    :class:`MxfpExpertBank`) as they stream in. Finalizes each ``(layer,
    bank_name)`` the moment both components have arrived -- immediately
    building the :class:`MxfpExpertBank` and releasing the raw references --
    rather than buffering the whole checkpoint's tensors until the stream
    ends, mirroring :func:`stream_moe_expert_sources_gptq`'s fix for the real
    RAM-blowup bug issue #145 found in an earlier draft of that function.

    Returns ``(gate_up_banks, down_banks)``, each a list of ``num_layers``
    :class:`MxfpExpertBank`. Every tensor stays in its packed, quantized
    form the whole way through -- never dequantized here.
    """
    buf: Dict[Tuple[int, str], Dict[str, torch.Tensor]] = {}
    finalized: Dict[Tuple[int, str], MxfpExpertBank] = {}

    for name, tensor in tensors:
        info = _parse_mxfp4_expert_key(name)
        if info is None:
            raise ValueError(f"Unexpected MXFP4 expert weight key: {name}")
        layer, bank_name, component = info
        if not (0 <= layer < config.num_layers):
            raise ValueError(f"Unexpected MoE expert layer {layer}; expected [0, {config.num_layers})")
        key = (layer, bank_name)
        if key in finalized:
            raise ValueError(
                f"Layer {layer} {bank_name!r}: tensor {name!r} arrived after this bank was already "
                "finalized -- duplicate key or an out-of-order/re-streamed source?"
            )
        by_component = buf.setdefault(key, {})
        if component in by_component:
            raise ValueError(f"Duplicate MXFP4 component {component!r} for layer {layer} {bank_name}")
        by_component[component] = tensor

        if set(by_component) == set(_MXFP4_COMPONENTS):
            blocks = by_component["blocks"]
            scales = by_component["scales"]
            if blocks.size(0) != config.num_experts:
                raise ValueError(
                    f"Layer {layer} {bank_name!r}: expected {config.num_experts} experts, got {blocks.size(0)}"
                )
            if tuple(blocks.shape[:-1]) != tuple(scales.shape):
                raise ValueError(
                    f"Layer {layer} {bank_name!r}: blocks/scales shape mismatch "
                    f"{tuple(blocks.shape[:-1])} vs {tuple(scales.shape)}"
                )
            finalized[key] = MxfpExpertBank(blocks=blocks, scales=scales)
            del buf[key]  # release the raw refs now that the packed bank owns the data

    missing = [
        (layer, bank_name)
        for layer in range(config.num_layers)
        for bank_name in ("gate_up", "down")
        if (layer, bank_name) not in finalized
    ]
    if missing:
        raise ValueError(f"Missing/incomplete MXFP4 MoE expert bank(s): {missing}")

    gate_up_banks = [finalized[(layer, "gate_up")] for layer in range(config.num_layers)]
    down_banks = [finalized[(layer, "down")] for layer in range(config.num_layers)]
    return gate_up_banks, down_banks


@dataclass(frozen=True)
class Int8ExpertBank:
    """One MoE layer's packed compressed-tensors ``pack-quantized`` INT8 bank
    for one projection slot (``gate_up`` or ``down``) -- the INT8 sibling of
    :class:`GptqExpertBank` (issue `moe-quant-banks-int8`, #154, part of
    epic #140).

    Corrected from an earlier, unverified draft (see issue #154's own
    comment trail): confirmed against a REAL deployed MoE checkpoint,
    ``rj1013/gemma-4-26B-A4B-it_q8`` (Gemma-4-26B-A4B, ``num_experts: 128``,
    ``quantization_config.quant_method: "compressed-tensors"``,
    ``format: "pack-quantized"``), and cross-checked bit-exact against the
    real ``compressed_tensors`` pip package's own ``unpack_from_int32`` /
    ``dequantize`` on that checkpoint's actual tensor bytes (fetched via
    HTTP range request) -- not the plain unpacked-int8 format the earlier
    draft guessed.

    ``weight_packed`` / ``weight_scale`` hold one row per expert (``[E,
    ...]``). ``weight_packed`` is ``[E, N, ceil(K/4)]`` int32 (4 int8 values
    densely packed per word along the K axis -- see
    :mod:`freetoken.kernel.triton.int8_packed_linear` for the full bit-layout
    docs). ``weight_scale`` is ``[E, N, num_groups]``: ``num_groups == 1`` is
    pure per-channel (this issue's original title), ``num_groups > 1`` is
    sequential-group (the real checkpoint found uses ``group_size=32`` for
    its MoE experts) -- both are the same on-disk mechanism, so one bank
    shape covers both. ``k`` (the real logical in-features, from the
    checkpoint's own ``weight_shape`` tensor) is shared across every expert
    of one projection type within a layer (same rationale as GPTQ's shared
    ``g_idx``: it is an architecture constant, not a per-expert value), so it
    is carried once here rather than duplicated ``E`` times.

    Deliberately packed, never dequantized here -- same RAM-blowup rationale
    as :class:`GptqExpertBank` (issue #134): dequantization happens lazily,
    per-expert, at compute time (:class:`freetoken.moe.offload_cache.
    SlotWeightAccessor`) against whichever rows the offload cache's device
    slot pool has fetched for the current step.
    """

    weight_packed: torch.Tensor  # [E, N, ceil(K/4)] int32
    weight_scale: torch.Tensor  # [E, N, num_groups]
    k: int  # real logical in-features, shared across every expert/component here


_INT8_COMPONENTS = ("weight_packed", "weight_scale", "weight_shape")


def _parse_int8_expert_key(key: str) -> Tuple[int, str, int, str] | None:
    """Parse a compressed-tensors pack-quantized INT8 per-expert weight key
    into ``(layer, proj, expert_id, component)``, or ``None`` if ``key`` is
    not one.

    Recognizes ``...layers.{L}.experts.{e}.{gate_proj|up_proj|down_proj}
    .{weight_packed|weight_scale|weight_shape}`` -- confirmed against the
    real ``rj1013/gemma-4-26B-A4B-it_q8`` checkpoint's own
    ``model.safetensors.index.json`` on HuggingFace: e.g.
    ``model.language_model.layers.1.experts.0.gate_proj.weight_packed`` is a
    real key in its index. Note this real checkpoint has NO ``mlp.`` segment
    between ``layers.{L}`` and ``experts`` (unlike GPTQ's ``layers.{L}.mlp.
    experts.{e}...``) -- also tolerates an intervening ``mlp`` token (i.e.
    ``layers.{L}.mlp.experts.{e}...``) since that is GPTQ's own real,
    confirmed convention and a future INT8 checkpoint may follow it instead;
    only the no-``mlp`` form is confirmed for INT8 specifically so far.
    """
    parts = key.split(".")
    if len(parts) < 2 or parts[-1] not in _INT8_COMPONENTS:
        return None
    component = parts[-1]
    body = parts[:-1]
    if "experts" not in body:
        return None
    e_pos = body.index("experts")
    if e_pos == 0:
        return None
    layer_pos = e_pos - 2 if body[e_pos - 1] == "mlp" else e_pos - 1
    if layer_pos < 0:
        return None
    try:
        layer = int(body[layer_pos])
    except ValueError:
        return None
    tail = body[e_pos + 1 :]
    if len(tail) != 2:
        return None
    try:
        expert_id = int(tail[0])
    except ValueError:
        return None
    proj = tail[1]
    if proj not in {"gate_proj", "up_proj", "down_proj"}:
        return None
    return layer, proj, expert_id, component


def stream_moe_expert_sources_int8(
    tensors: Iterator[Tuple[str, torch.Tensor]],
    config,
) -> Tuple[list, list]:
    """Stream compressed-tensors pack-quantized INT8 per-expert weight
    tensors into packed per-layer banks (issue `moe-quant-banks-int8`, #154)
    -- the INT8 sibling of :func:`stream_moe_expert_sources_gptq`.

    ``gate_up`` fuses ``gate_proj`` + ``up_proj`` (the checkpoint stores them
    as separate quantized tensors, not pre-fused): ``weight_packed`` /
    ``weight_scale`` concatenate along their ``N`` (output-channel, dim 0)
    axis -- both are per-row, and packing/grouping is entirely along the
    ORTHOGONAL ``K`` axis, so N-axis concatenation never crosses a group or
    word boundary (unlike block-FP8's ``N``-axis blocking, #152, which does
    need an alignment check here). ``k`` (from ``weight_shape``) must agree
    between ``gate_proj`` and ``up_proj`` (both take the same hidden_size
    input) -- asserted, not assumed.

    Returns ``(gate_up_banks, down_banks)``, each a list of ``num_layers``
    :class:`Int8ExpertBank`. Every tensor stays in its packed, quantized form
    the whole way through -- never dequantized here.

    Finalizes each ``(layer, bank_name)`` as soon as its last component
    arrives, exactly like :func:`stream_moe_expert_sources_gptq` -- never
    buffering more than one layer's worth of raw tensors per bank at once
    (issue #145 found and fixed a real whole-checkpoint-buffering RAM bug in
    the GPTQ streamer's first draft; this mirrors the fixed version, not the
    buggy one).
    """
    buf: Dict[Tuple[int, str], Dict[int, Dict[str, Dict[str, torch.Tensor]]]] = {}
    finalized: Dict[Tuple[int, str], Int8ExpertBank] = {}

    for name, tensor in tensors:
        info = _parse_int8_expert_key(name)
        if info is None:
            raise ValueError(f"Unexpected INT8 expert weight key: {name}")
        layer, proj, expert_id, component = info
        bank_name = "gate_up" if proj in ("gate_proj", "up_proj") else "down"
        if not (0 <= layer < config.num_layers):
            raise ValueError(f"Unexpected MoE expert layer {layer}; expected [0, {config.num_layers})")
        if not (0 <= expert_id < config.num_experts):
            raise ValueError(f"Unexpected MoE expert id {expert_id} in layer {layer}")
        key = (layer, bank_name)
        if key in finalized:
            raise ValueError(
                f"Layer {layer} {bank_name!r}: tensor {name!r} arrived after this bank was already "
                "finalized -- duplicate key or an out-of-order/re-streamed source?"
            )
        by_expert = buf.setdefault(key, {})
        by_proj = by_expert.setdefault(expert_id, {})
        by_component = by_proj.setdefault(proj, {})
        if component in by_component:
            raise ValueError(f"Duplicate INT8 component {component!r} for layer {layer} expert {expert_id} {proj}")
        by_component[component] = tensor

        fuse = bank_name == "gate_up"
        if _int8_bank_is_complete(by_expert, config.num_experts, fuse=fuse):
            finalized[key] = _finalize_int8_bank(by_expert, config.num_experts, fuse=fuse, layer=layer)
            del buf[key]  # release the raw per-expert tensors now that the packed bank owns the data

    missing = [
        (layer, bank_name)
        for layer in range(config.num_layers)
        for bank_name in ("gate_up", "down")
        if (layer, bank_name) not in finalized
    ]
    if missing:
        raise ValueError(f"Missing/incomplete INT8 MoE expert bank(s): {missing}")

    gate_up_banks = [finalized[(layer, "gate_up")] for layer in range(config.num_layers)]
    down_banks = [finalized[(layer, "down")] for layer in range(config.num_layers)]
    return gate_up_banks, down_banks


def _int8_bank_is_complete(
    by_expert: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
    num_experts: int,
    *,
    fuse: bool,
) -> bool:
    """True once every expert 0..num_experts-1 has all required projections
    (``gate_proj``+``up_proj`` for ``fuse=True``, else ``down_proj``), each
    with all three INT8 components -- i.e. this ``(layer, bank_name)`` is
    ready for :func:`_finalize_int8_bank`."""
    if set(by_expert) != set(range(num_experts)):
        return False
    required_projs = ("gate_proj", "up_proj") if fuse else ("down_proj",)
    for e in range(num_experts):
        by_proj = by_expert[e]
        for proj in required_projs:
            components = by_proj.get(proj)
            if components is None or set(components) != set(_INT8_COMPONENTS):
                return False
    return True


def _int8_logical_k(weight_shape: torch.Tensor) -> int:
    """The real logical in-features from a ``weight_shape`` tensor (``[2]``
    int64, ``[out_features, in_features]`` -- see ``Int8ExpertBank``'s own
    docstring for why this is stored separately from ``weight_packed``'s own
    shape rather than derived from it)."""
    return int(weight_shape[1].item())


def _finalize_int8_bank(
    by_expert: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
    num_experts: int,
    *,
    fuse: bool,
    layer: int,
) -> Int8ExpertBank:
    if set(by_expert) != set(range(num_experts)):
        missing = sorted(set(range(num_experts)) - set(by_expert))
        raise ValueError(f"Layer {layer}: missing INT8 experts {missing}")

    per_expert_rows = []
    k0 = None
    for e in range(num_experts):
        by_proj = by_expert[e]
        if fuse:
            gate = by_proj.get("gate_proj")
            up = by_proj.get("up_proj")
            if gate is None or up is None:
                raise ValueError(f"Layer {layer} expert {e}: missing gate_proj/up_proj INT8 components")
            k_gate = _int8_logical_k(gate["weight_shape"])
            k_up = _int8_logical_k(up["weight_shape"])
            if k_gate != k_up:
                raise ValueError(
                    f"Layer {layer} expert {e}: gate_proj K={k_gate} != up_proj K={k_up} "
                    "(both project the same hidden_size input; a mismatch means a wiring bug)"
                )
            k = k_gate
            weight_packed = torch.cat([gate["weight_packed"], up["weight_packed"]], dim=0)
            weight_scale = torch.cat([gate["weight_scale"], up["weight_scale"]], dim=0)
        else:
            down = by_proj.get("down_proj")
            if down is None:
                raise ValueError(f"Layer {layer} expert {e}: missing down_proj INT8 components")
            k = _int8_logical_k(down["weight_shape"])
            weight_packed, weight_scale = down["weight_packed"], down["weight_scale"]
        if k0 is None:
            k0 = k
        elif k != k0:
            raise ValueError(
                f"Layer {layer}: expert {e}'s K={k} differs from expert 0's K={k0} "
                "(K is an architecture constant, expected identical across every expert)"
            )
        per_expert_rows.append((weight_packed, weight_scale))

    # _stack_expert_rows (not torch.stack): the torch XPU build mishandles a
    # direct cat/stack of 2-D per-expert rows along a new leading dim (see
    # _stack_expert_rows's own docstring).
    return Int8ExpertBank(
        weight_packed=_stack_expert_rows([row[0] for row in per_expert_rows]),
        weight_scale=_stack_expert_rows([row[1] for row in per_expert_rows]),
        k=k0,
    )


__all__ = [
    "load_weight",
    "load_moe_expert_sources",
    "dummy_moe_expert_sources",
    "iter_safetensors",
    "_PlainBank",
    "GptqExpertBank",
    "stream_moe_expert_sources_gptq",
    "Fp8BlockExpertBank",
    "stream_moe_expert_sources_fp8",
    "MxfpExpertBank",
    "stream_moe_expert_sources_mxfp4",
    "Int8ExpertBank",
    "stream_moe_expert_sources_int8",
    "checkpoint_quant_method",
    "checkpoint_gptq_group_size",
]
