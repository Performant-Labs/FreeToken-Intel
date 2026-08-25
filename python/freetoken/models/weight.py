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
from typing import Iterator, Tuple

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
    """
    folder = download_hf_weight(model_path)
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
    for name, path in map_.items():
        with safe_open(path, framework="pt", device=str(device)) as f:
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
) -> Tuple[list, list]:
    """Per-layer MoE expert source banks: ``(gate_up_banks, down_banks)``.

    Each bank is a per-layer ``[num_experts, ...]`` tensor exposing ``.tensor`` /
    ``.pin()`` (see :class:`_PlainBank`). ``dummy=True`` fabricates the banks
    from the parsed config without reading the checkpoint.
    """
    config, spec = _spec_for_model_path(model_path)
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
    return stream_moe_expert_sources(src, config, dtype=dtype, layer_sink=layer_sink)


def _spec_for_model_path(model_path: str):
    hf_config = cached_load_hf_config(model_path)
    spec = get_model_spec(hf_config.architectures[0])
    parse_config = _load_attr(spec.module, spec.parse_config)
    return parse_config(hf_config), spec


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
    """Stream packed per-layer MoE expert tensors into final offload banks.

    The model's ``iter_weights`` normalizes expert weights to
    ``...experts.gate_up_proj`` / ``...experts.down_proj`` with shape
    ``[num_experts, ...]``. Each arrives whole-layer, so it is written straight
    into its own ``[num_experts, ...]`` per-layer bank. Returns
    ``(gate_up_banks, down_banks)`` (per-layer bank objects).
    """
    del layer_sink  # converter-facing sink not used on the B70 port
    banks: dict[str, list] = {
        "gate_up": [None] * config.num_layers,
        "down": [None] * config.num_layers,
    }
    row_shape: dict[str, tuple[int, ...]] = {}
    seen: dict[str, set[int]] = {"gate_up": set(), "down": set()}

    for name, tensor in tensors:
        info = _packed_expert_source_info(name)
        if info is None:
            raise ValueError(f"Unexpected expert weight key: {name}")
        layer, packed_name = info
        bank_name = "gate_up" if packed_name == "gate_up_proj" else "down"
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

    missing = {
        name: sorted(set(range(config.num_layers)) - seen)
        for name, seen in seen.items()
        if seen != set(range(config.num_layers))
    }
    if missing:
        raise ValueError(f"Missing MoE expert source layers: {missing}")
    return (
        [bank.tensor for bank in banks["gate_up"]],
        [bank.tensor for bank in banks["down"]],
    )


def _packed_expert_source_info(key: str) -> Tuple[int, str] | None:
    """Map a HF MoE expert key to ``(layer, packed_name)`` or None.

    Accepts the Qwen-style ``model.layers.{L}.mlp.experts.{e}.{proj}`` (per-expert,
    projected) keys and the packed ``...experts.{gate_up_proj|down_proj}`` (already
    stacked to ``[num_experts, ...]``) keys the model adapter normalizes to.
    """
    parts = key.split(".")
    if len(parts) < 5 or parts[0] != "model" or parts[1] != "layers":
        return None
    if parts[-2] != "experts" or parts[-1] not in {"gate_up_proj", "down_proj"}:
        return None
    try:
        return int(parts[2]), parts[-1]
    except ValueError:
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


__all__ = [
    "load_weight",
    "load_moe_expert_sources",
    "dummy_moe_expert_sources",
    "iter_safetensors",
    "_PlainBank",
]
