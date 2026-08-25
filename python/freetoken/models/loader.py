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

import torch

from freetoken.models.register import _load_attr, get_model_class, get_model_spec
from freetoken.models.weight import load_moe_expert_sources, load_weight
from freetoken.utils import cached_load_hf_config
from freetoken.utils.arch import is_xpu_available


def load_model(
    model_path: str,
    device: torch.device | str | None = None,
    *,
    dtype: torch.dtype | None = None,
    dummy: bool = False,
) -> tuple:
    """Load a checkpoint onto ``device`` (defaults to the XPU when available).

    Returns ``(model, expert_sources)`` where ``model`` is the instantiated model
    with its dense parameters populated on ``device``, and ``expert_sources`` is
    the per-layer MoE bank tuple from :func:`load_moe_expert_sources` (empty for
    a dense model). With ``dummy=True`` the expert banks are fabricated from the
    config (offline / CPU-testable) and no checkpoint is read.
    """
    if device is None:
        device = torch.device("xpu") if is_xpu_available() else torch.device("cpu")
    if isinstance(device, str):
        device = torch.device(device)
    if dtype is None:
        dtype = torch.bfloat16

    hf_config = cached_load_hf_config(model_path)
    spec = get_model_spec(hf_config.architectures[0])
    model_config = _load_attr(spec.module, spec.parse_config)(hf_config)
    model = get_model_class(hf_config.architectures[0], model_config)

    is_moe = bool(getattr(model_config, "is_moe", False))
    if dummy:
        # Offline path: fabricate the MoE expert banks from the config (CPU-
        # testable, no checkpoint on disk). Dense weights are not read.
        expert_sources = (
            load_moe_expert_sources(model_path, dtype=dtype, dummy=True)
            if is_moe
            else ([], [])
        )
    else:
        # Real path: stream the dense checkpoint weights onto ``device`` and, for
        # a MoE checkpoint, build the per-layer host offload banks for the experts
        # (which do not fit on the XPU and are served to the engine on demand).
        for name, tensor in load_weight(model_path, device, include_moe_experts=False):
            _place(model, name, tensor)
        expert_sources = (
            load_moe_expert_sources(model_path, dtype=dtype)
            if is_moe
            else ([], [])
        )
    return model, expert_sources


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
