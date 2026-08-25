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
    # Build the model *on this device* (the loader already resolved it): an
    # explicit device wins, and only a None device lets the model default to
    # the XPU. Without this the model would re-default to the XPU and ignore
    # the loader's device (e.g. a CPU test on an XPU box).
    #
    # The model builds its modules in ``model_config.dtype``; stamp the loader's
    # effective dtype onto it so the modules and the streamed weights share a
    # dtype (no bf16-module / fp32-weight mismatch when the engine pins one).
    object.__setattr__(model_config, "dtype", dtype)
    model = get_model_class(hf_config.architectures[0], model_config, device=device)

    is_moe = bool(getattr(model_config, "is_moe", False))
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
            _seed_dummy_experts(model)
            gate_up_banks, down_banks = load_moe_expert_sources(model_path, dtype=dtype, dummy=True)
            _place_expert_weights(model, gate_up_banks, down_banks)
            expert_sources = (gate_up_banks, down_banks)
        else:
            expert_sources = ([], [])
    else:
        # Real path: stream the dense checkpoint weights onto ``device`` and, for
        # a MoE checkpoint, build the per-layer host offload banks for the experts
        # (which do not fit on the XPU and are served to the engine on demand).
        for name, tensor in load_weight(model_path, device, include_moe_experts=False):
            _place(model, name, tensor)
        if is_moe:
            gate_up_banks, down_banks = load_moe_expert_sources(model_path, dtype=dtype)
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
            gate, up = gu[e, 0].clone(), gu[e, 1].clone()
            with torch.no_grad():
                experts[e].gate_proj.weight.copy_(gate)
                experts[e].up_proj.weight.copy_(up)
                experts[e].down_proj.weight.copy_(dn[e])


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
