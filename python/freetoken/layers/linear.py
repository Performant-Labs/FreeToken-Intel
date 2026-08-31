"""Pure-torch tensor-parallel Linear layers for the B70 XPU port.

Upstream (NVIDIA) ``freetoken.layers.linear`` defines the same five classes but
imports ``torch`` and ``torch.nn.functional`` at module top and relies on CUDA
``DistributedCommunicator.all_reduce`` for the TP>1 paths. The B70 runs TP=1
(single XPU device), so the ``all_reduce`` branch never fires and the only thing
that matters is the matmul math: ``F.linear(x, weight, bias) == x @ weight.T (+ bias)``.

``import torch`` is deferred so ``import freetoken.layers`` stays torch-free in
the CPU venv (the dual-venv contract). The ``torch.nn.Module`` base class is
resolved lazily too: the module-level ``_nn_module_base()`` returns the real
``nn.Module`` once torch is importable (the XPU venv) and returns ``object`` in
the torch-free CPU venv. This makes ``_LinearTPImpl`` subclass ``nn.Module`` on
the XPU venv -- which is what lets a model whose parent ``nn.Module`` subclasses
*rebind lazily* (``_ensure_torch``, e.g. qwen3_5_moe) register these Linears as
submodules: ``nn.Module``'s ``__setattr__`` only registers a child that
``isinstance(module, nn.Module)``, and a plain-``BaseOP`` Linear is not. Without
the ``nn.Module`` base, a Linear built inside a not-yet-rebound parent would be
invisible to the parent's ``named_parameters()`` (the loader could not fill it)
and ``.to(device)`` (it would stay on the CPU). On the CPU venv the base is
``object`` (no ``_modules`` / ``register_parameter``), so the classes remain
torch-free and are only *instantiable* once torch is present.
"""
from __future__ import annotations

from typing import List, Optional

from .base import BaseOP

# TP info is a lazily-set global (see freetoken.distributed.info). The engine sets
# it when a process is launched in a distributed group; a standalone model build
# (``load_model`` in a unit test, the CPU reference path, or a single-device serve)
# never does. Rather than hard-require it, these layers fall back to rank 0 / TP
# size 1 -- the B70's single-device reality -- so the classes are always
# constructible and the TP>1 all_reduce branch (dead on TP=1) simply stays dormant.
from freetoken.distributed import try_get_tp_info


class _TorchMixin:
    """Placeholder base for ``_LinearTPImpl`` when torch is not importable.

    A plain ``object`` base would not linearize: ``(object, BaseOP)`` is an
    inconsistent MRO (``BaseOP`` already inherits from ``object``). A fresh mixin
    (base ``object``) instead gives the clean MRO ``[cls, _TorchMixin, BaseOP,
    object]``, so the class body is valid in the torch-free CPU venv. On the XPU
    venv the real ``nn.Module`` is used instead (see :func:`_nn_module_base`).
    """


def _nn_module_base():
    """Return the base class ``_LinearTPImpl`` subclasses.

    ``torch.nn.Module`` when torch is importable (the XPU venv), else the
    :class:`_TorchMixin` placeholder (the torch-free CPU venv). Resolved at
    class-creation time: the model / test code that triggers the import runs
    ``import torch`` first, so ``nn.Module`` is present exactly where a Linear is
    actually constructed (the XPU venv).
    """
    try:
        import torch.nn as nn

        return nn.Module
    except Exception:
        return _TorchMixin


def _tp_rank_size():
    """Return ``(rank, tp_size)``: the set global if present, else the TP=1 default."""
    info = try_get_tp_info()
    if info is None:
        return 0, 1
    return info.rank, info.size


__all__ = [
    "_nn_module_base",
    "LinearReplicated",
    "LinearColParallelMerged",
    "LinearQKVMerged",
    "LinearOProj",
    "LinearRowParallel",
]


class _LinearTPImpl(_nn_module_base(), BaseOP):
    """Base tensor-parallel linear layer (weights sharded per the TP layout).

    Mirrors the upstream class: holds ``weight`` (``[local_osize, local_isize]``)
    and an optional ``bias`` (``[local_osize]``) as ``nn.Parameter`` so the layers
    stay drop-in for an ``nn.Module`` parent -- the checkpoint loader fills them
    via ``named_parameters()`` + ``param.copy_()`` and moves them with the model's
    ``.to(device)`` (a plain tensor would be invisible to both). ``forward`` is
    overridden by the row-parallel subclasses that add a TP>1 all_reduce.
    """

    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
        dtype: Optional[object] = None,
    ) -> None:
        # ``super().__init__`` resolves to ``nn.Module.__init__`` on the XPU venv
        # (the ``_nn_module_base()`` base) -- that is what makes the layer a real
        # ``nn.Module`` so a (lazily rebound) parent registers it as a submodule.
        # On the CPU venv the base is ``object``; ``object.__init__`` no-ops, so
        # the line is torch-free there and torch is imported only to allocate.
        super().__init__()
        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        self.has_bias = has_bias
        # Lazy import: torch is only needed to allocate the buffers. Registering
        # the weights as ``nn.Parameter`` (not plain tensors) keeps the layers
        # drop-in for an ``nn.Module`` parent: the checkpoint loader fills them
        # via ``named_parameters()`` + ``param.copy_()`` and moves them with the
        # model's ``.to(device)`` -- a plain tensor would be invisible to
        # ``named_parameters()`` (leaving weights at random init) and would not
        # be moved by ``.to``.
        import torch
        import torch.nn as nn

        # Allocate in the model's dtype (defaults to float32) so the weight
        # matches the (possibly bf16) hidden states at ``F.linear`` time -- the
        # loader then fills the buffer in-place (``param.copy_``) without a dtype
        # cast, and a bf16 weight would be invisible to a float32-input matmul.
        weight_dtype = dtype if dtype is not None else torch.float32
        self.weight = nn.Parameter(torch.empty(local_osize, local_isize, dtype=weight_dtype))
        self.bias = nn.Parameter(torch.empty(local_osize, dtype=weight_dtype)) if has_bias else None

    def __call__(self, x, out: Optional[object] = None):
        # ``nn.Module.__call__`` is what a parent module relies on when the model
        # invokes a layer positionally (e.g. ``self.q_proj(hidden)``). ``BaseOP``
        # exposes only ``forward`` (the pure-torch unit tests call ``.forward()``
        # directly), so without this the layers are not callable and the model's
        # attention/FFN call sites raise ``TypeError: ... not callable``. Delegates
        # to ``forward`` (which subclasses override), so TP>1 all_reduce still runs.
        return self.forward(x, out)

    def forward(self, x, out: Optional[object] = None):
        import torch.nn.functional as F

        y = F.linear(x, self.weight, self.bias)
        if out is not None:
            out.copy_(y)
            return out
        return y


class LinearReplicated(_LinearTPImpl):
    """Linear layer whose weights are replicated (not sharded) across TP ranks.

    Each device holds the full ``[output_size, input_size]`` weight, so the local
    and full sizes are identical and no all_reduce is needed in ``forward``.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
        dtype: Optional[object] = None,
    ) -> None:
        super().__init__(
            full_isize=input_size,
            full_osize=output_size,
            local_isize=input_size,
            local_osize=output_size,
            has_bias=has_bias,
            dtype=dtype,
        )


class LinearColParallelMerged(_LinearTPImpl):
    """Column-parallel linear over merged output heads (e.g. gate+up).

    The full output is ``sum(output_sizes)``; each rank holds the corresponding
    ``div_even(size, tp)`` slice of every output, so the local output size is
    ``sum(div_even(s, tp) for s in output_sizes)``.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        has_bias: bool,
        dtype: Optional[object] = None,
    ) -> None:
        # TP plumbing (try_get_tp_info / div_even) is torch-free; only the buffer
        # allocation in the base __init__ needs torch.
        from freetoken.utils import div_even

        _, tp_size = _tp_rank_size()
        tp_output_sizes = [div_even(size, tp_size) for size in output_sizes]
        output_size = sum(output_sizes)
        tp_output_size = sum(tp_output_sizes)
        super().__init__(input_size, output_size, input_size, tp_output_size, has_bias, dtype=dtype)


class LinearQKVMerged(_LinearTPImpl):
    """Merged q/k/v projection (QKV) with column-parallel head sharding."""

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
        dtype: Optional[object] = None,
    ) -> None:
        from freetoken.utils import div_even

        _, tp_size = _tp_rank_size()
        local_num_qo = div_even(num_qo_heads, tp_size)
        local_num_kv = div_even(num_kv_heads, tp_size, allow_replicate=True)
        full_isize = hidden_size
        full_osize = (num_qo_heads + 2 * num_kv_heads) * head_dim
        local_isize = hidden_size
        local_osize = (local_num_qo + 2 * local_num_kv) * head_dim
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias, dtype=dtype)


class LinearOProj(_LinearTPImpl):
    """Row-parallel output projection (the attention ``o_proj``).

    The input is column-sharded across TP ranks (each rank sees a slice of the
    head dim), so the local input size is ``div_even(input_size, tp)`` and the
    full output is unsharded; a TP>1 all_reduce combines the partial sums.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
        dtype: Optional[object] = None,
    ) -> None:
        from freetoken.utils import div_even

        _, tp_size = _tp_rank_size()
        self._tp_size = tp_size
        super().__init__(input_size, output_size, div_even(input_size, tp_size), output_size, has_bias, dtype=dtype)

    def forward(self, x, out: Optional[object] = None):
        import torch.nn.functional as F

        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            from freetoken.distributed import DistributedCommunicator

            y = DistributedCommunicator().all_reduce(y)
        if out is not None:
            out.copy_(y)
            return out
        return y


class LinearRowParallel(_LinearTPImpl):
    """Row-parallel linear (e.g. the down projection of a merged FFN).

    The input is column-sharded (local input ``div_even(input_size, tp)``), the
    output is unsharded, and a TP>1 all_reduce combines the per-rank partials.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
        dtype: Optional[object] = None,
    ) -> None:
        from freetoken.utils import div_even

        _, tp_size = _tp_rank_size()
        self._tp_size = tp_size
        super().__init__(input_size, output_size, div_even(input_size, tp_size), output_size, has_bias, dtype=dtype)

    def forward(self, x, out: Optional[object] = None):
        import torch.nn.functional as F

        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            from freetoken.distributed import DistributedCommunicator

            y = DistributedCommunicator().all_reduce(y)
        if out is not None:
            out.copy_(y)
            return out
        return y
