"""Pure-torch tensor-parallel Linear layers for the B70 XPU port.

Upstream (NVIDIA) ``freetoken.layers.linear`` defines the same five classes but
imports ``torch`` and ``torch.nn.functional`` at module top and relies on CUDA
``DistributedCommunicator.all_reduce`` for the TP>1 paths. The B70 runs TP=1
(single XPU device), so the ``all_reduce`` branch never fires and the only thing
that matters is the matmul math: ``F.linear(x, weight, bias) == x @ weight.T (+ bias)``.

``import torch`` is deferred into the methods so ``import freetoken.layers``
stays torch-free in the CPU venv (the dual-venv contract). The weight tensors are
still allocated here (``torch.empty``), so the classes are only *usable* once torch
is importable; they are imported lazily by the model / test code on the XPU venv.
"""
from __future__ import annotations

from typing import List, Optional

from .base import BaseOP

__all__ = [
    "LinearReplicated",
    "LinearColParallelMerged",
    "LinearQKVMerged",
    "LinearOProj",
    "LinearRowParallel",
]


class _LinearTPImpl(BaseOP):
    """Base tensor-parallel linear layer (weights sharded per the TP layout).

    Mirrors the upstream class: holds ``weight`` (``[local_osize, local_isize]``)
    and an optional ``bias`` (``[local_osize]``) as plain tensors so the
    torch-free ``BaseOP.state_dict`` / ``load_state_dict`` machinery can fill
    them in. ``forward`` is overridden by the parallel subclasses that need a
    TP>1 all_reduce.
    """

    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
    ) -> None:
        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        self.has_bias = has_bias
        # Lazy import: torch is only needed to allocate the buffers.
        import torch

        self.weight = torch.empty(local_osize, local_isize)
        self.bias = torch.empty(local_osize) if has_bias else None

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

    def __init__(self, input_size: int, output_size: int, has_bias: bool) -> None:
        super().__init__(
            full_isize=input_size,
            full_osize=output_size,
            local_isize=input_size,
            local_osize=output_size,
            has_bias=has_bias,
        )


class LinearColParallelMerged(_LinearTPImpl):
    """Column-parallel linear over merged output heads (e.g. gate+up).

    The full output is ``sum(output_sizes)``; each rank holds the corresponding
    ``div_even(size, tp)`` slice of every output, so the local output size is
    ``sum(div_even(s, tp) for s in output_sizes)``.
    """

    def __init__(self, input_size: int, output_sizes: List[int], has_bias: bool) -> None:
        # TP plumbing (get_tp_info / div_even) is torch-free; only the buffer
        # allocation in the base __init__ needs torch.
        from freetoken.distributed import get_tp_info
        from freetoken.utils import div_even

        tp_info = get_tp_info()
        tp_output_sizes = [div_even(size, tp_info.size) for size in output_sizes]
        output_size = sum(output_sizes)
        tp_output_size = sum(tp_output_sizes)
        super().__init__(input_size, output_size, input_size, tp_output_size, has_bias)


class LinearQKVMerged(_LinearTPImpl):
    """Merged q/k/v projection (QKV) with column-parallel head sharding."""

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
    ) -> None:
        from freetoken.distributed import get_tp_info
        from freetoken.utils import div_even

        tp_info = get_tp_info()
        local_num_qo = div_even(num_qo_heads, tp_info.size)
        local_num_kv = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        full_isize = hidden_size
        full_osize = (num_qo_heads + 2 * num_kv_heads) * head_dim
        local_isize = hidden_size
        local_osize = (local_num_qo + 2 * local_num_kv) * head_dim
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)


class LinearOProj(_LinearTPImpl):
    """Row-parallel output projection (the attention ``o_proj``).

    The input is column-sharded across TP ranks (each rank sees a slice of the
    head dim), so the local input size is ``div_even(input_size, tp)`` and the
    full output is unsharded; a TP>1 all_reduce combines the partial sums.
    """

    def __init__(self, input_size: int, output_size: int, has_bias: bool) -> None:
        from freetoken.distributed import get_tp_info
        from freetoken.utils import div_even

        tp_info = get_tp_info()
        self._tp_size = tp_info.size
        super().__init__(input_size, output_size, div_even(input_size, tp_info.size), output_size, has_bias)

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

    def __init__(self, input_size: int, output_size: int, has_bias: bool) -> None:
        from freetoken.distributed import get_tp_info
        from freetoken.utils import div_even

        tp_info = get_tp_info()
        self._tp_size = tp_info.size
        super().__init__(input_size, output_size, div_even(input_size, tp_info.size), output_size, has_bias)

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
