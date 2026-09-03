"""Stacked-tensor GDN (Gated Delta Net) recurrent state pool: ping-pong
copy-on-write slots, TP-aware sizing.

Upstream NVIDIA path: python/freetoken/kvcache/linear_state_pool.py
Issue: `semantic-cache-linear-pool` (#170, part of the `semantic-cache`
epic, #32).

Ported from the real upstream source (read directly, not guessed), with
one deliberate scope cut: upstream's constructor takes a
``LinearGatedDeltaGroupConfig`` (part of a much larger polymorphic
``AttentionGroupConfig`` hierarchy -- full/SWA/linear/DSV4 attention
groups, ``ModelConfig.linear_attention_group()``, etc.) that this port's
own ``ModelConfig`` has no equivalent of yet (only a bare, unfilled
``LinearGatedDeltaGroupConfig`` stub exists, in ``models/config.py``) --
building that whole abstraction is real, separate, much larger work with
its own scope, not needed to deliver THIS issue's own actual value (the
ping-pong COW slot pool). This module instead takes the raw GDN dimensions
directly (``num_key_heads``, ``num_value_heads``, etc.) -- the exact same
sharding math (:func:`_linear_local_dims`), just not gated behind a config
object this port doesn't have a real one of yet. A future wiring pass
(issue #171 or later) can still build a thin adapter from a real
``ModelConfig`` once one exists, without changing this module's own
tested core.

This is the STACKED-TENSOR design (``[n_layers, num_slots, ...]``, one
pool object) upstream uses -- distinct from (and NOT wired in place of)
this port's own existing ``qwen3_5_moe.__init__._LinearStatePool`` (a
simpler, list-of-per-layer-dicts design with no ping-pong/COW at all).
Wiring the model's ``_GatedDeltaNet`` layer to read/write through THIS
pool instead is real follow-up work (issue #171's own scope), not done
here -- this issue delivers the pool itself, tested standalone.
"""
from __future__ import annotations

import torch

from freetoken.env import ENV
from freetoken.utils import div_even

_SSM_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def ssm_state_dtype() -> torch.dtype:
    """Recurrent (SSM) state dtype, from FREETOKEN_MAMBA_SSM_DTYPE (default fp32)."""
    return _SSM_DTYPES.get(str(ENV.MAMBA_SSM_DTYPE).lower(), torch.float32)


def _linear_local_dims(
    num_key_heads: int, num_value_heads: int, key_head_dim: int, value_head_dim: int, tp_size: int
) -> tuple[int, int]:
    """TP-local ``(conv_dim, v_heads)`` for the GDN state tensors -- the
    single source of the sharding math shared by the pool allocation and
    the byte-estimate helper below."""
    local_k_heads = div_even(num_key_heads, tp_size, allow_replicate=True)
    local_v_heads = div_even(num_value_heads, tp_size, allow_replicate=True)
    local_conv_dim = 2 * local_k_heads * key_head_dim + local_v_heads * value_head_dim
    return local_conv_dim, local_v_heads


class LinearStatePool:
    """Per-request recurrent state (conv + SSM) for GatedDeltaNet layers,
    ping-pong/COW-capable (issue #32's own tool-call-anchor snapshot needs
    :meth:`copy_from` to freeze a live slot into an idle track slot without
    disturbing the live one).

    Unlike this port's own existing (simpler) ``_LinearStatePool`` in
    ``qwen3_5_moe/__init__.py``, slots here are NOT tied 1:1 to
    ``Req.table_idx`` -- a real hybrid-radix deployment needs more slots
    than concurrently-running requests (each request's own working set is
    1 live + 2 ping-pong + 1 locked-committed-snapshot, per issue #32's own
    epic body), so this pool is its own free-list allocator (:meth:`alloc`/
    :meth:`free`), matching :class:`freetoken.kvcache.mha_pool.MHAKVCache`'s
    own pattern for the KV side.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_key_heads: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        conv_kernel_dim: int,
        num_slots: int,
        dtype: torch.dtype,
        device: torch.device,
        layer_ids: list[int] | None = None,
        tp_size: int = 1,
    ) -> None:
        self._device = device
        self._num_slots = num_slots

        local_conv_dim, local_v_heads = _linear_local_dims(
            num_key_heads, num_value_heads, key_head_dim, value_head_dim, tp_size
        )

        # conv left-context: the last (kernel-1) timesteps of the conv input stream.
        self.conv_states = torch.zeros(
            (num_layers, num_slots, local_conv_dim, conv_kernel_dim - 1),
            dtype=dtype,
            device=device,
        )
        # SSM recurrent state. fp32 by default (matches HF mamba_ssm_dtype);
        # the dtype is overridable via FREETOKEN_MAMBA_SSM_DTYPE (see
        # ssm_state_dtype).
        self.recurrent_states = torch.zeros(
            (num_layers, num_slots, local_v_heads, key_head_dim, value_head_dim),
            dtype=ssm_state_dtype(),
            device=device,
        )
        # Maps a global (possibly non-contiguous, interleaved with
        # full-attention layers) layer id to this pool's own local tensor
        # index. Defaults to 0..num_layers-1 (no interleaving) when the
        # caller doesn't have real global ids handy yet (e.g. a unit test).
        ids = layer_ids if layer_ids is not None else list(range(num_layers))
        if len(ids) != num_layers:
            raise ValueError(f"layer_ids has {len(ids)} entries, expected num_layers={num_layers}")
        self._local_index = {layer_id: i for i, layer_id in enumerate(ids)}

        # Free-list allocator over slots 1..num_slots-1 (slot 0 reserved as
        # a padding sink, matching MHAKVCache's own convention -- see its
        # docstring). Live working slots, ping-pong track slots, and
        # radix-tree-donated snapshots are all drawn from this single
        # free-list, so memory flows between them by demand.
        self.padding_slot = 0
        self._free_slots: list[int] = list(range(1, num_slots))

    @property
    def num_free_slots(self) -> int:
        return len(self._free_slots)

    def alloc(self, n: int = 1) -> list[int]:
        """Pop ``n`` free slot ids (LIFO). Raises if the pool is exhausted."""
        if n > len(self._free_slots):
            raise RuntimeError(f"LinearStatePool exhausted: need {n}, have {len(self._free_slots)}")
        return [self._free_slots.pop() for _ in range(n)]

    def reclaim_all_slots(self) -> None:
        """Restore the free-list to all non-padding slots. Idle-only: the
        caller (e.g. a CacheManager rebuild that discards the tree owning
        donated snapshots) must guarantee no running request holds a slot,
        otherwise live state would be handed out twice."""
        self._free_slots = list(range(1, self._num_slots))

    def rebuild(self, num_slots: int) -> None:
        """Reallocate the conv + recurrent state tensors for ``num_slots``
        slots IN PLACE.

        Geometry (layers, conv dim, head dims) and dtypes are taken from
        the existing tensors; only the slot count changes. Object identity
        is preserved so cached references (``ctx.linear_state_pool``) stay
        valid. Idle-only and destructive: every live/snapshot state is
        dropped, so the caller must guarantee no running request holds a
        slot and the radix tree owning donated snapshots is discarded too.
        """
        n_layers, _, local_conv_dim, km1 = self.conv_states.shape
        _, _, local_v_heads, key_head_dim, value_head_dim = self.recurrent_states.shape
        conv_dtype, rec_dtype = self.conv_states.dtype, self.recurrent_states.dtype
        device = self._device
        self.conv_states = torch.zeros(
            (n_layers, num_slots, local_conv_dim, km1), dtype=conv_dtype, device=device
        )
        self.recurrent_states = torch.zeros(
            (n_layers, num_slots, local_v_heads, key_head_dim, value_head_dim),
            dtype=rec_dtype,
            device=device,
        )
        self._num_slots = num_slots
        self._free_slots = list(range(1, num_slots))

    def free(self, slots) -> None:
        """Return slot ids to the free-list. Accepts an int, list, or 1-D tensor."""
        if isinstance(slots, torch.Tensor):
            slots = slots.flatten().tolist()
        elif isinstance(slots, int):
            slots = [slots]
        self._free_slots.extend(int(s) for s in slots)

    def clear_slots(self, slots) -> None:
        """Zero conv + recurrent state at ``slots`` across all linear layers (fresh sequence)."""
        if isinstance(slots, (list, tuple)):
            slots = torch.as_tensor(slots, dtype=torch.long, device=self._device)
        self.conv_states[:, slots] = 0
        self.recurrent_states[:, slots] = 0

    def copy_from(self, src: int, dst: int) -> None:
        """Copy a whole-sequence snapshot (conv + recurrent, all layers)
        from slot ``src`` to ``dst``. Used for COW-on-restore (donated
        snapshot -> fresh live slot) and for freezing a live slot into an
        idle ping-pong track slot (issue #32's own tool-call-anchor
        snapshot)."""
        self.conv_states[:, dst].copy_(self.conv_states[:, src])
        self.recurrent_states[:, dst].copy_(self.recurrent_states[:, src])

    def is_linear_layer(self, layer_id: int) -> bool:
        return layer_id in self._local_index

    def local_index(self, layer_id: int) -> int:
        return self._local_index[layer_id]

    def conv_state(self, layer_id: int, table_idx: int) -> torch.Tensor:
        return self.conv_states[self._local_index[layer_id], table_idx]

    def recurrent_state(self, layer_id: int, table_idx: int) -> torch.Tensor:
        return self.recurrent_states[self._local_index[layer_id], table_idx]

    def reset(self, table_idx: int) -> None:
        """Zero a slot across all linear layers (new request takes this table_idx)."""
        self.conv_states[:, table_idx].zero_()
        self.recurrent_states[:, table_idx].zero_()

    @property
    def num_linear_layers(self) -> int:
        return len(self._local_index)

    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def device(self) -> torch.device:
        return self._device

    def bytes_per_slot(self) -> int:
        """Total state bytes for one request (all linear layers)."""
        per = (
            self.conv_states[:, 0].numel() * self.conv_states.element_size()
            + self.recurrent_states[:, 0].numel() * self.recurrent_states.element_size()
        )
        return int(per)


def linear_state_bytes_per_req(
    *,
    num_layers: int,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    conv_kernel_dim: int,
    tp_size: int,
    dtype: torch.dtype,
) -> int:
    """Linear-state bytes for one request across all linear layers (TP-local)."""
    local_conv_dim, local_v_heads = _linear_local_dims(
        num_key_heads, num_value_heads, key_head_dim, value_head_dim, tp_size
    )
    conv_elems = local_conv_dim * (conv_kernel_dim - 1)
    rec_elems = local_v_heads * key_head_dim * value_head_dim
    conv_bytes = conv_elems * dtype.itemsize  # conv state in model dtype
    rec_bytes = rec_elems * ssm_state_dtype().itemsize  # recurrent state (default fp32)
    return int(num_layers * (conv_bytes + rec_bytes))


__all__ = ["LinearStatePool", "linear_state_bytes_per_req", "ssm_state_dtype"]
