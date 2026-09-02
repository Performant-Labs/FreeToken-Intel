"""FTW (FreeToken Weight) archive: a flat, mmap-friendly checkpoint format.

Upstream NVIDIA path: python/freetoken/checkpoint/ftw.py
Issue: `ftw-checkpoint` (#11, see docs/architecture.md).

Upstream's FTW is Xe2-tiled expert packing driven through the backend's
own parallel-O_DIRECT loader and post-fusion/TP-shard dense weights --
tied to CUDA host-pinned banks this port does not have (see
``models/weight.py``'s ``_PlainBank`` note: no CUDA host-alloc on the
Intel build). That whole machinery is out of scope here.

This is the part of FTW that is actually worth having on its own merits
and is fully portable: **one flat binary blob + one JSON index**, instead
of safetensors' one-file(-or-shard)-per-checkpoint-with-a-header-per-tensor
layout. A checkpoint with many small tensors (an MoE's per-expert weights
are exactly this shape) pays safetensors' per-shard header-parse /
``safe_open`` overhead on every load; FTW pays it once (one JSON index
read) and then does zero-copy ``mmap`` slices for every tensor. That is
where the "load banks faster than raw safetensors" accept criterion comes
from -- see ``benchmarks/bench_load_weight_generic.py``.

Directory layout (self-contained -- ``config.json`` / tokenizer files are
copied alongside by :func:`freetoken.checkpoint.convert.convert_checkpoint`,
so the FTW dir is a drop-in ``--model`` path, same as an HF checkpoint dir):

    <dir>/ftw_index.json    {"tensors": {name: {dtype, shape, offset, nbytes}}}
    <dir>/ftw_weights.bin   flat concatenation of every tensor's raw bytes,
                            in index order, no padding/alignment
"""
from __future__ import annotations

import json
import mmap
import os
import warnings
from typing import Dict, Iterator, Tuple

import torch

INDEX_NAME = "ftw_index.json"
WEIGHTS_NAME = "ftw_weights.bin"


def is_ftw_dir(path: str) -> bool:
    """True if ``path`` is a directory holding an FTW archive."""
    return os.path.isfile(os.path.join(path, INDEX_NAME))


def _dtype_to_str(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_from_str(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown FTW tensor dtype {name!r}")
    return dtype


class FtwArchive:
    """Reader/writer for one FTW directory."""

    def __init__(self, path: str) -> None:
        self.path = path

    def write(self, tensors: Dict[str, torch.Tensor] | Iterator[Tuple[str, torch.Tensor]]) -> None:
        """Write ``tensors`` (a ``{name: tensor}`` dict, or a ``(name, tensor)``
        iterator/generator) to this archive's directory, creating it if needed.

        Accepting an iterator lets a caller (:func:`freetoken.checkpoint.
        convert.convert_checkpoint`) stream straight from a checkpoint reader
        without first materializing every tensor into one Python dict --
        PR-Agent review on #126 flagged that a real multi-expert checkpoint
        can be large enough for that intermediate dict alone to exhaust host
        RAM during offline conversion.
        """
        os.makedirs(self.path, exist_ok=True)
        items = tensors.items() if isinstance(tensors, dict) else tensors
        index: Dict[str, dict] = {}
        offset = 0
        with open(os.path.join(self.path, WEIGHTS_NAME), "wb") as f:
            for name, tensor in items:
                # Reinterpret the tensor's own bytes as a flat uint8 view (works for
                # every torch dtype, including bf16/fp8, which lack numpy natives) --
                # no cast, no precision loss, just the raw storage.
                raw = tensor.contiguous().cpu().view(torch.uint8).numpy().tobytes()
                f.write(raw)
                index[name] = {
                    "dtype": _dtype_to_str(tensor.dtype),
                    "shape": list(tensor.shape),
                    "offset": offset,
                    "nbytes": len(raw),
                }
                offset += len(raw)
        with open(os.path.join(self.path, INDEX_NAME), "w", encoding="utf-8") as f:
            json.dump({"tensors": index}, f)

    def read_index(self) -> Dict[str, dict]:
        with open(os.path.join(self.path, INDEX_NAME), encoding="utf-8") as f:
            return json.load(f)["tensors"]

    def read(self, device: torch.device | str = "cpu") -> Iterator[Tuple[str, torch.Tensor]]:
        """Yield ``(name, tensor)`` for every tensor in this archive, on ``device``.

        The blob is mmap'd once and each tensor is sliced straight out of the
        page cache (one big sequential ``mmap`` instead of a safetensors
        ``safe_open`` header-parse per shard) -- that single mmap is what
        actually makes this fast. Each yielded tensor is a ``.clone()`` of
        its slice, not a view: the mmap is closed once this generator is
        exhausted (or garbage-collected), and a caller routinely holds
        tensors well past that point (e.g. building the model's state dict),
        so a raw view would go stale.
        """
        index = self.read_index()
        weights_path = os.path.join(self.path, WEIGHTS_NAME)
        with open(weights_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for name, meta in index.items():
                dtype = _dtype_from_str(meta["dtype"])
                offset, nbytes, shape = meta["offset"], meta["nbytes"], meta["shape"]
                # frombuffer() over a read-only mmap warns that the buffer is
                # non-writable -- expected and harmless here (immediately
                # .clone()'d below, never written through).
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="The given buffer is not writable")
                    flat = torch.frombuffer(mm, dtype=torch.uint8, count=nbytes, offset=offset)
                tensor = flat.view(dtype).view(shape).clone()
                yield name, tensor.to(device)
        finally:
            mm.close()
