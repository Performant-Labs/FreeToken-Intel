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
read), then one bulk sequential read of the whole blob into a single
torch-owned buffer, and slices every tensor out of that buffer as a plain
view (no per-tensor copy). That is where the "load banks faster than raw
safetensors" accept criterion comes from -- see
``benchmarks/bench_load_weight_generic.py``, and :meth:`FtwArchive.read`'s
own docstring for why an earlier per-tensor-``mmap``-slice-plus-``clone()``
design measured SLOWER than safetensors instead.

Directory layout (self-contained -- ``config.json`` / tokenizer files are
copied alongside by :func:`freetoken.checkpoint.convert.convert_checkpoint`,
so the FTW dir is a drop-in ``--model`` path, same as an HF checkpoint dir):

    <dir>/ftw_index.json    {"tensors": {name: {dtype, shape, offset, nbytes}}}
    <dir>/ftw_weights.bin   flat concatenation of every tensor's raw bytes,
                            in index order, each start offset padded up to
                            an 8-byte boundary (see FtwArchive.write's own
                            docstring for why)
"""
from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Dict, Iterator, Tuple

if TYPE_CHECKING:
    import torch

# torch is imported lazily inside each function/method that needs it, not at
# module level: this module is on the `ft checkpoint` CLI's import path
# (checkpoint/__main__.py -> convert.py -> ftw.py), which must stay
# importable (e.g. `ft checkpoint --help`) in a torch-free environment --
# the CPU CLI-smoke lane runs exactly that (caught in PR #127).

INDEX_NAME = "ftw_index.json"
WEIGHTS_NAME = "ftw_weights.bin"
# Every tensor's start offset is padded up to this boundary (see write()'s
# own docstring for why) -- 8 bytes covers every dtype this module supports
# (the widest is 8 bytes: int64/float64), so a single alignment value
# suffices without a per-dtype table.
_ALIGNMENT = 8


def is_ftw_dir(path: str) -> bool:
    """True if ``path`` is a directory holding an FTW archive."""
    return os.path.isfile(os.path.join(path, INDEX_NAME))


def _dtype_to_str(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_from_str(name: str) -> torch.dtype:
    import torch

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

        Each tensor's start offset is padded up to an ``_ALIGNMENT``-byte
        boundary (up to 7 bytes of zero padding, negligible next to a real
        tensor's size): :meth:`read` reinterprets every tensor as a plain
        VIEW into one big shared buffer, and ``torch.Tensor.view(dtype)``
        requires the view's byte offset to be a multiple of the target
        dtype's own element size (e.g. 8 for int64/float64) -- an unaligned
        offset raises ``RuntimeError`` at read time. A real checkpoint's
        widest common dtype is 8 bytes, so aligning to 8 covers every
        dtype this module supports without needing a per-dtype alignment
        table.
        """
        import torch

        os.makedirs(self.path, exist_ok=True)
        items = tensors.items() if isinstance(tensors, dict) else tensors
        index: Dict[str, dict] = {}
        offset = 0
        with open(os.path.join(self.path, WEIGHTS_NAME), "wb") as f:
            for name, tensor in items:
                pad = (-offset) % _ALIGNMENT
                if pad:
                    f.write(b"\x00" * pad)
                    offset += pad
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

        The whole flat blob is ``mmap``'d ONCE and wrapped in a single
        ``torch.frombuffer`` call; every tensor is then yielded as a plain
        VIEW into that one shared buffer -- no per-tensor copy at all on
        the CPU-device path (a real, measured performance requirement: see
        below). This relies on two things, both verified directly (not
        assumed) before landing this design:

        1. ``torch.frombuffer(mm, ...)`` keeps ``mm`` (and the file object
           used to create it) alive via its own internal reference, for as
           long as the returned tensor -- or ANY view sliced from it -- is
           still referenced, even after the local ``mm``/file variables in
           this method go out of scope, and even after the backing file on
           disk is deleted (POSIX unlink-after-mmap semantics; this is the
           SAME mechanism ``safetensors``' own ``safe_open`` relies on).
        2. Each further per-tensor slice (``whole[offset:offset+nbytes]``)
           chains through ``torch.Tensor._base``, which keeps ITS base (the
           whole-buffer tensor, and transitively ``mm``) alive too -- so a
           caller that drops every reference to this ``FtwArchive`` and
           this generator right after collecting the yielded tensors (the
           realistic usage pattern; see ``test_read_tensors_survive_after_
           generator_exhausted``) still holds valid tensors afterward.

        Two earlier designs were tried and measured wrong before this one,
        both against a 128-expert synthetic checkpoint via
        ``benchmarks/bench_load_weight_generic.py`` (not assumed correct):
        a per-tensor ``mmap`` slice + ``.clone()`` (the clone -- needed
        because each per-tensor mmap object was closed once the read()
        generator exhausted -- was ~half of total read time, and made FTW
        measurably SLOWER than raw safetensors for many-small-tensor
        checkpoints, a direct contradiction of this issue's own "load
        banks faster than raw safetensors" accept criterion); and a single
        bulk ``readinto`` of the whole file into one freshly-allocated
        buffer (correct and memory-safe, but an eager full-file copy is
        real, unavoidable work safetensors' own zero-copy ``get_tensor()``
        never pays -- 10x+ slower than this mmap-based design in the same
        benchmark). ``view(dtype)`` also requires each tensor's byte
        offset to be a multiple of that dtype's element size (e.g. 8 for
        int64/float64) -- :meth:`write`'s own per-tensor alignment padding
        exists specifically so this never raises.
        """
        import mmap
        import warnings

        import torch

        index = self.read_index()
        weights_path = os.path.join(self.path, WEIGHTS_NAME)
        with open(weights_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        # frombuffer() over a read-only mmap warns that the buffer is
        # non-writable -- expected and harmless (every tensor sliced from
        # `whole` below is only ever read, never written through). Applied
        # once for this single whole-buffer call, not per tensor.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given buffer is not writable")
            whole = torch.frombuffer(mm, dtype=torch.uint8)
        for name, meta in index.items():
            dtype = _dtype_from_str(meta["dtype"])
            offset, nbytes, shape = meta["offset"], meta["nbytes"], meta["shape"]
            tensor = whole[offset : offset + nbytes].view(dtype).view(shape)
            yield name, tensor.to(device)
