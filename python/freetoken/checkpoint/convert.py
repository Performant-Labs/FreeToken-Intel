"""HF safetensors -> FTW converter.

Upstream NVIDIA path: python/freetoken/checkpoint/convert.py
Issue: `ftw-checkpoint` (#11, see docs/architecture.md).

Model-agnostic (matching upstream's own framing): reads a checkpoint's raw
tensors via the existing generic :func:`freetoken.models.weight.iter_safetensors`
(the same reader every model's ``iter_weights`` already uses) and repacks
them 1:1 into an FTW archive -- no per-model conversion code, no fusion, no
device placement. Scoped down from upstream's converter, which additionally
drives the per-model loaders to bake in TP-sharding and backend expert
repacking; that is real, separate work (out of scope here, same as
``models/weight.py``'s own "FTW / GGUF branches ... out of scope" note,
which this issue starts to close).
"""
from __future__ import annotations

import os
import shutil

from .ftw import FtwArchive

# torch is imported lazily inside convert_checkpoint(), not at module level:
# this module is on the `ft checkpoint` CLI's import path
# (checkpoint/__main__.py -> convert.py), which must stay importable
# (`ft checkpoint --help`) in a torch-free environment -- the CPU CLI-smoke
# lane runs exactly that (caught in PR #127).


class _CountingIterator:
    """Wraps an iterator, counting items as they pass through, so the caller
    can tell (after a single-pass consumer like FtwArchive.write() has
    drained it) whether it yielded anything -- without buffering it."""

    def __init__(self, source) -> None:
        self._source = iter(source)
        self.count = 0

    def __iter__(self):
        for item in self._source:
            self.count += 1
            yield item


# Non-tensor checkpoint files worth carrying over so the FTW dir is a
# self-contained drop-in --model path, same as the source HF dir.
_COPY_GLOBS = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)


def convert_checkpoint(model_path: str, output_path: str) -> str:
    """Convert the safetensors checkpoint at ``model_path`` into an FTW
    archive at ``output_path``. Returns ``output_path``.

    Every tensor is carried over unchanged (dtype, shape, values) -- this
    repacks the storage format, it does not requantize or reshape anything.
    """
    import torch

    from freetoken.models.weight import iter_safetensors

    os.makedirs(output_path, exist_ok=True)
    # Stream tensor-by-tensor straight from the reader into the archive writer
    # rather than materializing a {name: tensor} dict of the whole checkpoint
    # first -- a real multi-expert checkpoint can be large enough for that
    # intermediate dict alone to exhaust host RAM (PR-Agent review, PR #126).
    written = _CountingIterator(iter_safetensors(model_path, torch.device("cpu")))
    FtwArchive(output_path).write(written)
    if written.count == 0:
        # Nothing to convert -- remove the (empty) archive files write() just
        # created rather than leaving a broken FTW dir behind.
        for fname in ("ftw_index.json", "ftw_weights.bin"):
            fpath = os.path.join(output_path, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)
        raise ValueError(f"no tensors found in checkpoint at {model_path!r}")

    for name in _COPY_GLOBS:
        src = os.path.join(model_path, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_path, name))
    # A source checkpoint's shard index is meaningless once repacked into a
    # single FTW blob -- do not carry it over (its presence would make the
    # FTW dir look like an unindexed safetensors checkpoint to anything that
    # scans for one instead of checking ftw_index.json first).
    return output_path


def convert_summary(output_path: str) -> dict:
    """Small JSON-able summary of a converted archive, for CLI/log output."""
    index = FtwArchive(output_path).read_index()
    total_bytes = sum(meta["nbytes"] for meta in index.values())
    return {"tensors": len(index), "total_bytes": total_bytes, "path": output_path}
