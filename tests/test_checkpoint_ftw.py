"""Correctness tests for the FTW checkpoint format (issue `ftw-checkpoint`, #11).

CPU-only: FTW is a storage-format round trip, no XPU dependency.
"""
from __future__ import annotations

import json
import os

import pytest

torch = pytest.importorskip("torch")

from freetoken.checkpoint.convert import convert_checkpoint, convert_summary
from freetoken.checkpoint.ftw import FtwArchive, is_ftw_dir
from freetoken.models.weight import iter_safetensors


def test_write_read_round_trip_various_dtypes(tmp_path):
    torch.manual_seed(0)
    tensors = {
        "a.weight": torch.randn(4, 8, dtype=torch.float32),
        "b.weight": torch.randn(3, 5, dtype=torch.bfloat16),
        "c.weight": torch.randint(0, 100, (7,), dtype=torch.int64),
        "d.weight": (torch.randn(2, 2) * 0.1).to(torch.float8_e4m3fn),
    }
    archive_dir = str(tmp_path / "archive")
    FtwArchive(archive_dir).write(tensors)

    assert is_ftw_dir(archive_dir)
    back = dict(FtwArchive(archive_dir).read())
    assert set(back) == set(tensors)
    for name, original in tensors.items():
        got = back[name]
        assert got.shape == original.shape
        assert got.dtype == original.dtype
        if original.dtype == torch.float8_e4m3fn:
            # fp8 has no torch.equal-friendly comparator via allclose; compare as fp32.
            torch.testing.assert_close(got.to(torch.float32), original.to(torch.float32))
        else:
            assert torch.equal(got, original)


def test_read_is_zero_copy_on_cpu(tmp_path):
    """Regression guard for the real perf bug this format's read() path had:
    an earlier per-tensor mmap-slice-plus-.clone() design (and, briefly, a
    bulk-readinto-into-one-fresh-buffer design) made FTW measurably SLOWER
    than raw safetensors for many-small-tensor checkpoints -- a direct
    contradiction of this issue's own "load banks faster than raw
    safetensors" accept criterion (see benchmarks/bench_load_weight_generic.py,
    which measures the real speedup; ~2.4x on a 128/512-expert synthetic
    checkpoint after the fix). A CPU-device read must be a plain VIEW into
    the shared mmap'd buffer, not a copy -- checked here via torch's own
    ``_base`` chain rather than timing (flaky in CI), since a copy would
    have ``_base is None``."""
    tensors = {"w": torch.randn(4, 4)}
    archive_dir = str(tmp_path / "archive")
    FtwArchive(archive_dir).write(tensors)

    (name, tensor), = FtwArchive(archive_dir).read()
    assert tensor._base is not None, "read() copied instead of viewing -- the real perf regression this guards"


def test_write_pads_tensor_offsets_to_alignment_boundary(tmp_path):
    """Regression guard: read() reinterprets every tensor as a view.T.view(dtype)
    into one shared buffer, and torch.Tensor.view(dtype) requires the view's
    byte offset to be a multiple of the target dtype's element size (e.g. 8
    for int64/float64) -- an unaligned offset raised RuntimeError before
    write() started padding each tensor's start offset. This fixture's odd
    tensor sizes (an int8 of length 3, i.e. 3 bytes) deliberately produce a
    misaligned NEXT offset if write() does not pad."""
    tensors = {
        "a": torch.arange(3, dtype=torch.int8),  # 3 bytes -> misaligned next offset if unpadded
        "b": torch.arange(4, dtype=torch.int64),  # needs 8-byte-aligned offset to view() correctly
    }
    archive_dir = str(tmp_path / "archive")
    FtwArchive(archive_dir).write(tensors)

    back = dict(FtwArchive(archive_dir).read())
    for name, original in tensors.items():
        torch.testing.assert_close(back[name], original)


def test_read_tensors_survive_after_generator_exhausted(tmp_path):
    """Guards the mmap-lifetime bug class: a caller collecting every yielded
    tensor into a dict (e.g. load_weight building a state dict) must still
    be able to use them after the read() generator itself has returned --
    the mmap backing the raw bytes is closed at that point."""
    tensors = {"w": torch.arange(16, dtype=torch.float32).reshape(4, 4)}
    archive_dir = str(tmp_path / "archive")
    FtwArchive(archive_dir).write(tensors)

    collected = dict(FtwArchive(archive_dir).read())  # generator fully drained here
    torch.testing.assert_close(collected["w"], tensors["w"])
    # Touch it again well after the read: would segfault/corrupt if the
    # tensor were still a view into the (by now closed) mmap.
    assert collected["w"].sum().item() == tensors["w"].sum().item()


def test_convert_checkpoint_round_trips_and_copies_config(tmp_path):
    from safetensors.torch import save_file

    src = tmp_path / "src_ckpt"
    src.mkdir()
    torch.manual_seed(1)
    tensors = {
        "model.embed_tokens.weight": torch.randn(16, 8),
        "model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(12, 8),
        "model.layers.0.mlp.experts.1.gate_proj.weight": torch.randn(12, 8),
    }
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(src / "model.safetensors"))
    config = {"architectures": ["Qwen3MoeForCausalLM"], "hidden_size": 8}
    (src / "config.json").write_text(json.dumps(config))

    dst = tmp_path / "ftw_ckpt"
    convert_checkpoint(str(src), str(dst))

    assert is_ftw_dir(str(dst))
    assert (dst / "config.json").is_file()
    assert json.loads((dst / "config.json").read_text()) == config

    summary = convert_summary(str(dst))
    assert summary["tensors"] == len(tensors)

    back = dict(iter_safetensors(str(dst)))
    assert set(back) == set(tensors)
    for name, original in tensors.items():
        torch.testing.assert_close(back[name], original)


def test_ftw_archive_write_accepts_a_streaming_iterator(tmp_path):
    """convert_checkpoint streams tensor-by-tensor into write() rather than
    materializing a {name: tensor} dict of the whole checkpoint first (a real
    multi-expert checkpoint can be large enough for that dict alone to
    exhaust host RAM -- PR-Agent review, PR #126). This proves write() itself
    accepts a plain generator, not just a dict."""
    torch.manual_seed(4)
    source = {"x": torch.randn(2, 2), "y": torch.randn(3, 3)}

    def _gen():
        yield from source.items()

    archive_dir = str(tmp_path / "streamed")
    FtwArchive(archive_dir).write(_gen())

    back = dict(FtwArchive(archive_dir).read())
    assert set(back) == set(source)
    for name, original in source.items():
        torch.testing.assert_close(back[name], original)


def test_convert_checkpoint_rejects_empty_source(tmp_path):
    src = tmp_path / "empty_ckpt"
    src.mkdir()
    (src / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="no tensors"):
        convert_checkpoint(str(src), str(tmp_path / "out"))


def test_iter_safetensors_auto_detects_ftw_dir(tmp_path):
    """The generic reader every model's iter_weights calls picks up FTW
    without any per-model change -- the mechanism behind the "ft serve
    --model auto-detects an FTW dir" accept criterion."""
    tensors = {"w": torch.randn(3, 3)}
    ftw_dir = str(tmp_path / "ftw_only")
    FtwArchive(ftw_dir).write(tensors)
    assert not os.path.isfile(os.path.join(ftw_dir, "model.safetensors"))

    back = dict(iter_safetensors(ftw_dir))
    torch.testing.assert_close(back["w"], tensors["w"])
