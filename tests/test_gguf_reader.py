"""Tests for the GGUF binary format reader (issue `models-gguf-reader`, #270).

Two layers of validation, per the issue's own test strategy ("validate against
a small, real GGUF file ... don't fabricate a synthetic GGUF blob as the only
test"):

1. Real-file tests (marked ``slow``: they pull real fixtures from the Hugging
   Face Hub on first run, then read from the local HF cache) against
   ``ggml-org/models``' ``tinyllamas/stories260K.gguf`` (a tiny, unquantized
   F32 llama.cpp checkpoint used by llama.cpp's own CI) and
   ``tinyllamas/stories15M-q8_0.gguf`` (a Q8_0-quantized one, for a non-F32
   GGML quant type). Values are cross-checked two ways: by hand (offsets
   spot-checked by summing byte sizes across the fixture's real tensor-info
   table) and against the reference ``gguf`` pip package's own ``GGUFReader``
   (already vendored in this repo's ``.venv`` -- see ``gguf-dump`` etc. under
   ``.venv/bin/``), which is never used by ``freetoken.models.gguf.reader``
   itself (that stays a from-scratch, dependency-free ``struct`` parser per
   the issue's "pure Python struct/mmap only" constraint) but is a solid
   independent oracle for this test.
2. Synthetic-bytes tests for format-level edge cases (every KV value type
   including nested arrays/negative ints/bools, malformed-file rejection,
   directory-with-multiple-.gguf-files rejection, GGUF v1's narrower
   uint32 header counts) that the two real fixtures don't happen to exercise.
"""
from __future__ import annotations

import struct

import pytest

from freetoken.models.gguf import (
    GGUFFormatError,
    GGUFValueType,
    gguf_tensor_info,
    gguf_tensor_names,
    is_gguf_path,
    load_gguf,
    load_gguf_metadata,
)

# --------------------------------------------------------------------------
# Real-file fixtures (network on first run; cached by huggingface_hub after).
# --------------------------------------------------------------------------

_HF_REPO = "ggml-org/models"
_F32_FIXTURE = "tinyllamas/stories260K.gguf"  # tiny (~1.2MB), unquantized F32
_Q8_FIXTURE = "tinyllamas/stories15M-q8_0.gguf"  # ~26MB, Q8_0-quantized


def _hf_download(filename: str) -> str:
    huggingface_hub = pytest.importorskip("huggingface_hub")
    try:
        return huggingface_hub.hf_hub_download(_HF_REPO, filename)
    except Exception as exc:  # pragma: no cover - network/offline environment
        pytest.skip(f"could not download real GGUF fixture {filename} from the Hub: {exc}")


@pytest.fixture(scope="module")
def f32_gguf_path() -> str:
    return _hf_download(_F32_FIXTURE)


@pytest.fixture(scope="module")
def q8_gguf_path() -> str:
    return _hf_download(_Q8_FIXTURE)


@pytest.mark.slow
def test_is_gguf_path_true_for_real_file(f32_gguf_path):
    assert is_gguf_path(f32_gguf_path)


@pytest.mark.slow
def test_is_gguf_path_true_for_directory_containing_one_gguf_file(f32_gguf_path, tmp_path):
    import shutil

    shutil.copy(f32_gguf_path, tmp_path / "model.gguf")
    assert is_gguf_path(str(tmp_path))


def test_is_gguf_path_false_for_non_gguf_directory(tmp_path):
    # A plain safetensors-style checkpoint dir: no GGUF magic anywhere.
    (tmp_path / "config.json").write_text('{"architectures": ["FooForCausalLM"]}')
    (tmp_path / "model.safetensors").write_bytes(b"\x00" * 16)
    assert not is_gguf_path(str(tmp_path))


def test_is_gguf_path_false_for_missing_path(tmp_path):
    assert not is_gguf_path(str(tmp_path / "does_not_exist"))


def test_is_gguf_path_false_for_file_with_wrong_magic(tmp_path):
    bad = tmp_path / "not_really.gguf"
    bad.write_bytes(b"OOPS" + b"\x00" * 32)
    assert not is_gguf_path(str(bad))


@pytest.mark.slow
def test_load_gguf_metadata_matches_hand_verified_values(f32_gguf_path):
    # Hand-verified against the real file's raw bytes (parsed byte-for-byte
    # with an independent throwaway script before this reader existed):
    # magic=b"GGUF", version=3, tensor_count=48, kv_count=19, and these
    # exact KV values.
    meta = load_gguf_metadata(f32_gguf_path)
    assert meta["general.architecture"] == "llama"
    assert meta["general.name"] == "llama"
    assert meta["tokenizer.ggml.model"] == "llama"
    assert meta["llama.block_count"] == 5
    assert meta["llama.context_length"] == 2048
    assert meta["llama.embedding_length"] == 64
    assert meta["llama.feed_forward_length"] == 172
    assert meta["llama.attention.head_count"] == 8
    assert meta["llama.attention.head_count_kv"] == 4
    assert meta["llama.rope.dimension_count"] == 8
    assert meta["tokenizer.ggml.bos_token_id"] == 1
    assert meta["tokenizer.ggml.eos_token_id"] == 2
    # A STRING array: 512 vocab entries, first three hand-verified against
    # the raw bytes.
    tokens = meta["tokenizer.ggml.tokens"]
    assert isinstance(tokens, list)
    assert len(tokens) == 512
    assert tokens[:3] == ["<unk>", "<s>", "</s>"]
    # A FLOAT32 array of per-token scores, same length as the vocab.
    assert len(meta["tokenizer.ggml.scores"]) == 512
    assert meta["tokenizer.ggml.scores"][0] == pytest.approx(0.0)


@pytest.mark.slow
def test_gguf_tensor_info_matches_hand_verified_offsets(f32_gguf_path):
    # Hand-verified F32 offsets: element_count * 4 bytes/elem, cumulative,
    # starting at 0 (see this test module's docstring).
    info = gguf_tensor_info(f32_gguf_path)
    names = gguf_tensor_names(f32_gguf_path)
    assert names[:4] == [
        "token_embd.weight",
        "output_norm.weight",
        "output.weight",
        "blk.0.attn_q.weight",
    ]
    assert len(names) == 48

    embd = info["token_embd.weight"]
    assert embd.shape == (64, 512)
    assert embd.ggml_type == 0  # F32
    assert embd.ggml_type_name == "F32"
    assert embd.n_elements == 64 * 512

    norm = info["output_norm.weight"]
    assert norm.shape == (64,)
    # 64*512 elements * 4 bytes/elem = 131072 bytes after token_embd.weight.
    assert norm.offset == embd.offset + 64 * 512 * 4

    out = info["output.weight"]
    assert out.shape == (64, 512)
    assert out.offset == norm.offset + 64 * 4

    q0 = info["blk.0.attn_q.weight"]
    assert q0.shape == (64, 64)
    assert q0.offset == out.offset + 64 * 512 * 4


@pytest.mark.slow
def test_gguf_tensor_info_q8_quant_type_matches_hand_verified_values(q8_gguf_path):
    info = gguf_tensor_info(q8_gguf_path)
    embd = info["token_embd.weight"]
    assert embd.ggml_type == 8  # Q8_0
    assert embd.ggml_type_name == "Q8_0"
    assert embd.shape == (288, 32000)
    norm = info["output_norm.weight"]
    assert norm.ggml_type == 0  # F32 (norm weights are never quantized)
    # Q8_0 block layout: 32 elements/block, 34 bytes/block (2-byte f16 scale
    # + 32 int8 values). 288*32000 elements / 32 = 288000 blocks * 34 bytes.
    assert norm.offset == embd.offset + (288 * 32000 // 32) * 34


@pytest.mark.slow
def test_matches_reference_gguf_package(f32_gguf_path, q8_gguf_path):
    """Cross-check against the reference `gguf` pip package's own
    GGUFReader -- an independent oracle, never used by our own parser."""
    gguf_ref = pytest.importorskip("gguf")

    for path in (f32_gguf_path, q8_gguf_path):
        ours = load_gguf(path)
        theirs = gguf_ref.GGUFReader(path)

        assert len(ours.tensors) == len(theirs.tensors)
        assert ours.metadata["general.architecture"] == str(
            theirs.fields["general.architecture"].parts[-1].tobytes().decode("utf-8")
        )

        for ref_tensor in theirs.tensors:
            ours_info = ours.tensors[ref_tensor.name]
            assert ours_info.shape == tuple(int(d) for d in ref_tensor.shape)
            assert ours_info.ggml_type == int(ref_tensor.tensor_type)
            assert ours_info.n_elements == ref_tensor.n_elements


def test_import_is_torch_free():
    # The load-bearing assertion is that this import succeeds at all: this
    # test module (and freetoken.models.gguf.reader) must be importable in
    # the torch-free CPU-CLI-smoke venv. If torch were imported at module
    # scope, `import freetoken.models.gguf` would already have failed before
    # this test function even ran when torch is absent (see convert.py's own
    # docstring for the same constraint this mirrors).
    import freetoken.models.gguf.reader as reader_mod

    assert reader_mod is not None


# --------------------------------------------------------------------------
# Synthetic-bytes tests: format-level edge cases real fixtures don't exercise.
# --------------------------------------------------------------------------


def _gguf_string(s: str) -> bytes:
    data = s.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _build_minimal_gguf(*, version: int = 3, kv: list, tensors: list, alignment: int | None = None) -> bytes:
    """Hand-assemble a minimal GGUF file's bytes for edge-case coverage.

    ``kv`` is a list of ``(key, value_type, raw_value_bytes)``.
    ``tensors`` is a list of ``(name, shape, ggml_type, rel_offset)``.
    """
    count_fmt = "<I" if version == 1 else "<Q"
    out = bytearray()
    out += b"GGUF"
    out += struct.pack("<I", version)
    out += struct.pack(count_fmt, len(tensors))
    out += struct.pack(count_fmt, len(kv))
    for key, value_type, raw in kv:
        out += _gguf_string(key)
        out += struct.pack("<I", value_type)
        out += raw
    for name, shape, ggml_type, rel_offset in tensors:
        out += _gguf_string(name)
        out += struct.pack("<I", len(shape))
        for d in shape:
            out += struct.pack("<Q", d)
        out += struct.pack("<I", ggml_type)
        out += struct.pack("<Q", rel_offset)
    return bytes(out)


def test_synthetic_all_scalar_value_types_and_nested_array(tmp_path):
    kv = [
        ("a.uint8", GGUFValueType.UINT8, struct.pack("<B", 200)),
        ("a.int8", GGUFValueType.INT8, struct.pack("<b", -5)),
        ("a.uint16", GGUFValueType.UINT16, struct.pack("<H", 40000)),
        ("a.int16", GGUFValueType.INT16, struct.pack("<h", -1000)),
        ("a.uint32", GGUFValueType.UINT32, struct.pack("<I", 3_000_000_000)),
        ("a.int32", GGUFValueType.INT32, struct.pack("<i", -70000)),
        ("a.float32", GGUFValueType.FLOAT32, struct.pack("<f", 3.5)),
        ("a.bool_true", GGUFValueType.BOOL, struct.pack("<B", 1)),
        ("a.bool_false", GGUFValueType.BOOL, struct.pack("<B", 0)),
        ("a.string", GGUFValueType.STRING, _gguf_string("hello gguf")),
        ("a.uint64", GGUFValueType.UINT64, struct.pack("<Q", 2**63 + 5)),
        ("a.int64", GGUFValueType.INT64, struct.pack("<q", -(2**40))),
        ("a.float64", GGUFValueType.FLOAT64, struct.pack("<d", 2.718281828)),
        (
            "a.int_array",
            GGUFValueType.ARRAY,
            struct.pack("<IQ", GGUFValueType.INT32, 3) + struct.pack("<3i", -1, 0, 1),
        ),
        (
            "a.string_array",
            GGUFValueType.ARRAY,
            struct.pack("<IQ", GGUFValueType.STRING, 2) + _gguf_string("x") + _gguf_string("yy"),
        ),
        ("a.empty_array", GGUFValueType.ARRAY, struct.pack("<IQ", GGUFValueType.INT8, 0)),
    ]
    data = _build_minimal_gguf(kv=kv, tensors=[])
    path = tmp_path / "synthetic.gguf"
    path.write_bytes(data)

    assert is_gguf_path(str(path))
    meta = load_gguf_metadata(str(path))
    assert meta["a.uint8"] == 200
    assert meta["a.int8"] == -5
    assert meta["a.uint16"] == 40000
    assert meta["a.int16"] == -1000
    assert meta["a.uint32"] == 3_000_000_000
    assert meta["a.int32"] == -70000
    assert meta["a.float32"] == pytest.approx(3.5)
    assert meta["a.bool_true"] is True
    assert meta["a.bool_false"] is False
    assert meta["a.string"] == "hello gguf"
    assert meta["a.uint64"] == 2**63 + 5
    assert meta["a.int64"] == -(2**40)
    assert meta["a.float64"] == pytest.approx(2.718281828)
    assert meta["a.int_array"] == [-1, 0, 1]
    assert meta["a.string_array"] == ["x", "yy"]
    assert meta["a.empty_array"] == []


def test_synthetic_tensor_info_and_alignment_padding(tmp_path):
    tensors = [
        ("t0", (2, 3), 0, 0),
        ("t1", (4,), 1, 24),
    ]
    data = _build_minimal_gguf(kv=[], tensors=tensors)
    path = tmp_path / "synthetic_tensors.gguf"
    path.write_bytes(data)

    parsed = load_gguf(str(path))
    assert parsed.tensor_names == ["t0", "t1"]
    assert parsed.tensors["t0"].shape == (2, 3)
    assert parsed.tensors["t0"].ggml_type_name == "F32"
    assert parsed.tensors["t1"].ggml_type_name == "F16"
    # Default alignment (32): tensor_data_start rounds header_end up to the
    # next 32-byte boundary, and each tensor's absolute offset is
    # tensor_data_start + its on-disk relative offset.
    assert parsed.tensor_data_start % 32 == 0
    assert parsed.tensor_data_start >= len(data)
    assert parsed.tensors["t0"].offset == parsed.tensor_data_start + 0
    assert parsed.tensors["t1"].offset == parsed.tensor_data_start + 24


def test_synthetic_general_alignment_override(tmp_path):
    kv = [("general.alignment", GGUFValueType.UINT32, struct.pack("<I", 16))]
    tensors = [("t0", (1,), 0, 0)]
    data = _build_minimal_gguf(kv=kv, tensors=tensors)
    path = tmp_path / "synthetic_align16.gguf"
    path.write_bytes(data)

    parsed = load_gguf(str(path))
    assert parsed.tensor_data_start % 16 == 0
    assert parsed.metadata["general.alignment"] == 16


def test_synthetic_version1_uses_uint32_header_counts(tmp_path):
    data = _build_minimal_gguf(version=1, kv=[], tensors=[("only", (5,), 0, 0)])
    path = tmp_path / "synthetic_v1.gguf"
    path.write_bytes(data)

    parsed = load_gguf(str(path))
    assert parsed.version == 1
    assert parsed.tensor_names == ["only"]


def test_load_gguf_raises_on_bad_magic(tmp_path):
    path = tmp_path / "bad_magic.gguf"
    path.write_bytes(b"NOPE" + b"\x00" * 32)
    with pytest.raises(GGUFFormatError):
        load_gguf(str(path))


def test_load_gguf_raises_on_truncated_file(tmp_path):
    # Valid header claiming a KV entry that is never actually present.
    data = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, 1)
    path = tmp_path / "truncated.gguf"
    path.write_bytes(data)
    with pytest.raises(GGUFFormatError):
        load_gguf(str(path))


def test_is_gguf_path_false_for_directory_with_multiple_gguf_files(tmp_path):
    (tmp_path / "a.gguf").write_bytes(b"GGUF" + b"\x00" * 32)
    (tmp_path / "b.gguf").write_bytes(b"GGUF" + b"\x00" * 32)
    assert not is_gguf_path(str(tmp_path))
