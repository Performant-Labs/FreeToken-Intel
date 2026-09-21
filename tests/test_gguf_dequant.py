"""Tests for GGML K-quant dequantization (issue `models-gguf-dequant`, #271).

Per the issue's own test strategy ("cross-check against llama.cpp's own
dequantized output ... not just does it run, but does it produce the same
floats -- this is the one place in the whole epic where getting the bit
layout subtly wrong produces plausible-looking-but-wrong numbers"), every
quant type is cross-checked two ways:

1. Real-file tests (marked ``slow``: pull real fixtures from the Hugging
   Face Hub on first run, cached after) against a real, tiny GGUF checkpoint
   quantized with ``Q4_K``/``Q6_K`` -- ``mradermacher/TinyStories-LLaMA2-20M-
   256h-4l-GQA-GGUF``'s ``Q4_K_M`` and ``Q6_K`` files (a ~14-17MB toy llama
   checkpoint with a 256-wide embedding dim, a multiple of the K-quant
   super-block size ``QK_K=256`` -- llama.cpp's own official tiny CI fixture
   repo, ``ggml-org/models``, has no K-quant fixtures at all: its "stories"
   toy checkpoints use a 64-wide embedding dim, too narrow for any K-quant,
   which is why llama.cpp's own quantizer falls back to ``Q8_0``/``Q5_0`` for
   them -- confirmed by inspecting their real tensor-info quant types with
   this repo's own GGUF reader before picking a different fixture).
2. Cross-checked against the reference ``gguf`` pip package's own
   ``gguf.quants.dequantize`` (already a real runtime dependency of this
   port, see ``pyproject.toml`` -- used here only as an independent oracle,
   never as this module's own implementation) on the exact same raw tensor
   bytes, read via this repo's own GGUF reader.
3. Synthetic-bytes tests (hand-assembled block bytes, values computed by
   hand) for ``Q4_0``/``Q8_0`` and small edge cases the real fixture doesn't
   happen to exercise (e.g. a single all-zero block, unsupported types).
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from freetoken.models.gguf import gguf_tensor_info
from freetoken.models.gguf.dequant import DequantError, dequantize

# --------------------------------------------------------------------------
# Real-file fixtures (network on first run; cached by huggingface_hub after).
# --------------------------------------------------------------------------

_HF_REPO = "mradermacher/TinyStories-LLaMA2-20M-256h-4l-GQA-GGUF"
_Q4_K_M_FIXTURE = "TinyStories-LLaMA2-20M-256h-4l-GQA.Q4_K_M.gguf"
_Q6_K_FIXTURE = "TinyStories-LLaMA2-20M-256h-4l-GQA.Q6_K.gguf"


def _hf_download(filename: str) -> str:
    huggingface_hub = pytest.importorskip("huggingface_hub")
    try:
        return huggingface_hub.hf_hub_download(_HF_REPO, filename)
    except Exception as exc:  # pragma: no cover - network/offline environment
        pytest.skip(f"could not download real GGUF fixture {filename} from the Hub: {exc}")


@pytest.fixture(scope="module")
def q4_k_m_gguf_path() -> str:
    return _hf_download(_Q4_K_M_FIXTURE)


@pytest.fixture(scope="module")
def q6_k_gguf_path() -> str:
    return _hf_download(_Q6_K_FIXTURE)


def _read_raw_tensor_bytes(gguf_path: str, name: str, info) -> bytes:
    tensor_info = info[name]
    with open(gguf_path, "rb") as f:
        f.seek(tensor_info.offset)
        # Read exactly this tensor's byte span: derive it from the type's
        # element/byte block ratio so we don't read into the next tensor.
        gguf_ref = pytest.importorskip("gguf")
        block_size, type_size = gguf_ref.GGML_QUANT_SIZES[
            gguf_ref.GGMLQuantizationType(tensor_info.ggml_type)
        ]
        n_bytes = (tensor_info.n_elements // block_size) * type_size
        return f.read(n_bytes)


def _assert_matches_reference_gguf(gguf_path: str, tensor_name: str, rtol: float, atol: float):
    gguf_ref = pytest.importorskip("gguf")

    info = gguf_tensor_info(gguf_path)
    tensor_info = info[tensor_name]
    raw = _read_raw_tensor_bytes(gguf_path, tensor_name, info)

    ours = dequantize(tensor_info.ggml_type, raw, tensor_info.shape)
    assert ours.dtype == torch.bfloat16
    assert tuple(ours.shape) == tuple(tensor_info.shape)

    qtype = gguf_ref.GGMLQuantizationType(tensor_info.ggml_type)
    block_size, type_size = gguf_ref.GGML_QUANT_SIZES[qtype]
    n_blocks = len(raw) // type_size
    raw_arr = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, type_size)
    theirs = gguf_ref.dequantize(raw_arr, qtype).reshape(tensor_info.shape)

    ours_f32 = ours.to(torch.float32).numpy()
    np.testing.assert_allclose(ours_f32, theirs, rtol=rtol, atol=atol)


@pytest.mark.slow
def test_q4_k_matches_reference_gguf_package(q4_k_m_gguf_path):
    info = gguf_tensor_info(q4_k_m_gguf_path)
    q4_k_names = [n for n, t in info.items() if t.ggml_type_name == "Q4_K"]
    assert q4_k_names, "fixture has no Q4_K tensors -- picked the wrong fixture"
    # bf16 rounds a float32 mantissa to 7 bits, so allow bf16-scale tolerance
    # (this port's own convention: dequantize to bf16, per #138's dtype
    # discipline, not exact f32) -- ~1e-2 relative is generous for bf16.
    for name in q4_k_names[:3]:
        _assert_matches_reference_gguf(q4_k_m_gguf_path, name, rtol=1e-2, atol=1e-3)


@pytest.mark.slow
def test_q6_k_matches_reference_gguf_package(q6_k_gguf_path):
    info = gguf_tensor_info(q6_k_gguf_path)
    q6_k_names = [n for n, t in info.items() if t.ggml_type_name == "Q6_K"]
    assert q6_k_names, "fixture has no Q6_K tensors -- picked the wrong fixture"
    for name in q6_k_names[:3]:
        _assert_matches_reference_gguf(q6_k_gguf_path, name, rtol=1e-2, atol=1e-3)


@pytest.mark.slow
def test_q4_k_m_output_head_is_q6_k(q4_k_m_gguf_path):
    # llama.cpp's own `q4_k_m` naming convention: dominant quant is Q4_K, but
    # a few sensitive tensors (notably the output head) get Q6_K instead --
    # this is the exact real-checkpoint pattern #271/#199 target (the real
    # Qwen3.6-35B-A3B q4_k_m quant mixes the same two types).
    info = gguf_tensor_info(q4_k_m_gguf_path)
    assert info["output.weight"].ggml_type_name == "Q6_K"
    assert info["token_embd.weight"].ggml_type_name == "Q4_K"


@pytest.mark.slow
def test_f32_norm_tensor_passthrough(q4_k_m_gguf_path):
    # Norm weights are never quantized (F32) even in a K-quant checkpoint --
    # dequantize() must handle the plain F32 passthrough case too.
    info = gguf_tensor_info(q4_k_m_gguf_path)
    name = "blk.0.attn_norm.weight"
    tensor_info = info[name]
    assert tensor_info.ggml_type_name == "F32"
    with open(q4_k_m_gguf_path, "rb") as f:
        f.seek(tensor_info.offset)
        raw = f.read(tensor_info.n_elements * 4)

    ours = dequantize(tensor_info.ggml_type, raw, tensor_info.shape)
    expected = np.frombuffer(raw, dtype="<f4")
    np.testing.assert_allclose(ours.to(torch.float32).numpy(), expected, rtol=1e-2, atol=1e-3)


# --------------------------------------------------------------------------
# Synthetic-bytes tests: hand-computed blocks for formats/edge cases the
# real fixture doesn't happen to exercise directly.
# --------------------------------------------------------------------------


def _pack_q4_0_block(d: float, nibbles: list[int]) -> bytes:
    assert len(nibbles) == 32
    lo = nibbles[:16]
    hi = nibbles[16:]
    qs = bytes((lo[i] & 0x0F) | ((hi[i] & 0x0F) << 4) for i in range(16))
    return struct.pack("<e", d) + qs


def test_q4_0_synthetic_all_zero_block():
    # nibble value 8 decodes to (8 - 8) = 0 regardless of scale.
    raw = _pack_q4_0_block(2.5, [8] * 32)
    out = dequantize("Q4_0", raw, (32,))
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, torch.zeros(32, dtype=torch.bfloat16))


def test_q4_0_synthetic_known_values():
    # nibble 0 -> -8, nibble 15 -> +7; d=2.0 -> values -16.0 / 14.0.
    nibbles = [0] * 16 + [15] * 16
    raw = _pack_q4_0_block(2.0, nibbles)
    out = dequantize("Q4_0", raw, (32,))
    expected = torch.tensor([-16.0] * 16 + [14.0] * 16, dtype=torch.bfloat16)
    assert torch.equal(out, expected)


def _pack_q8_0_block(d: float, values: list[int]) -> bytes:
    assert len(values) == 32
    qs = bytes(v & 0xFF for v in values)
    return struct.pack("<e", d) + qs


def test_q8_0_synthetic_known_values():
    values = [i - 16 for i in range(32)]  # -16 .. 15
    raw = _pack_q8_0_block(0.5, values)
    out = dequantize("Q8_0", raw, (32,))
    expected = torch.tensor([v * 0.5 for v in values], dtype=torch.bfloat16)
    assert torch.equal(out, expected)


def test_dequantize_accepts_int_or_name_interchangeably():
    values = [1] * 32
    raw = _pack_q8_0_block(1.0, values)
    by_name = dequantize("Q8_0", raw, (32,))
    by_int = dequantize(8, raw, (32,))  # 8 == GGML Q8_0 enum value
    assert torch.equal(by_name, by_int)


def test_dequantize_rejects_unsupported_type():
    with pytest.raises(DequantError):
        dequantize("IQ2_XXS", b"\x00" * 66, (256,))


def test_dequantize_rejects_unknown_type_name():
    with pytest.raises(DequantError):
        dequantize("NOT_A_REAL_TYPE", b"\x00" * 18, (32,))


def test_dequantize_rejects_malformed_byte_length():
    with pytest.raises(DequantError):
        dequantize("Q4_0", b"\x00" * 17, (32,))  # not a multiple of 18
