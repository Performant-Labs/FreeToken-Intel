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


def test_q6_k_matches_frozen_real_block_bytes():
    """Fast, network-free regression guard for Q6_K's bit-unpacking.

    Frozen from one real Q6_K block (the first block of a real
    checkpoint's own `output.weight`, from the same fixture family
    `test_q6_k_matches_reference_gguf_package` above downloads live) --
    expected values computed once via the reference `gguf` pip package's
    own `dequantize` and hardcoded here so this specific case runs in the
    default CI lane (unlike the `slow`-marked real-file tests above, which
    need network access and are excluded from CI's default
    `pytest -m "not xpu and not slow"` run -- this test exists
    specifically to close that coverage gap for Q6_K's bit-unpacking, the
    one place in this module where a subtly wrong element-index alignment
    would produce plausible-looking-but-wrong numbers, not a crash).
    """
    raw_bytes = bytes.fromhex(
        'abb538fca4700e1a283c70b00ae6e0e91fd6857f05e76fa760db4b97414e4163d986f108356b3661c56e092864692d6a5a1377079538c925203191cec61ad178f1c2d278fd13630deacc0e2e5483c4d0096b0d0a9dfbc826e5d2f5f11c9638c10bd5c50f385adcf28448937df108f0db0de424960bd7c2d0866af8fd259205fc7e92f3214c190364b55e42203ee5c20cfe69cc1d86a8c4299347a00f109ca0991446f73e2e0f439ffd3a3f6bf1708141cca39a7b3bd0f0415aa3e8c74930aa7484828f72666f788784807a7882767d848282'
    )
    assert len(raw_bytes) == 210
    out = dequantize("Q6_K", raw_bytes, (256,))
    expected = torch.tensor([
        0.052195072174072266, 0.023725032806396484, 0.11388015747070312, -0.018980026245117188, -0.1328601837158203, -0.07592010498046875,
        0.1423501968383789, -0.10439014434814453, -0.037960052490234375, 0.05694007873535156, 0.0, -0.1518402099609375,
        0.04745006561279297, -0.04745006561279297, 0.0, -0.10913515090942383, 0.07232308387756348, -0.048215389251708984,
        -0.13018155097961426, -0.0048215389251708984, 0.024107694625854492, -0.12053847312927246, -0.08196616172790527, -0.043393850326538086,
        0.07714462280273438, 0.13018155097961426, -0.10125231742858887, 0.11089539527893066, -0.14946770668029785, -0.08678770065307617,
        -0.14946770668029785, -0.06268000602722168, 0.10810196399688721, -0.1124260425567627, -0.13404643535614014, -0.10377788543701172,
        0.09080564975738525, 0.04756486415863037, -0.1124260425567627, -0.06486117839813232, -0.04756486415863037, 0.12972235679626465,
        -0.09945380687713623, -0.10377788543701172, 0.08648157119750977, -0.030268549919128418, -0.08215749263763428, 0.1124260425567627,
        -0.11342096328735352, -0.013087034225463867, -0.10033392906188965, -0.10033392906188965, 0.04798579216003418, -0.03489875793457031,
        0.030536413192749023, -0.021811723709106445, 0.13959503173828125, 0.06543517112731934, 0.13523268699645996, -0.13087034225463867,
        0.11342096328735352, -0.11342096328735352, 0.13523268699645996, -0.03489875793457031, -0.10148191452026367, 0.0195157527923584,
        -0.07415986061096191, -0.058547258377075195, 0.08586931228637695, 0.03512835502624512, 0.12490081787109375, -0.0039031505584716797,
        -0.07025671005249023, 0.050740957260131836, 0.09757876396179199, -0.04293465614318848, -0.062450408935546875, -0.054644107818603516,
        0.07025671005249023, 0.07025671005249023, -0.07220828533172607, -0.05521810054779053, 0.10194110870361328, 0.03822791576385498,
        0.13592147827148438, -0.059465646743774414, 0.11043620109558105, -0.04247546195983887, 0.04247546195983887, 0.08070337772369385,
        -0.016990184783935547, 0.0976935625076294, 0.05097055435180664, 0.05097055435180664, -0.016990184783935547, 0.04247546195983887,
        0.013775825500488281, -0.03673553466796875, -0.1423501968383789, 0.146942138671875, 0.05969524383544922, 0.11939048767089844,
        0.13316631317138672, 0.04591941833496094, -0.055103302001953125, 0.04591941833496094, 0.0734710693359375, 0.1377582550048828,
        0.11939048767089844, -0.10102272033691406, -0.08265495300292969, 0.11939048767089844, 0.09723436832427979, -0.06945312023162842,
        0.10649478435516357, -0.14816665649414062, 0.04167187213897705, 0.013890624046325684, 0.12964582443237305, -0.13890624046325684,
        0.009260416030883789, -0.06019270420074463, 0.04167187213897705, -0.09260416030883789, -0.09260416030883789, 0.0046302080154418945,
        0.06019270420074463, 0.03241145610809326, -0.1470952033996582, 0.009490013122558594, 0.08541011810302734, 0.037960052490234375,
        0.06168508529663086, 0.09015512466430664, 0.09015512466430664, 0.1376051902770996, -0.02847003936767578, 0.05694007873535156,
        0.1423501968383789, 0.1423501968383789, -0.05694007873535156, -0.1376051902770996, -0.05694007873535156, -0.07592010498046875,
        -0.1126556396484375, 0.1322479248046875, 0.0636749267578125, 0.127349853515625, 0.1420440673828125, -0.1028594970703125,
        -0.1175537109375, -0.048980712890625, 0.0244903564453125, 0.088165283203125, -0.1322479248046875, 0.0832672119140625,
        -0.01959228515625, -0.127349853515625, 0.0391845703125, -0.1518402099609375, 0.023342370986938477, 0.05135321617126465,
        0.05135321617126465, -0.14472270011901855, -0.11204338073730469, -0.12138032913208008, 0.0933694839477539, -0.08403253555297852,
        -0.0933694839477539, -0.03734779357910156, -0.08870100975036621, -0.06069016456604004, 0.14472270011901855, 0.11204338073730469,
        0.14939117431640625, 0.0980379581451416, -0.13316631317138672, 0.12857437133789062, -0.018367767333984375, -0.027551651000976562,
        -0.05051136016845703, 0.11479854583740234, 0.1377582550048828, 0.146942138671875, -0.027551651000976562, 0.10102272033691406,
        -0.03673553466796875, 0.013775825500488281, -0.02295970916748047, 0.1377582550048828, -0.02295970916748047, 0.018367767333984375,
        -0.0048215389251708984, -0.09643077850341797, 0.13982462882995605, 0.11089539527893066, 0.07232308387756348, -0.14946770668029785,
        -0.12536001205444336, -0.07714462280273438, 0.14464616775512695, 0.13500308990478516, 0.07714462280273438, 0.009643077850341797,
        0.10125231742858887, 0.11571693420410156, -0.09643077850341797, -0.09160923957824707, 0.14449310302734375, -0.027092456817626953,
        0.07224655151367188, -0.07224655151367188, -0.1128852367401123, 0.004515409469604492, -0.12643146514892578, 0.13546228408813477,
        0.009030818939208984, -0.0587003231048584, -0.06773114204406738, 0.07676196098327637, 0.13997769355773926, -0.1128852367401123,
        -0.013546228408813477, -0.12643146514892578, 0.15306472778320312, 0.014349818229675293, -0.13393163681030273, 0.15306472778320312,
        0.13871490955352783, 0.12914836406707764, 0.014349818229675293, -0.07174909114837646, -0.11479854583740234, 0.13393163681030273,
        0.11001527309417725, 0.04304945468902588, -0.14828145503997803, 0.07653236389160156, -0.07174909114837646, 0.014349818229675293,
        0.07592010498046875, 0.06643009185791016, 0.009490013122558594, -0.03321504592895508, -0.1518402099609375, 0.1376051902770996,
        0.1328601837158203, -0.01423501968383789, -0.037960052490234375, 0.02847003936767578, 0.1470952033996582, 0.1470952033996582,
        -0.06643009185791016, -0.10913515090942383, 0.0, -0.004745006561279297,
    ], dtype=torch.float32)
    np.testing.assert_allclose(out.to(torch.float32).numpy(), expected.numpy(), rtol=1e-2, atol=1e-3)


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
