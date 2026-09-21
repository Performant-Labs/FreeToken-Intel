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
def test_q5_k_matches_reference_gguf_package(q4_k_m_gguf_path):
    # Q5_K isn't in the toy Q4_K_M fixture's own quant mix (a real Q4_K_M
    # recipe mixes Q4_K/Q6_K, not Q5_K -- see issue #281), so this test
    # reuses the real target checkpoint that motivated #281 in the first
    # place (already staged on this box, not re-pulled): the real
    # Qwen3.6-35B-A3B q4_k_m checkpoint's own `q4_k_m` recipe does mix in
    # Q5_K for 38 tensors (llama.cpp's per-tensor quant-type selection
    # within a named "quant level" is more nuanced than the level's own
    # name suggests, per #281's own issue body).
    q5_k_path = "/models/weights/staging/qwen3.6-35b-a3b-q4_k_m-gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    import os

    if not os.path.exists(q5_k_path):
        pytest.skip(f"real checkpoint fixture not staged on this box: {q5_k_path}")

    info = gguf_tensor_info(q5_k_path)
    q5_k_names = [n for n, t in info.items() if t.ggml_type_name == "Q5_K"]
    assert q5_k_names, "real checkpoint has no Q5_K tensors -- picked the wrong fixture"

    # Only read a handful of blocks (not the full multi-hundred-MB MoE
    # expert-bank tensor) -- this test only needs to prove the bit
    # unpacking matches the reference, not exercise the whole tensor.
    gguf_ref = pytest.importorskip("gguf")
    type_size = gguf_ref.GGML_QUANT_SIZES[gguf_ref.GGMLQuantizationType.Q5_K][1]
    n_blocks = 100
    tensor_info = info[q5_k_names[0]]
    with open(q5_k_path, "rb") as f:
        f.seek(tensor_info.offset)
        raw = f.read(n_blocks * type_size)

    ours = dequantize("Q5_K", raw, (n_blocks * 256,))
    assert ours.dtype == torch.bfloat16

    raw_arr = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, type_size)
    theirs = gguf_ref.dequantize(raw_arr, gguf_ref.GGMLQuantizationType.Q5_K).reshape(-1)

    np.testing.assert_allclose(ours.to(torch.float32).numpy(), theirs, rtol=1e-2, atol=1e-3)


@pytest.mark.slow
def test_bf16_matches_reference_gguf_package(q4_k_m_gguf_path):
    # BF16 also isn't in the toy fixture (it only has F32/Q4_K/Q6_K); reuse
    # the real checkpoint, which has 2 raw BF16 tensors (issue #281).
    bf16_path = "/models/weights/staging/qwen3.6-35b-a3b-q4_k_m-gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    import os

    if not os.path.exists(bf16_path):
        pytest.skip(f"real checkpoint fixture not staged on this box: {bf16_path}")

    info = gguf_tensor_info(bf16_path)
    bf16_names = [n for n, t in info.items() if t.ggml_type_name == "BF16"]
    assert bf16_names, "real checkpoint has no BF16 tensors -- picked the wrong fixture"

    raw = _read_raw_tensor_bytes(bf16_path, bf16_names[0], info)
    tensor_info = info[bf16_names[0]]

    ours = dequantize("BF16", raw, tensor_info.shape)
    assert ours.dtype == torch.bfloat16
    assert tuple(ours.shape) == tuple(tensor_info.shape)

    gguf_ref = pytest.importorskip("gguf")
    raw_arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 2)
    theirs = gguf_ref.dequantize(raw_arr, gguf_ref.GGMLQuantizationType.BF16).reshape(tensor_info.shape)

    np.testing.assert_allclose(ours.to(torch.float32).numpy(), theirs, rtol=1e-2, atol=1e-3)


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


def test_q5_k_matches_frozen_real_block_bytes():
    """Fast, network-free regression guard for Q5_K's bit-unpacking.

    Frozen from one real Q5_K block (the first block of the real
    Qwen3.6-35B-A3B q4_k_m checkpoint's `blk.0.ffn_down_exps.weight`, the
    same real fixture `test_q5_k_matches_reference_gguf_package` above
    reads live) -- expected values computed once via the reference `gguf`
    pip package's own `dequantize` and hardcoded here so this specific case
    runs in the default CI lane (unlike the `slow`-marked real-file tests
    above, which are excluded from CI's default `pytest -m "not xpu and
    not slow"` run -- mirrors `test_q6_k_matches_frozen_real_block_bytes`,
    added to #271 after its own review for the same reason).
    """
    raw_bytes = bytes.fromhex(
        '74018a0df0e1e8e6b1f6faf0eaf37f1d2f1d3f3fef3f2f2f2fe76f4e2a232b9f1f2e2f0d362eabef0e3e692bcf3f2f259f009090909b90909320909295909090f09b40c0979790909670509095949010363639d6313636c626e69636750b31b63f3686e08636303f36369b33863638413c6530343c013c8f3c3c3c30dce95701b039365c3f3c0c37683a3c9c46303c5ccd2dcd6d75cdfdddc602c9cec7cdc70dbdc7c2ce780fade7c90dc03770cdc03d'
    )
    assert len(raw_bytes) == 176
    out = dequantize("Q5_K", raw_bytes, (256,))
    expected = torch.tensor([
        0.01642751693725586, 0.0004630088806152344, 0.0004630088806152344, 0.0004630088806152344, 0.0004630088806152344, 0.01217031478881836,
        0.0004630088806152344, 0.0004630088806152344, 0.0036559104919433594, 0.0004630088806152344, 0.0004630088806152344, -0.014437198638916016,
        -0.01124429702758789, 0.0004630088806152344, 0.0004630088806152344, 0.0004630088806152344, 0.0004630088806152344, -0.004858493804931641,
        0.0004630088806152344, 0.0004630088806152344, -0.00911569595336914, -0.00911569595336914, 0.0004630088806152344, 0.0004630088806152344,
        -0.010179996490478516, -0.016565799713134766, 0.0004630088806152344, 0.0004630088806152344, 0.005784511566162109, 0.004720211029052734,
        0.0004630088806152344, 0.0004630088806152344, 3.647804260253906e-05, -0.018256187438964844, 3.647804260253906e-05, 3.647804260253906e-05,
        3.647804260253906e-05, 3.647804260253906e-05, 3.647804260253906e-05, 3.647804260253906e-05, 3.647804260253906e-05, -0.005085468292236328,
        3.647804260253906e-05, 3.647804260253906e-05, 3.647804260253906e-05, 3.647804260253906e-05, 3.647804260253906e-05, 3.647804260253906e-05,
        0.004426717758178711, 3.647804260253906e-05, -0.0036220550537109375, -0.0094757080078125, 3.647804260253906e-05, 3.647804260253906e-05,
        3.647804260253906e-05, 3.647804260253906e-05, 3.647804260253906e-05, -0.0014269351959228516, -0.014597654342651367, 3.647804260253906e-05,
        3.647804260253906e-05, 3.647804260253906e-05, 3.647804260253906e-05, -0.01752448081970215, -9.632110595703125e-05, -9.632110595703125e-05,
        0.0025644302368164062, -9.632110595703125e-05, -0.004530906677246094, -9.632110595703125e-05, -9.632110595703125e-05, -9.632110595703125e-05,
        -9.632110595703125e-05, -9.632110595703125e-05, -9.632110595703125e-05, -9.632110595703125e-05, -0.015173912048339844, -0.009852409362792969,
        -0.018721580505371094, -9.632110595703125e-05, 0.007885932922363281, -9.632110595703125e-05, -9.632110595703125e-05, -0.005417823791503906,
        -9.632110595703125e-05, -9.632110595703125e-05, -0.019608497619628906, 0.007885932922363281, -9.632110595703125e-05, -9.632110595703125e-05,
        -0.009852409362792969, -0.01694774627685547, -9.632110595703125e-05, -9.632110595703125e-05, 0.0016775131225585938, -0.004530906677246094,
        -0.00021886825561523438, -0.00021886825561523438, -0.00021886825561523438, 0.008206844329833984, -0.00021886825561523438, -0.00021886825561523438,
        -0.00021886825561523438, 0.0073642730712890625, -0.0010614395141601562, -0.004431724548339844, 0.004836559295654297, -0.00021886825561523438,
        0.003151416778564453, -0.01622772216796875, -0.00021886825561523438, 0.006521701812744141, -0.00021886825561523438, -0.00021886825561523438,
        0.003993988037109375, 0.009049415588378906, -0.009487152099609375, -0.00021886825561523438, -0.00021886825561523438, -0.00021886825561523438,
        -0.00021886825561523438, -0.00021886825561523438, 0.004836559295654297, -0.00021886825561523438, 0.003993988037109375, -0.00021886825561523438,
        -0.00021886825561523438, -0.012857437133789062, -0.00011920928955078125, 0.011455059051513672, 0.005024909973144531, 0.010169029235839844,
        -0.00011920928955078125, 0.006310939788818359, -0.00011920928955078125, 0.003738880157470703, -0.00011920928955078125, -0.00011920928955078125,
        -0.00011920928955078125, -0.015551567077636719, -0.00011920928955078125, -0.003977298736572266, -0.006549358367919922, 0.006310939788818359,
        0.005024909973144531, -0.003977298736572266, -0.00783538818359375, -0.00011920928955078125, 0.024315357208251953, -0.00011920928955078125,
        -0.00011920928955078125, -0.006549358367919922, -0.005263328552246094, 0.017885208129882812, -0.00011920928955078125, -0.00011920928955078125,
        -0.00783538818359375, 0.005024909973144531, -0.00011920928955078125, -0.00011920928955078125, 0.00018668174743652344, -0.014513969421386719,
        0.00018668174743652344, 0.00018668174743652344, 0.00018668174743652344, -0.0032057762145996094, 0.00018668174743652344, 0.005840778350830078,
        0.00018668174743652344, 0.00018668174743652344, 0.00018668174743652344, -0.01790642738342285, 0.011494874954223633, 0.012625694274902344,
        0.0024483203887939453, -0.021298885345458984, -0.008859872817993164, 0.00018668174743652344, 0.00018668174743652344, -0.01564478874206543,
        0.00018668174743652344, 0.00018668174743652344, -0.0032057762145996094, 0.00018668174743652344, -0.014513969421386719, 0.00018668174743652344,
        0.00018668174743652344, 0.006971597671508789, -0.01677560806274414, 0.00018668174743652344, 0.00018668174743652344, 0.0024483203887939453,
        -0.00043463706970214844, -0.00043463706970214844, -0.00043463706970214844, -0.00043463706970214844, 0.010740518569946289, -0.00043463706970214844,
        -0.00043463706970214844, -0.00043463706970214844, -0.010212898254394531, 0.006549835205078125, 0.016328096389770508, 0.02331256866455078,
        -0.008816003799438477, -0.00043463706970214844, -0.008816003799438477, -0.00043463706970214844, -0.00043463706970214844, -0.008816003799438477,
        -0.01580047607421875, 0.0009622573852539062, -0.007419109344482422, 0.002359151840209961, -0.00043463706970214844, 0.013534307479858398,
        -0.006022214889526367, -0.00043463706970214844, 0.0037560462951660156, -0.008816003799438477, 0.0037560462951660156, -0.00043463706970214844,
        -0.01859426498413086, -0.00043463706970214844, -0.0003352165222167969, -0.013860702514648438, -0.0003352165222167969, -0.008450508117675781,
        0.014542818069458008, -0.0003352165222167969, 0.0037224292755126953, 0.0010173320770263672, -0.0003352165222167969, 0.005074977874755859,
        -0.0003352165222167969, -0.0003352165222167969, -0.0003352165222167969, -0.0003352165222167969, -0.0003352165222167969, 0.005074977874755859,
        -0.001687765121459961, -0.0003352165222167969, -0.0003352165222167969, -0.0003352165222167969, -0.007097959518432617, -0.016565799713134766,
        0.0186004638671875, 0.024010658264160156, -0.0003352165222167969, -0.016565799713134766, -0.0003352165222167969, -0.012508153915405273,
        0.014542818069458008, -0.0003352165222167969, -0.0003352165222167969, -0.012508153915405273,
    ], dtype=torch.float32)
    np.testing.assert_allclose(out.to(torch.float32).numpy(), expected.numpy(), rtol=1e-2, atol=1e-3)


def test_bf16_matches_frozen_real_bytes():
    """Fast, network-free regression guard for BF16's passthrough.

    Frozen from the first 16 real BF16 values of the real Qwen3.6-35B-A3B
    q4_k_m checkpoint's `blk.40.ffn_gate_inp.weight` (the same real
    fixture `test_bf16_matches_reference_gguf_package` above reads live) --
    expected values computed once via the reference `gguf` pip package's
    own `dequantize` and hardcoded here so this runs in the default CI
    lane, mirroring `test_q6_k_matches_frozen_real_block_bytes`.
    """
    raw_bytes = bytes.fromhex(
        "51bb35bb0b3ce03af9baf3bb403b8e3a92bbee3bc93b6e3c6f3c88ba3abab53b"
    )
    assert len(raw_bytes) == 32
    out = dequantize("BF16", raw_bytes, (16,))
    expected = torch.tensor([
        -0.0031890869140625, -0.0027618408203125, 0.00848388671875, 0.001708984375,
        -0.00189971923828125, -0.007415771484375, 0.0029296875, 0.0010833740234375,
        -0.00445556640625, 0.00726318359375, 0.006134033203125, 0.0145263671875,
        0.01458740234375, -0.00103759765625, -0.00070953369140625, 0.005523681640625,
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
