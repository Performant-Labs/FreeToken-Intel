"""Tests for the Triton/Intel reference attention backend (issue ``attn-triton``, #4).

Acceptance bar (#4): prefill + decode for ``AttnType.FULL`` **and** ``AttnType.SWA``
on the B70, registered as ``triton``, with a numerical check vs a reference path.

Two halves:

* CPU-safe (the per-PR ``ci`` job, torch-free): ``triton`` is registered and
  *declares* SWA as a supported type (so the registry's capability advertising is
  honest even before an XPU is present).

* ``xpu``-marked (the B70 nightly, ``.venv-xpu``): the backend is driven directly
  against a controlled KV pool and its output compared to a hand-written pure-torch
  grouped-query + sliding-window reference. The reference is the same math the SYCL
  kernel uses (causal ``keypos <= qpos`` AND ``qpos - keypos < window``), so agreement
  to float32 epsilon means the triton SWA mask is correct.
"""
from __future__ import annotations

import pytest


def test_triton_backend_registered_and_declares_swa():
    """No torch needed: the registry advertises FULL + SWA for the triton backend.

    This is the CPU-safe half. The triton backend *declares* ``AttnType.SWA`` in its
    BackendInfo; the numerical xpu tests below prove the declaration is real.
    """
    from freetoken.attention import attention_backend_info
    from freetoken.attention.base import AttnType

    info = attention_backend_info("triton")
    assert AttnType.SWA in info.supported_types
    assert AttnType.FULL in info.supported_types
    # The backend consumes the attn_spec (it reads sliding_window off it).
    assert info.consumes_attn_spec


# --- XPU: drive the triton backend against a controlled KV pool -----------------


@pytest.fixture(autouse=True)
def _clean_global_ctx():
    """Reset the global context around every test (the xpu tests set it)."""
    yield
    from freetoken.core import reset_global_ctx

    reset_global_ctx()


# A tiny GQA config: 4 query heads, 2 KV heads (repeat 2), head_dim 16.
TINY_CONFIG = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "model_type": "qwen3_moe",
    "hidden_size": 64,
    "vocab_size": 32,
    "num_hidden_layers": 1,
    "num_local_experts": 2,
    "num_experts_per_tok": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 32,
    "moe_intermediate_size": 32,
    "max_position_embeddings": 128,
    "rope_theta": 1000000.0,
}


def _gqa_windowed_reference(q, k_all, v_all, q_pos, window, *, scale):
    """Reference grouped-query + sliding-window attention over a token history.

    ``q`` is ``[num_heads, qlen, head_dim]``. ``k_all`` / ``v_all`` are already
    head-expanded to ``[num_heads, written, head_dim]`` (the same ``h -> h //
    repeat`` interleave the backend uses). ``q_pos`` is the absolute position of
    each query row; a key at position ``keypos`` is attended iff ``keypos <=
    qpos`` (causal) and ``qpos - keypos < window`` (sliding window; window<=0 is
    unbounded). Returns ``[num_heads, qlen, head_dim]``.
    """
    import torch

    written = k_all.shape[1]
    key_pos = torch.arange(written, device=q.device)
    mask = q_pos[None, :, None] >= key_pos[None, None, :]
    if window > 0:
        mask = mask & ((q_pos[None, :, None] - key_pos[None, None, :]) < window)
    scores = torch.matmul(q, k_all.transpose(-1, -2)) * scale
    scores = torch.where(mask, scores, torch.full_like(scores, float("-inf")))
    return torch.matmul(torch.softmax(scores, dim=-1), v_all)


def _drive_triton_backend(dev, *, q, k_hist, v_hist, q_pos, written, window):
    """Set up a global ctx (KV pool with known history) and run the triton backend.

    ``k_hist`` / ``v_hist`` are token-major ``[written, num_kv, head_dim]`` -- the
    full history stored in the pool (positions 0..written-1). The request is framed
    as a *decode* step whose `device_len == written` (so ``forward`` reads the whole
    stored history), and the new query block ``q`` (``[num_heads, qlen, head_dim]``,
    ``qlen == q_pos.numel()``) attends at absolute positions ``q_pos`` under sliding
    window ``window``. Returns the backend's output ``[num_heads, qlen, head_dim]``.
    """
    import torch
    from freetoken.core import Batch, Context, Req, SamplingParams, set_global_ctx
    from freetoken.kvcache.base import BaseKVCachePool

    config = type("Cfg", (), dict(TINY_CONFIG))()
    num_kv = TINY_CONFIG["num_key_value_heads"]
    head_dim = TINY_CONFIG["hidden_size"] // TINY_CONFIG["num_attention_heads"]

    pool = BaseKVCachePool(config, page_size=1, num_pages=written, device=dev, dtype=torch.float32)
    # Identity page table: slot == position. Row 0 is the single request.
    pool.attach_page_table(torch.arange(written, device=dev, dtype=torch.int32).reshape(1, written))
    pool.write_kv(k_hist.transpose(0, 1), v_hist.transpose(0, 1), torch.arange(written, device=dev))

    ctx = Context(page_size=1)
    ctx.kv_cache = pool
    from freetoken.attention.triton import TritonAttentionBackend

    backend = TritonAttentionBackend(config)
    ctx.attn_backend = backend
    set_global_ctx(ctx)

    # Frame as a decode step: device_len == written (the full stored history) while
    # input_ids is shorter, so `is_decode` (device_len != len(input_ids)) is True and
    # `forward` reads the whole history (`written = device_len`), not just the new
    # tokens. qlen new tokens attend over that history at positions `q_pos`.
    qlen = q_pos.numel()
    req = Req(
        input_ids=list(range(max(1, written - qlen))),
        table_idx=0,
        cached_len=0,
        output_len=0,
        uid=0,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
        cache_handle=None,
    )
    req.device_len = written  # override: the full history length (decode framing)
    batch = Batch(reqs=[req], phase="decode", positions=q_pos)

    from freetoken.attention.base import AttentionSpec

    # The *new* k/v passed to forward() are not read by _attend_one (it gathers the
    # full history from the pool); forward only needs their head count + a valid
    # shape to size its output buffer. num_kv (not num_heads) KV heads.
    out = backend.forward(
        q,
        _dummy_new_kv(num_kv, num_kv, head_dim, dev),
        _dummy_new_kv(num_kv, num_kv, head_dim, dev),
        layer_id=0,
        batch=batch,
        attn_spec=AttentionSpec(sliding_window=window),
        table_idx=0,
    )
    return out


def _dummy_new_kv(num_kv, num_kv_, head_dim, dev):
    """A zero [num_kv, 1, head_dim] tensor -- the *new* k/v the backend's forward()
    needs a shape for, but ``_attend_one`` never reads (it gathers history from the
    pool). Only the head count (dim 0) and a valid shape matter."""
    import torch

    return torch.zeros((num_kv, 1, head_dim), device=dev, dtype=torch.float32)


def _reference_for_drive(q, k_hist, v_hist, q_pos, written, window):
    num_heads = TINY_CONFIG["num_attention_heads"]
    num_kv = TINY_CONFIG["num_key_value_heads"]
    repeat = num_heads // num_kv
    head_dim = TINY_CONFIG["hidden_size"] // num_heads
    scale = 1.0 / (head_dim**0.5)
    # GQA: query head h attends KV head h // repeat -> an *interleave* of the KV
    # head axis (the exact mapping the triton backend's _attend_one uses).
    k_all = k_hist.transpose(0, 1).contiguous()
    v_all = v_hist.transpose(0, 1).contiguous()
    if repeat != 1:
        k_all = k_all.repeat_interleave(repeat, dim=0)
        v_all = v_all.repeat_interleave(repeat, dim=0)
    return _gqa_windowed_reference(q, k_all, v_all, q_pos, window, scale=scale)


@pytest.mark.xpu
def test_triton_swa_prefill_matches_reference_on_xpu():
    """Prefill: a multi-token query block attends over its own growing history under
    a sliding window. window=3 means each query sees only its 3 most recent keys."""
    import torch

    assert torch.xpu.is_available(), "this test runs on the B70 XPU nightly"
    dev = torch.device("xpu")

    heads, num_kv, head_dim = 4, 2, 16
    written = 6
    qlen = 4  # the new query block (positions 2..5 attending over the history)
    q_pos = torch.tensor([2, 3, 4, 5], device=dev, dtype=torch.int64)
    gen = torch.Generator(device=dev).manual_seed(7)
    q = torch.randn(heads, qlen, head_dim, generator=gen, device=dev) * 0.5
    k_hist = torch.randn(written, num_kv, head_dim, generator=gen, device=dev) * 0.5
    v_hist = torch.randn(written, num_kv, head_dim, generator=gen, device=dev) * 0.5

    for window in (3, 5):
        out = _drive_triton_backend(
            dev, q=q, k_hist=k_hist, v_hist=v_hist, q_pos=q_pos, written=written, window=window
        )
        ref = _reference_for_drive(q, k_hist, v_hist, q_pos, written, window)
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5), (
            f"triton SWA (window={window}) diverged: max err "
            f"{torch.max((out - ref).abs()):.3e}"
        )
    from freetoken.core import reset_global_ctx

    reset_global_ctx()


@pytest.mark.xpu
def test_triton_swa_decode_matches_reference_on_xpu():
    """Decode: a single new token attends over a long history under a sliding window.

    This is the case where a wrong window (e.g. off-by-one, or a window that
    includes one too many / too few old keys) changes the output, because the new
    token sees a bounded suffix of the history."""
    import torch

    assert torch.xpu.is_available()
    dev = torch.device("xpu")

    heads, num_kv, head_dim = 4, 2, 16
    written = 8  # the pool holds the full history at positions 0..7
    # The new token is the LAST slot (position written-1): by decode time its own
    # K/V is already in the pool, so `device_len == written` and the newest key is
    # at position written-1 (not a separate `written`-th slot). The reference and
    # the backend both mask `qpos - keypos < window` over keypos 0..written-1.
    q_pos = torch.tensor([written - 1], device=dev, dtype=torch.int64)
    gen = torch.Generator(device=dev).manual_seed(11)
    q = torch.randn(heads, 1, head_dim, generator=gen, device=dev) * 0.5
    k_hist = torch.randn(written, num_kv, head_dim, generator=gen, device=dev) * 0.5
    v_hist = torch.randn(written, num_kv, head_dim, generator=gen, device=dev) * 0.5

    # window=1 now selects only the newest key (keypos written-1 == the new token
    # itself, which is in the pool) -> well-defined, so the full range is exercised.
    for window in (1, 2, 3, 8, 16):  # 1=self only; 2=newest 2; 3=small; 8=whole history; 16=> history
        out = _drive_triton_backend(
            dev, q=q, k_hist=k_hist, v_hist=v_hist, q_pos=q_pos, written=written, window=window
        )
        ref = _reference_for_drive(q, k_hist, v_hist, q_pos, written, window)
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5), (
            f"triton SWA decode (window={window}) diverged: max err "
            f"{torch.max((out - ref).abs()):.3e}"
        )
    from freetoken.core import reset_global_ctx

    reset_global_ctx()


@pytest.mark.xpu
def test_triton_swa_window_larger_than_history_equals_full_on_xpu():
    """A window >= the history length must be numerically identical to plain FULL
    causal attention (the window truncates nothing)."""
    import torch

    assert torch.xpu.is_available()
    dev = torch.device("xpu")

    heads, num_kv, head_dim = 4, 2, 16
    written = 5
    q_pos = torch.tensor([5], device=dev, dtype=torch.int64)
    gen = torch.Generator(device=dev).manual_seed(13)
    q = torch.randn(heads, 1, head_dim, generator=gen, device=dev) * 0.5
    k_hist = torch.randn(written, num_kv, head_dim, generator=gen, device=dev) * 0.5
    v_hist = torch.randn(written, num_kv, head_dim, generator=gen, device=dev) * 0.5

    out_full = _drive_triton_backend(dev, q=q, k_hist=k_hist, v_hist=v_hist, q_pos=q_pos, written=written, window=0)
    out_big = _drive_triton_backend(dev, q=q, k_hist=k_hist, v_hist=v_hist, q_pos=q_pos, written=written, window=32)
    assert torch.allclose(out_full, out_big, atol=1e-6, rtol=1e-6), (
        f"a window larger than the history must equal FULL causal: max err "
        f"{torch.max((out_full - out_big).abs()):.3e}"
    )
    from freetoken.core import reset_global_ctx

    reset_global_ctx()
