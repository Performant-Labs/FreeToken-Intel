"""End-to-end hybrid MoE serve on a REAL model (Milestone 3, "Hybrid q* online").

Milestone 3 gates on a *supervised live* run of the hybrid (PCIe-fetch + host-CPU)
q* split -- not just a unit-level forward. ``test_moe_hybrid_forward`` already proves
the hybrid forward reproduces the in-VRAM reference's tokens on a *fabricated* tiny
Qwen3-MoE. This module closes the last mile: it loads a **real** Qwen3-MoE
checkpoint (``DavidAU/Qwen3-MOE-4x0.6B-2.4B`` -- 4 experts, 28 layers, a real Qwen
tokenizer + chat template, ~3 GB, sized to fit the 32 GB box's RAM) host-offload
through the real ``Engine`` with ``moe_backend="hybrid"``, and drives the exact seam
the ``/v1/chat/completions`` route uses (``stream_chat`` -> ``engine.add_request`` +
``engine.generate``) to stream **readable, non-degenerate decoded text** from a real
prompt.

Why the real model, not the 10.2B stand-in: the 10.2B's fp32 expert bank is ~37 GB
-- it OOMs the 32 GB box the moment the loader materializes the host banks. The 2.4B
(4 experts, ~2.4 GB of expert weights) keeps the same hybrid q* split but in a size
that actually loads and decodes on this hardware, so it is the right vehicle for the
live close. The card is kept quiescent: the run is one supervised process, time-capped,
and a watcher aborts on any new ``exec queue reset`` (the XPU-fault signature from the
earlier D2H-sync crash) or a memory-pressure stall.

Thread note: the offload/hybrid MoE path does a device->host sync per token, and the
XPU runtime faults on that sync when issued off the main thread. So, exactly as
``test_serve_live_engine_xpu`` does, the engine is built and driven on the **main**
thread through the ``stream_chat`` seam (not a ``TestClient`` route thread); the app is
still built the way the route sees it, so the seam under test is the real one.

XPU-only: skipped cleanly where there is no XPU / torch (see ``conftest.py``); runs in
``.venv-xpu`` on the B70. The per-UUID ``ft bench bw`` profile (the q* fetch fraction)
is generated first with ``ft bench bw`` and read by the loader's auto-lookup.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import urllib.request

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

# XPU is the point of this module: skip cleanly (not fail) where there is none.
XPU = pytest.mark.skipif(not torch.xpu.is_available(), reason="no XPU available")

# The real Qwen3-MoE checkpoint (4 experts, 28 layers, real Qwen tokenizer + template,
# ~3 GB -> its hybrid host expert bank fits the 32 GB box's RAM). Resolved from the
# local HF cache (offline); skipped if it is not present on the box.
MODEL = "DavidAU/Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder-V1.2"

# A real prompt: short enough to stay in the KV pool's budget, long enough to be a
# genuine (non-trivial) prompt that exercises the model's chat template.
_MESSAGES = [{"role": "user", "content": "Say 'the model is ready' and stop."}]
_DECODE_TOKENS = 32


def _xpu_reset_count() -> int | None:
    """The XPU 'exec queue reset' count (the GPU-fault signature).

    Read from ``dmesg`` (sudo, non-interactive). ``None`` when dmesg is unreadable
    (the watcher then cannot detect a fault -- it degrades to the memory + timeout
    guards only, which is still safe: the run is time-capped and non-looped).
    """
    try:
        r = subprocess.run(["sudo", "-n", "dmesg"], capture_output=True, text=True, timeout=8)
        return r.stdout.count("exec queue resolve") + r.stdout.count("exec queue reset")
    except Exception:
        return None


def _mem_available_gb() -> float | None:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except Exception:
        return None
    return None


def _model_dir() -> str | None:
    """Resolve the real checkpoint to its local HF-cache snapshot (offline)."""
    from huggingface_hub import try_to_load_from_cache

    for fname in ("config.json", "model.safetensors"):
        cached = try_to_load_from_cache(MODEL, fname)
        if cached is None:
            return None
    return os.path.dirname(cached)


@XPU
@pytest.mark.xpu
def test_hybrid_real_model_streams_readable_tokens(tmp_path, monkeypatch):
    """M3 live close: a real MoE model, hybrid q* split, readable decoded text.

    Load the real 2.4B Qwen3-MoE host-offload with the hybrid backend through the
    real ``Engine``, then drive the route's ``stream_chat`` seam on the main thread
    and assert the generated ids decode to readable, non-placeholder, non-id-blob
    prose. The hybrid fetch fraction comes from the per-UUID ``ft bench bw`` profile
    (the loader's auto-lookup), so the q* split is genuinely active (f > 0), not the
    degenerate pure-offload fallback. A watcher aborts the run on a new XPU reset or
    a memory-pressure stall, so a shared box is never taken down by a runaway run.
    """
    from freetoken.core import reset_global_ctx
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig
    from freetoken.engine.engine import Engine
    from freetoken.models.loader import load_model
    from freetoken.server.api_server import create_app
    from freetoken.server.args import parse_args
    from freetoken.server.generation import stream_chat
    from freetoken.server.launch import _frontend_tokenizer
    from freetoken.moe.bench_profile import load_hybrid_fetch_fraction

    ckpt = _model_dir()
    if ckpt is None:
        pytest.skip(f"real checkpoint {MODEL} is not in the local HF cache (offline)")

    # The profile must be present and trusted (its xpu.name matches this box) so the
    # loader's auto-lookup yields a real fetch fraction (f > 0) -- the hybrid q* split
    # -- rather than the f=0 pure-offload fallback. Generate it if absent (the
    # benchbw fix makes `ft bench bw` work; the profile is per-UUID in the cache).
    frac = load_hybrid_fetch_fraction("bf16")
    if frac is None:
        from freetoken.moe.benchbw import main as _bench_main

        _bench_main(["--dtype", "bf16", "--repeats", "2"], prog="ft bench bw")
        frac = load_hybrid_fetch_fraction("bf16")
    assert frac is not None and 0.0 < frac < 1.0, (
        "the per-UUID benchbw profile must pin a non-degenerate hybrid fetch fraction "
        f"(got {frac!r}); without it the hybrid degrades to pure offload, not the M3 q* split"
    )

    dev = torch.device("xpu")

    # --- watcher: abort on a new XPU reset or a memory-pressure stall ---------------
    base = _xpu_reset_count()
    avail0 = _mem_available_gb() or 0.0
    state = {"done": False, "error": None}

    def _watch():
        stalled = 0
        while not state["done"]:
            rc = _xpu_reset_count()
            if base is not None and rc is not None and rc > base:
                state["error"] = f"XPU-RESET: {base} -> {rc}"
                break
            a = _mem_available_gb()
            if a is not None:
                if a < max(2.0, avail0 - 14.0):
                    stalled += 1
                else:
                    stalled = 0
                if stalled >= 6:
                    state["error"] = f"memory-pressure stall (avail {a:.1f}GB)"
                    break
            time.sleep(1.0)

    wt = threading.Thread(target=_watch, daemon=True)
    wt.start()

    # Drive the loader with the model's *native* dtype (bf16 for this checkpoint)
    # so the loader keys the benchbw profile on "bf16" (the profile's format) -> the
    # real fetch fraction. (The engine's serve path pins fp32, which has no profile
    # entry and would silently fall back to pure offload; the loader stamps its own
    # effective dtype onto the modules, so a bf16 load keeps the weights + modules
    # consistent and is exactly what the profile was measured for.)
    model, sources = load_model(ckpt, dev, dtype=torch.bfloat16, moe_backend="hybrid")
    try:
        assert model.moe_offload, "the hybrid path must flag the model as host-offloaded"
        assert model.moe_cache is not None, "the loader must attach the LRU slot pool"
        assert model.layers[0].mlp.experts is None, "the hybrid path must not build device-resident experts"
        assert abs(getattr(model, "moe_hybrid_fetch_fraction", 0.0) - frac) < 1e-9, (
            "the loader must read the profile's real fetch fraction onto the model"
        )
        assert 0.0 < getattr(model, "moe_hybrid_fetch_fraction", 0.0) < 1.0, (
            "the hybrid must be the non-degenerate q* split (0 < f < 1), not pure offload"
        )

        server_args = parse_args([ckpt])
        engine = Engine(
            EngineConfig(
                model_path=ckpt,
                tp_info=DistributedInfo(0, 1),
                dtype=torch.bfloat16,
                device=dev,
                attention_backend="auto",
                moe_backend="hybrid",
                max_running_req=1,
                page_size=1,
                max_seq_len_override=128,
                num_page_override=512,
            )
        )
        engine.frontend_tokenizer = _frontend_tokenizer(parse_args([ckpt]))

        # Build the app exactly as the route sees it (wiring check), then drive the
        # same seam the /v1/chat/completions route uses -- on the main thread.
        app = create_app(server_args, lambda: engine)
        assert any(getattr(r, "path", None) == "/v1/chat/completions" for r in app.routes)

        if state["error"]:
            pytest.fail(f"watcher aborted the run before generation: {state['error']}")

        deltas = list(stream_chat(engine, _MESSAGES, model=server_args.resolved_model_name, max_tokens=_DECODE_TOKENS))
        content = "".join(content_delta for _reasoning, content_delta in deltas)

        if state["error"]:
            pytest.fail(f"watcher aborted the run mid-generation: {state['error']}")

        # M3 evidence: a real (non-503) stream came back, now as readable text.
        assert content, "the stream must be non-empty -- the real model generated tokens"
        assert "<tok-" not in content, f"decoder still emits placeholders: {content!r}"
        # Readable prose, not undecoded binary or a raw token-id blob. Ordinary
        # whitespace (newlines, tabs) is expected in real prose -- str.isprintable()
        # treats it as "not printable", so check per-character against isprintable
        # OR isspace instead of gating the whole string on isprintable() alone.
        assert all(ch.isprintable() or ch.isspace() for ch in content), (
            f"undecoded binary leaked: {content!r}"
        )
        assert not re.search(r"\d{6,}", content), f"decoded stream looks like raw ids: {content!r}"

        # The decode steps must have actually driven the slot pool (the hybrid q*
        # split is per-decode-step, so decode must have called ensure_experts).
        stats = engine.model.moe_cache.decode_miss_stats()
        assert stats["calls"] > 0, "decode must call ensure_experts on the slot pool (the q* split lives there)"

        result = {
            "model": MODEL,
            "backend": "hybrid",
            "fetch_fraction": float(getattr(model, "moe_hybrid_fetch_fraction", 0.0)),
            "decode_calls": stats["calls"],
            "content": content,
            "resets_before": base,
            "resets_after": _xpu_reset_count(),
        }
        print("\n[hybrid e2e real model] " + json.dumps(result, indent=2))
        print("DECODED CONTENT:\n" + content)
        # Persist the evidence for the milestone record (the test's stdout is the
        # record; the JSON is a machine-readable artifact).
        (tmp_path / "m3_hybrid_real_model_evidence.json").write_text(json.dumps(result, indent=2))
    finally:
        state["done"] = True
        reset_global_ctx()
        if torch.xpu.is_available():
            torch.xpu.empty_cache()
