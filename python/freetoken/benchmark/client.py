"""bs=1 decode-step timing against a real Engine (issue `benchmarks`).

Upstream NVIDIA path: python/freetoken/benchmark/client.py
Fill in: GitHub issue `benchmarks` (see docs/architecture.md).

Drives the exact per-step primitive ``ft serve`` drives (``Engine.step``,
``@torch.inference_mode()``) directly on the main thread, one request at a
time -- not a threaded HTTP client. Two reasons:

* The offload/cpu/hybrid MoE paths do a device->host sync per decode step;
  the XPU runtime faults on that sync when it is issued off the main thread
  (see ``test_serve_live_engine_xpu.py``'s docstring for the same constraint
  on the route's own ``TestClient``-driven tests). Calling ``Engine.step``
  directly keeps everything on the thread that built the engine.
* ``Engine.generate()`` (which the real ``/v1/chat/completions`` route calls
  through ``stream_chat``) runs its whole decode loop synchronously and only
  *then* hands back the full id list for the caller to decode incrementally
  -- so timing ``stream_chat``'s yielded text deltas would not measure real
  per-token latency, everything has already happened by the first yield.
  Looping ``Engine.step`` ourselves is the same primitive the route uses
  underneath, just with per-step timestamps.
"""
from __future__ import annotations

import time

from freetoken.benchmark.perf import StepTiming
from freetoken.core import Req, SamplingParams


def run_client(
    engine,
    prompt_ids: list[int],
    *,
    backend: str,
    max_tokens: int,
    uid: int,
) -> StepTiming:
    """Admit one bs=1 request and time its steps: first = prefill (TTFT).

    Every step after the first is a decode step; ``StepTiming.decode_tok_s``
    covers only those. Raises if the engine produces no tokens at all (an
    empty ``next_token_ids`` on the very first step means admission failed).
    """
    req = Req(
        input_ids=list(prompt_ids),
        table_idx=0,  # add_request reassigns the real scheduler slot
        cached_len=0,
        output_len=max_tokens,
        uid=uid,
        sampling_params=SamplingParams(max_tokens=max_tokens),
        cache_handle=None,
    )
    engine.add_request(req)

    step_times: list[float] = []
    while len(step_times) < max_tokens:
        t0 = time.perf_counter()
        out = engine.step()
        t1 = time.perf_counter()
        if out.next_token_ids is None or len(out.next_token_ids) == 0:
            break
        step_times.append(t1 - t0)

    if not step_times:
        raise RuntimeError(f"backend={backend!r}: engine produced no tokens (bad admission?)")

    ttft, *decode_step_times = step_times
    return StepTiming(
        backend=backend,
        ttft_s=ttft,
        decode_tokens=len(decode_step_times),
        decode_s=sum(decode_step_times),
    )


__all__ = ["run_client"]
