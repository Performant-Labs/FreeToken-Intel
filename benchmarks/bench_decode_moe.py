"""bs=1 decode throughput of a served MoE on B70, per backend.

Upstream NVIDIA path: benchmarks/bench_decode_moe.py
Fill in: GitHub issue `benchmarks` (see docs/architecture.md).

Run from the repo root, in ``.venv-xpu``, pinned to one XPU::

    ZE_AFFINITY_MASK=0 .venv-xpu/bin/python benchmarks/bench_decode_moe.py \\
        <checkpoint path or HF id> --backends offload,cpu,hybrid

For each ``--backends`` entry this builds a fresh ``Engine`` (real weights,
real ``ft bench bw`` profile for hybrid), runs ``--repeats`` bs=1 decodes
through the same per-step primitive ``ft serve`` drives
(``freetoken.benchmark.client.run_client`` -- see its docstring for why this
is not a threaded HTTP client), then fully tears the engine down (dropped
Python refs, ``torch.xpu.empty_cache()``) *before* building the next
backend's -- one MoE backend's host banks / XPU state on the box at a time,
never two overlapping.

Prints a table (mean time-to-first-token, mean decode tok/s -- the number
comparable to llama.cpp / vLLM SYCL "tg" throughput on the same card) via
``freetoken.benchmark.perf``.
"""
from __future__ import annotations

import argparse
import gc
import sys


def _build_argparser(prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description=__doc__)
    p.add_argument("model", help="checkpoint path or HF id (a MoE architecture)")
    p.add_argument(
        "--backends",
        default="offload,cpu,hybrid",
        help="comma-separated MoE backends to compare (default: %(default)s)",
    )
    p.add_argument(
        "--prompt",
        default="Explain what a mixture-of-experts model is, in two sentences.",
        help="the (single-turn) user prompt to decode from",
    )
    p.add_argument("--max-tokens", type=int, default=64, help="decode budget per repeat")
    p.add_argument("--repeats", type=int, default=3, help="bs=1 decodes averaged per backend")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    return p


def main(argv: list[str] | None = None, prog: str = "bench_decode_moe") -> int:
    args = _build_argparser(prog).parse_args(argv)

    import torch

    from freetoken.benchmark.client import run_client
    from freetoken.benchmark.perf import format_table, summarize
    from freetoken.core import reset_global_ctx
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig
    from freetoken.engine.engine import Engine
    from freetoken.server.args import parse_args as parse_server_args
    from freetoken.server.generation import _prompt_from_messages, _prompt_token_ids
    from freetoken.server.launch import _frontend_tokenizer
    from freetoken.utils.arch import is_xpu_available

    if not is_xpu_available():
        print("bench_decode_moe: no XPU available -- nothing to benchmark", file=sys.stderr)
        return 1

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = torch.device("xpu")
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    messages = [{"role": "user", "content": args.prompt}]

    samples = []
    for backend in backends:
        print(f"[bench] backend={backend}: loading...", file=sys.stderr)
        server_args = parse_server_args([args.model])
        engine = Engine(
            EngineConfig(
                model_path=args.model,
                tp_info=DistributedInfo(0, 1),
                dtype=dtype,
                device=device,
                attention_backend="auto",
                moe_backend=backend,
                max_running_req=1,
                page_size=1,
                max_seq_len_override=max(128, args.max_tokens + 64),
                num_page_override=512,
            )
        )
        engine.frontend_tokenizer = _frontend_tokenizer(server_args)
        prompt_ids = _prompt_token_ids(engine, _prompt_from_messages(messages), messages)

        for i in range(args.repeats):
            sample = run_client(engine, prompt_ids, backend=backend, max_tokens=args.max_tokens, uid=i)
            samples.append(sample)
            print(
                f"  repeat {i}: ttft={sample.ttft_s:.3f}s  decode_tok/s={sample.decode_tok_s:.2f}"
                f"  ({sample.decode_tokens} decode steps)",
                file=sys.stderr,
            )

        # Fully release this backend's engine (weights + host banks + XPU
        # state) before the next backend's build -- never two MoE backends'
        # state resident on the box at once.
        del engine
        reset_global_ctx()
        torch.xpu.empty_cache()
        gc.collect()

    summary = summarize(samples)
    print(format_table(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
