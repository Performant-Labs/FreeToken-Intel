"""``ft serve`` arguments.

Upstream NVIDIA path: python/freetoken/server/args.py

This is the Intel port's serving-args surface. It deliberately carries only
what ``ft serve`` needs to stand up an OpenAI-compatible HTTP endpoint on the
B70; the full upstream argument set (MoE cache sizing, tokenizer process pool,
KV capacity overrides, …) lands together with the engine and scheduler epics
(#14, #13) that actually consume those knobs. Adding a knob later is a one-line
argparse addition — the ``ServerArgs`` dataclass is the single source of
defaults.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, fields

# Upstream serves on 127.0.0.1:1919 so Codex and friends "just work".
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 1919

DTYPE_CHOICES = ("auto", "float16", "bfloat16", "float32")
TOOL_CALL_PARSER_CHOICES = (
    "auto",
    "llama3",
    "qwen25",
    "qwen3_coder",
    "mistral",
    "deepseekv32",
    "gemma4",
    "glm47",
    "minimax",
    "gpt_oss",
)
REASONING_PARSER_CHOICES = ("auto", "off", "deepseekv32", "gpt_oss", "qwen3", "glm", "minimax", "gemma4")


@dataclass(frozen=True)
class ServerArgs:
    """Serving configuration for ``ft serve``.

    ``model_path`` accepts a Hugging Face repo id or a local path (the loader,
    issue #17, resolves it). ``served_model_name`` is the id reported by
    ``/v1/models`` and defaults to the model reference's basename.
    """

    model: str
    server_host: str = DEFAULT_HOST
    server_port: int = DEFAULT_PORT
    dtype: str = "auto"
    served_model_name: str | None = None
    tool_call_parser: str = "auto"
    reasoning_parser: str | None = "auto"
    max_output_tokens: int | None = None
    shell_mode: bool = False
    # MoE backend selection (issue #8 / ADR 0002). "auto" is the default: the
    # engine resolves it (host-RAM offload on a B70 MoE). Explicit "cpu" runs the
    # routed-expert GEMM on the host CPU instead of streaming experts to the XPU;
    # "offload" is the in-VRAM / host-pinned-streaming default; "hybrid" (#9) will
    # split hot experts to the XPU and the tail to the CPU.
    moe_backend: str = "auto"
    # Host CPU threads for the CPU MoE GEMM (issue #8); 0 = torch default.
    moe_cpu_threads: int = 0
    # Comma-separated MoE layer indices to run on the CPU (issue #8); None = all
    # MoE layers when the backend is cpu/hybrid.
    moe_cpu_layers: str | None = None
    # Issue #9 (moe-hybrid): cap on the per-step PCIe-fetched expert count. -1 (the
    # default) = fully profile-driven (the fetch fraction from `ft bench bw`); a
    # non-negative int caps the number of routed experts fetched to the XPU each
    # decode step (the rest compute on the host CPU). Threads into EngineConfig.
    moe_hybrid_max_fetch: int = -1
    # Issue #16 (elastic-memory): how to size the MoE expert slot cache.
    # "auto" (default) plans the split off the device's total VRAM
    # (memory_ratio + kv_reserve_tokens, MoE-priority / KV-floor); a positive
    # integer pins the slot count (no planning, no VRAM read); the rate is an
    # optional fraction-of-VRAM override when auto. None = use auto defaults.
    moe_cache_auto: bool = True
    moe_cache_size: int | None = None
    moe_cache_rate: float | None = None
    # Fraction of total VRAM treated as the addressable budget (0,1].
    memory_ratio: float = 0.9
    # KV pool floor, in tokens, reserved for long-context scheduling.
    kv_reserve_tokens: int = 8192

    def __post_init__(self) -> None:
        if self.server_port < 0 or self.server_port > 65535:
            raise ValueError(f"server_port must be in [0, 65535], got {self.server_port}")

    @property
    def model_path(self) -> str:
        """Alias for :attr:`model` — the loader (#17) keys on ``model_path``."""
        return self.model

    @property
    def resolved_model_name(self) -> str:
        if self.served_model_name:
            return self.served_model_name
        return os.path.basename(os.path.normpath(self.model)) or self.model


def _infer_tool_call_parser(args: "ServerArgs") -> str:
    """Best-effort per-family parser selection (upstream parity, no HF round-trip).

    The real parser implementations land with the generation path (#14/#25
    streaming); until then this just records which grammar the serving layer
    should parse, inferred from the model reference.
    """
    marker = os.path.basename(args.model).lower()
    if "gpt" in marker and "oss" in marker:
        return "gpt_oss"
    if "qwen3" in marker and "coder" in marker:
        return "qwen3_coder"
    if "qwen" in marker:
        return "qwen25"
    if "deepseek" in marker:
        return "deepseekv32"
    if "gemma" in marker:
        return "gemma4"
    if "glm" in marker:
        return "glm47"
    if "minimax" in marker:
        return "minimax"
    if "mistral" in marker:
        return "mistral"
    return "llama3"


def _infer_reasoning_parser(args: "ServerArgs") -> str | None:
    marker = os.path.basename(args.model).lower()
    if "gpt" in marker and "oss" in marker:
        return "gpt_oss"
    if "qwen3" in marker:
        return "qwen3"
    if "glm" in marker:
        return "glm"
    if "minimax" in marker:
        return "minimax"
    if "gemma" in marker:
        return "gemma4"
    return None


def parse_args(args: list[str] | None = None, prog: str | None = None) -> ServerArgs:
    """Parse ``ft serve`` arguments.

    ``args`` is the sub-command argv *after* ``serve`` (i.e. what the CLI
    dispatcher hands over); ``None`` falls back to ``sys.argv[1:]`` for direct
    use. Unknown flags are a usage error (argparse exits 2).
    """
    if args is None:
        import sys

        args = sys.argv[1:]

    parser = argparse.ArgumentParser(prog=prog or "ft serve", description="FreeToken-Intel server")
    parser.add_argument(
        "model",
        help="Model reference: HF repo id or local path (resolved by the loader, issue #17).",
    )
    parser.add_argument("--host", dest="server_host", default=DEFAULT_HOST, help=f"Bind host (default: {DEFAULT_HOST})")
    parser.add_argument("--port", dest="server_port", type=int, default=DEFAULT_PORT, help=f"Bind port (default: {DEFAULT_PORT})")
    parser.add_argument("--dtype", default="auto", choices=DTYPE_CHOICES, help="Weight dtype; 'auto' follows the checkpoint.")
    parser.add_argument("--served-model-name", dest="served_model_name", default=None, help="Model id reported by /v1/models (default: model basename).")
    parser.add_argument("--tool-call-parser", dest="tool_call_parser", default="auto", choices=TOOL_CALL_PARSER_CHOICES, help="Tool-call grammar for OpenAI tool responses.")
    parser.add_argument("--reasoning-parser", dest="reasoning_parser", default="auto", choices=REASONING_PARSER_CHOICES, help="Split chain-of-thought into reasoning_content. 'off' disables.")
    parser.add_argument("--max-output-tokens", dest="max_output_tokens", type=int, default=None, help="Default max decode tokens for requests that omit one.")
    parser.add_argument("--shell-mode", dest="shell_mode", action="store_true", help="Run the server attached to a terminal shell (ft shell).")
    parser.add_argument(
        "--moe-backend",
        dest="moe_backend",
        default="auto",
        choices=("auto", "cpu", "offload", "hybrid"),
        help="MoE backend. 'auto' (default) picks host-RAM offload on a B70 MoE; "
        "'cpu' runs routed-expert GEMMs on the host CPU (issue #8); 'offload' streams "
        "activated experts to the XPU; 'hybrid' (issue #9) splits hot experts to XPU "
        "and the tail to CPU.",
    )
    parser.add_argument(
        "--moe-cpu-threads",
        dest="moe_cpu_threads",
        type=int,
        default=0,
        help="Host CPU threads for the CPU MoE GEMM (issue #8). 0 = torch default.",
    )
    parser.add_argument(
        "--moe-cpu-layers",
        dest="moe_cpu_layers",
        default=None,
        help="Comma-separated MoE layer indices to run on the CPU (issue #8). "
        "Empty/omitted = all MoE layers when the backend is cpu/hybrid.",
    )
    parser.add_argument(
        "--moe-hybrid-max-fetch",
        dest="moe_hybrid_max_fetch",
        type=int,
        default=-1,
        help="Issue #9 (moe-hybrid): cap on the per-step PCIe-fetched expert count. "
        "-1 (default) = fully profile-driven via the `ft bench bw` fetch fraction; "
        "a non-negative int caps the routed experts fetched to the XPU each decode "
        "step (the rest compute on the host CPU).",
    )
    parser.add_argument(
        "--moe-cache-auto",
        dest="moe_cache_auto",
        default=True,
        action="store_true",
        help="Issue #16 (elastic-memory): plan the MoE expert-cache / KV split from "
        "the device's total VRAM (default). Pass --moe-cache-size to pin the slot "
        "count instead, or combine --moe-cache-rate with auto to steer the fraction.",
    )
    parser.add_argument(
        "--moe-cache-no-auto",
        dest="moe_cache_auto",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--moe-cache-size",
        dest="moe_cache_size",
        type=int,
        default=None,
        help="Pin the MoE expert slot-cache size (slots). When set, the cache is this "
        "size and the KV pool takes the remaining VRAM (no auto planning). "
        "Default: plan from VRAM.",
    )
    parser.add_argument(
        "--moe-cache-rate",
        dest="moe_cache_rate",
        type=float,
        default=None,
        help="When --moe-cache-auto, fraction of the addressable VRAM the MoE cache "
        "should take (the rest goes to KV). Omit to use the MoE-priority policy.",
    )
    parser.add_argument(
        "--memory-ratio",
        dest="memory_ratio",
        type=float,
        default=0.9,
        help="Fraction of total VRAM treated as the addressable budget (0,1] "
        "(issue #16). Headroom outside this is reserved for the OS / runtime.",
    )
    parser.add_argument(
        "--kv-reserve-tokens",
        dest="kv_reserve_tokens",
        type=int,
        default=8192,
        help="KV pool floor in tokens, always reserved for long-context scheduling "
        "(issue #16). The MoE cache is sized from the VRAM left after this floor.",
    )

    ns = parser.parse_args(args)
    kwargs = {f.name: getattr(ns, f.name) for f in fields(ServerArgs)}
    args_obj = ServerArgs(**kwargs)

    if args_obj.tool_call_parser == "auto":
        args_obj = _replace_parser(args_obj, tool_call_parser=_infer_tool_call_parser(args_obj))
    if args_obj.reasoning_parser == "auto":
        args_obj = _replace_parser(args_obj, reasoning_parser=_infer_reasoning_parser(args_obj))
    elif args_obj.reasoning_parser == "off":
        args_obj = _replace_parser(args_obj, reasoning_parser=None)
    return args_obj


def _replace_parser(args: ServerArgs, **changes) -> ServerArgs:
    current = {f.name: getattr(args, f.name) for f in fields(ServerArgs)}
    current.update(changes)
    return ServerArgs(**current)
