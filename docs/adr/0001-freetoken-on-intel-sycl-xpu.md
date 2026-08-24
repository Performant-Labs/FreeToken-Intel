# ADR 0001: Port FreeToken onto Intel SYCL / XPU for Arc Pro B70

- Status: Accepted
- Date: 2026-08-23
- Issues: [#1](https://github.com/Performant-Labs/FreeToken-Intel/issues/1)

## Context

[FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken) is an
edge-native MoE server built around CUDA (RTX 30/40/50): expert LRU cache
in host RAM, bandwidth-adaptive CPU–GPU decode, OpenAI/Anthropic APIs.
This repo exists to run that *design* on Intel Arc Pro B70 (Xe2-HPG,
32 GB, 608 GB/s), not to invent a new serving engine.

CUDA, FlashInfer, sglang-kernel, NCCL, and NVFP4 do not run on Battlemage.
IPEX-LLM was archived in January 2026. llama.cpp (Vulkan/SYCL) and vLLM
XPU already generate tokens on B70, but they do not implement FreeToken’s
MoE offload / hybrid policy.

## Options

1. **Fork FreeToken and replace the device layer** — keep `ft`, package
   layout, MoE backend names, and HTTP dialects; swap CUDA for SYCL /
   `torch.xpu` / Triton-Intel / oneCCL.
2. **Wrap llama.cpp or vLLM-XPU** — faster time-to-token, lose expert
   cache, `q*` hybrid split, FTW, and the agent-oriented KV work.
3. **New engine from scratch** — no compatibility with upstream issues,
   CLI, or later ports of FreeToken code.

## Decision

Option 1. The stack and the reasons for each layer are recorded in
[../stack.md](../stack.md). Individual replacements (for example MXFP4
vs INT4, Triton-Intel vs SYCL attention as default) get their own ADRs
when they are decided.

## Consequences

- Upstream FreeToken code can be ported module-for-module; CUDA names are
  aliased or renamed in a one-line map (`pynccl` → `pyoneccl`,
  `cuda_graph_bs` → `xpu_graph_bs`).
- We will not take IPEX-LLM, ZLUDA, or a CUDA container on Intel.
- First model and hardware: Qwen3.6-35B-A3B on a single B70; multi-card
  is oneCCL later, not NCCL.
- Stubs raise `NotYetImplemented` until the matching GitHub issue lands.
