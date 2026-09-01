# Architecture: FreeToken → FreeToken-Intel (Arc Pro B70)

This repo is an Intel-stack equivalent of
[FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken): an
edge-native Mixture-of-Experts serving engine. The NVIDIA original targets
CUDA (RTX 30/40/50). This port targets **Intel Arc Pro B70** (Battlemage G31 /
Xe2-HPG, 32 GB GDDR6, 608 GB/s). Why each layer of that stack was chosen:
[stack.md](stack.md). Decisions: [adr/](adr/).

Most modules are **stubs**. Each stub names the GitHub issue slug that owns
the real implementation. `freetoken._stub.unimplemented()` raises
`NotYetImplemented` with that slug.

## Hardware target

| | Arc Pro B70 |
| --- | --- |
| Architecture | Xe2-HPG (Battlemage G31) |
| Xe-cores / XMX | 32 / 256 |
| VRAM | 32 GB GDDR6 (ECC capable) |
| Bandwidth | 608 GB/s |
| TBP | ~230 W |
| Software | Linux x86_64, Level Zero, oneAPI DPC++ (`icpx`), PyTorch `torch.xpu` |

608 GB/s is far below an RTX 5090. Bandwidth-adaptive **CPU–XPU co-execution**
and expert offload matter more here than on NVIDIA high-end cards.

## Package map (same shape as upstream)

```
python/freetoken/
  cli.py              ft entry (serve, shell, ctl, daemon, launch, checkpoint, bench, device)
  engine/             decode/prefill loop, SYCL-graph capture, cache budget
  scheduler/          chunked prefill, decode batching
  moe/                fused | offload | cpu | hybrid
  attention/          triton (Intel GPU) | sycl
  kvcache/            paged MHA, radix prefix cache
  kernel/             SYCL csrc, Triton-Intel, oneCCL, pinned USM
  models/             architecture registry (Qwen3.6 MoE first)
  server/             OpenAI + Anthropic HTTP
  daemon/             persistent supervisor
  checkpoint/         FTW fast-load format
```

## NVIDIA → Intel substitutions

| Upstream (CUDA) | This port (Intel) |
| --- | --- |
| `torch.cuda` | `torch.xpu` |
| `nvcc` / CUDA 13 | `icpx -fsycl` / oneAPI + Level Zero |
| FlashInfer, sglang-kernel | SYCL fused kernels + Triton-Intel + oneDNN/IPEX |
| CUDA graphs | Level Zero command lists / SYCL graphs (`xpu_graph_bs`) |
| NCCL / `pynccl` | oneCCL / `pyoneccl` |
| NVTX | ITT / unitrace (`itt_annotate`) |
| NVFP4 / Marlin | MXFP4 + INT8/INT4 on Xe2 XMX (`mxfp4_backend`) |
| `cudaMemcpyBatchAsync` | Level Zero batched copy |
| `SM90` / `SM100` probes | `is_xe2_family()` / `is_battlemage()` |
| IPEX-LLM | **not used** (archived Jan 2026) |

MoE backend names are unchanged: `fused`, `offload`, `cpu`, `hybrid`, `auto`.

Attention backends drop CUDA-only `fi` / `fa` / `trtllm`. Intel registers
`triton` and `sycl`.

## First model

`Qwen/Qwen3.6-35B-A3B` (and FP8 / MXFP4 variants that fit a 32 GB card with
offload). Other architectures are registered as stubs.

## Issue slugs

Epic: [#1 Port FreeToken to Intel Arc Pro B70](https://github.com/Performant-Labs/FreeToken-Intel/issues/1).
Stub modules mention the slug in `unimplemented(..., "<slug>")`.

| Slug | Issue |
| --- | --- |
| `device-layer` | [#2](https://github.com/Performant-Labs/FreeToken-Intel/issues/2) |
| `kernel-sycl` | [#3](https://github.com/Performant-Labs/FreeToken-Intel/issues/3) |
| `attn-triton` | [#4](https://github.com/Performant-Labs/FreeToken-Intel/issues/4) |
| `attn-sycl` | [#5](https://github.com/Performant-Labs/FreeToken-Intel/issues/5) |
| `moe-fused` | [#6](https://github.com/Performant-Labs/FreeToken-Intel/issues/6) |
| `moe-offload` | [#7](https://github.com/Performant-Labs/FreeToken-Intel/issues/7) |
| `moe-cpu` | [#8](https://github.com/Performant-Labs/FreeToken-Intel/issues/8) — done: `--moe-backend cpu` runs the routed-expert GEMM on the host from the pinned banks (pure-PyTorch single-thread `CpuMoeExecutor`, ADR 0002), and `--moe-cpu-layers` partitions MoE layers between the CPU executor and the XPU offload slot pool (`parse_moe_cpu_layers`, pure-Python). The AVX-512/AMX thread-pool GEMM (`kernel/csrc/cpu_moe/placeholder.cpp`) is a deferred kernel follow-up, not part of this slice. |
| `moe-hybrid` | [#9](https://github.com/Performant-Labs/FreeToken-Intel/issues/9) |
| `quant-xpu` | [#10](https://github.com/Performant-Labs/FreeToken-Intel/issues/10) |
| `ftw-checkpoint` | [#11](https://github.com/Performant-Labs/FreeToken-Intel/issues/11) |
| `kvcache` | [#12](https://github.com/Performant-Labs/FreeToken-Intel/issues/12) — done: paged MHA/GQA pool (`allocate`/`free` + `[L,S,H,D]` buffer) and radix prefix match/insert/evict (`_compare.py` pure-torch key fallback). `hybrid_swa_pool.py` (M5) and `scheduler/cache.py` (engine wiring, #14) remain stubs. |
| `scheduler` | [#13](https://github.com/Performant-Labs/FreeToken-Intel/issues/13) |
| `engine-loop` | [#14](https://github.com/Performant-Labs/FreeToken-Intel/issues/14) |
| `engine-graph` | [#15](https://github.com/Performant-Labs/FreeToken-Intel/issues/15) |
| `elastic-memory` | [#16](https://github.com/Performant-Labs/FreeToken-Intel/issues/16) |
| `models-loader` | [#17](https://github.com/Performant-Labs/FreeToken-Intel/issues/17) |
| `models-qwen35` | [#18](https://github.com/Performant-Labs/FreeToken-Intel/issues/18) |
| `models-qwen3-moe` | [#19](https://github.com/Performant-Labs/FreeToken-Intel/issues/19) |
| `models-dense` | [#20](https://github.com/Performant-Labs/FreeToken-Intel/issues/20) |
| `models-dsv4` | [#21](https://github.com/Performant-Labs/FreeToken-Intel/issues/21) |
| `models-glm` | [#22](https://github.com/Performant-Labs/FreeToken-Intel/issues/22) |
| `models-gpt-oss` | [#23](https://github.com/Performant-Labs/FreeToken-Intel/issues/23) |
| `layers` | [#24](https://github.com/Performant-Labs/FreeToken-Intel/issues/24) |
| `server-openai` | [#25](https://github.com/Performant-Labs/FreeToken-Intel/issues/25) |
| `server-anthropic` | [#26](https://github.com/Performant-Labs/FreeToken-Intel/issues/26) |
| `shell-daemon` | [#27](https://github.com/Performant-Labs/FreeToken-Intel/issues/27) |
| `agent-launch` | [#28](https://github.com/Performant-Labs/FreeToken-Intel/issues/28) |
| `oneccl-tp` | [#29](https://github.com/Performant-Labs/FreeToken-Intel/issues/29) |
| `benchmarks` | [#30](https://github.com/Performant-Labs/FreeToken-Intel/issues/30) |
| `ci-packaging` | [#31](https://github.com/Performant-Labs/FreeToken-Intel/issues/31) |
| `semantic-cache` | [#32](https://github.com/Performant-Labs/FreeToken-Intel/issues/32) |
