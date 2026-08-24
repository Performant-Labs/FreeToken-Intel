# Stack and why it was chosen

FreeToken-Intel is the [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken)
MoE serving design running on **Intel Arc Pro B70**, not a new engine and
not a llama.cpp wrapper. This page is the rationale for each layer.
The CUDA → Intel *name* map lives in [architecture.md](architecture.md).
The decision to port FreeToken rather than start over is
[ADR 0001](adr/0001-freetoken-on-intel-sycl-xpu.md).

```
  coding agents / curl / OpenAI & Anthropic SDKs
                    │
                    ▼
         ft  (Python 3.10+, FastAPI, port 1919)
                    │
     engine / scheduler / MoE policy (fused | offload | cpu | hybrid)
                    │
     ┌──────────────┼──────────────┐
     ▼              ▼              ▼
 torch.xpu     host RAM       CPU AMX / AVX-512
     │         expert banks    (miss overflow)
     ├─ Triton-Intel (default portable kernels)
     ├─ in-tree SYCL via icpx (fused / copy path)
     ├─ oneDNN / IPEX GEMM (optional)
     └─ oneCCL (multi-B70, later)
                    │
                    ▼
   Level Zero  ←  Intel Arc driver  ←  Arc Pro B70 (Xe2)
```

## Hardware: Arc Pro B70

| | Why this SKU |
| --- | --- |
| 32 GB GDDR6 | A 35B-A3B MoE fits; larger MoEs use FreeToken’s host-RAM expert cache. That is the same reason FreeToken exists on NVIDIA. |
| 608 GB/s | Well below RTX 5090-class HBM. Offload and **hybrid CPU–XPU** are the default story, not a fallback. |
| Xe2 XMX (256 engines) | Native BF16 / FP8 / MXFP4 / INT8 GEMM. NVFP4 CUDA kernels are not used. |
| ~230 W workstation card | The intended machine is a desktop or 1–4 card box, not a datacenter GPU. |

Linux x86_64 only for v1, matching upstream’s real development OS and
Intel’s oneAPI/Level Zero support matrix. Windows is out of scope until
the Linux path is real.

## Device runtime: driver, Level Zero, oneAPI

| Piece | Role | Why |
| --- | --- | --- |
| Intel Arc compute driver | Owns the device | Required for any XPU work. |
| **Level Zero** | Submit kernels, allocate USM, copy | This is Intel’s CUDA-driver analogue. SYCL, PyTorch XPU, and oneCCL all sit on it. Prefer a **single** Arc ICD; mixing Data Center GPU ICDs in containers split-brains device discovery ([intel-b70.md](intel-b70.md)). |
| **oneAPI DPC++ (`icpx -fsycl`)** | Compile in-tree kernels | Replaces `nvcc`. Same language FreeToken’s C++/CUDA kernels would be rewritten into; SPIR-V/AOT cache replaces cubins ([#3](https://github.com/Performant-Labs/FreeToken-Intel/issues/3)). |

**Rejected:** OpenCL as the primary API (Level Zero is what PyTorch XPU
and DPC++ actually use). Vulkan as the engine backend (fine for llama.cpp
bring-up, not for USM expert streaming and torch tensors).

## Framework: PyTorch `torch.xpu`

Upstream FreeToken is a PyTorch program (tensors, dtypes, sampling,
`EngineConfig.dtype`). The Intel equivalent is **PyTorch’s native XPU
backend**, not a side fork of Tensor.

| Choice | Why |
| --- | --- |
| `torch.xpu` | Drop-in for `torch.cuda` in the engine, KV pools, and Python kernels. Wheels come from the PyTorch/Intel XPU index ([install.md](install.md)). |
| **Not IPEX-LLM** | Archived January 2026. Any guide that leads with it is stale. |
| IPEX (`intel_extension_for_pytorch`) | **Optional** GEMM/oneDNN boost, probed at runtime (`is_ipex_installed()`). Not a hard dependency so CPU CI and a bare XPU wheel still import. |

**Rejected:** JAX, raw SYCL-only with no PyTorch (would throw away the
FreeToken Python engine). Building against CUDA and translating with
ZLUDA or similar (unsupported, hides Xe2 features).

## Kernels: Triton-Intel, then SYCL, then oneDNN

Upstream has three kernel tiers: pure Triton (always there), FlashInfer,
and sglang-kernel. Intel cannot use the last two.

| Tier | When we use it | Why |
| --- | --- | --- |
| **Triton with the Intel GPU backend** | Default attention and many MoE/linear kernels | Same programming model as upstream’s CUDA-Triton fallback. Fastest path to *correct* kernels we can share with later Xe SKUs. Registered attention backend: `triton` ([#4](https://github.com/Performant-Labs/FreeToken-Intel/issues/4)). |
| **In-tree SYCL** | Copies, expert admission, fused attention/MoE once Triton is the ceiling | Replaces CUDA `.cu` and FlashInfer. Needed for batched host→device copies (upstream `cudaMemcpyBatchAsync`) and a decode-fast attention path (`sycl` backend, [#5](https://github.com/Performant-Labs/FreeToken-Intel/issues/5)). |
| **oneDNN / IPEX** | GEMM where Xe2 XMX is already well-tuned | Do not hand-write every matmul. Optional; fused MoE can call it without owning the algorithm. |

AOT artifacts live in `freetoken-kernel-cache/` (SPIR-V / Triton-Intel
cache), keyed by oneAPI + driver + Xe ISA — same idea as upstream cubins.

**Rejected:** Shipping CUDA cubins. Depending on vLLM’s XPU kernels as
the only MoE path (version skew with transformers, and it does not
implement FreeToken offload).

## MoE policy: fused / offload / cpu / hybrid

These names are **unchanged** from upstream so `--moe-backend` and later
ports of FreeToken code stay familiar.

| Backend | What it is | Why it stays |
| --- | --- | --- |
| `fused` | Experts resident in XPU VRAM | Best decode when the active set fits in 32 GB. Never auto-selected for huge MoEs. |
| `offload` | Experts in host RAM, LRU slots on XPU, misses over PCIe | The original FreeToken idea. B70’s 32 GB + modest HBM make it the default for anything larger than a 35B-A3B. |
| `cpu` | Misses computed on the host | Intel desktop/workstation CPUs have AVX-512 and often AMX; host RAM bandwidth can beat a narrow PCIe link. |
| `hybrid` | Fetch some misses, compute the rest on CPU, overlap | Upstream `q*` policy. **More important on B70 than on a 5090** because 608 GB/s HBM and PCIe 4.0 x8 (OCuLink) boxes starve the XPU. Calibrate with `ft bench bw` on the actual machine. |
| `auto` | Dense → fused; MoE → offload, upgrade to hybrid from a cached bw profile | Same rule as upstream. |

CPU expert GEMM is in-tree C++ (`kernel/csrc/cpu_moe/`), not a second
Python framework.

## Quantization: MXFP4 / FP8 / BF16 / INT8 — not NVFP4

Xe2 XMX does not run NVIDIA NVFP4 (Marlin / FlashInfer) kernels.
`EngineConfig.mxfp4_backend` replaces `--nvfp4-backend`.

| Format | Role |
| --- | --- |
| BF16 | Reference and residual path |
| FP8 | Dense and some expert weights when checkpoints exist |
| MXFP4 | Primary 4-bit expert format on Xe2 (OCP MX) |
| INT8 / INT4 | Fallback / oneDNN-friendly path |

Checkpoints published as NVFP4 are either rejected or dequantized on
load until an MXFP4/INT pack exists ([#10](https://github.com/Performant-Labs/FreeToken-Intel/issues/10)).

## Memory and graphs

| Piece | Why |
| --- | --- |
| USM / pinned host tensors | Expert banks must DMA into XPU slots without bounce buffers. Replaces CUDA pinned + `cudaMemcpyBatchAsync`. |
| Paged KV + radix cache | Same agentic prefix reuse as upstream; tensors live on XPU. |
| Elastic expert-cache vs KV split | 32 GB is tight once context grows; rebuild without reloading weights. |
| SYCL graphs / Level Zero command lists (`xpu_graph_bs`) | Decode capture. If a driver cannot replay graphs, eager decode is the fallback — do not pretend CUDA graphs exist. |

## Distributed: oneCCL, not NCCL

Tensor parallel on 2–4 B70s uses **oneCCL** (`pyoneccl`). NCCL is NVIDIA
only. Multi-card is explicitly later
([#29](https://github.com/Performant-Labs/FreeToken-Intel/issues/29));
single-card correctness comes first. Tracing uses ITT/unitrace instead
of NVTX (`itt_annotate`).

## Serving and CLI

| Piece | Why keep it |
| --- | --- |
| Package name `freetoken`, CLI `ft` | Later ports of upstream code and existing agent docs keep working. |
| FastAPI + Uvicorn | Upstream server; OpenAI `/v1/chat/completions` and Anthropic `/v1/messages` on **1919**. |
| `transformers` + `safetensors` + optional GGUF | Load the same HF checkpoints FreeToken already lists. |
| `ft launch` | Wire Claude Code / Codex / OpenCode to the local server — a FreeToken product surface, not a kernel concern. |
| Python 3.10+ | Matches upstream. |
| **uv** | Matches upstream install; lockfiles are still gitignored because XPU wheels go stale. |

**Rejected as the product:** making llama.cpp or vLLM the thing users
talk to, with FreeToken as a sidecar. Those are valid *bring-up* tools
on a new B70 (see [intel-b70.md](intel-b70.md)); they are not the
serving engine this repo is for.

## First model

**Qwen3.6-35B-A3B** (and FP8/MXFP4 variants). Upstream already tunes
this MoE for laptops and desktops; 32 GB plus offload is enough; it
exercises GQA attention and routed experts without DeepSeek-V4’s sparse
stack. Other architectures are registered as stubs so later issues can
fill them without renaming the registry.

## Explicit non-goals

- NVIDIA CUDA, FlashInfer, sglang-kernel, TensorRT-LLM, NCCL, NVFP4 Marlin
- IPEX-LLM
- Vulkan or OpenVINO as the primary runtime
- Windows or macOS as a v1 target
- A different CLI, port, or MoE backend vocabulary from upstream FreeToken
