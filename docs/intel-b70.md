# Intel Arc Pro B70 notes

## Why this card

32 GB at workstation pricing is the reason B70 is the first Intel SKU. A
35B-A3B MoE fits; larger MoEs need the same host-RAM expert cache FreeToken
was designed for.

## Stack that works on B70 today (outside this repo)

These are independent of FreeToken-Intel and are useful for bringing the
machine up:

1. **llama.cpp + Vulkan** — fastest path to a working generate, no oneAPI.
2. **llama.cpp + SYCL** (`-DGGML_SYCL=ON -DGGML_SYCL_F16=ON`) — better
   prefill, needs oneAPI.
3. **vLLM XPU containers** — possible, but watch Level Zero / OpenCL ICD
   conflicts between Arc and Data Center GPU libraries.

IPEX-LLM was archived in January 2026; do not depend on it.

## FreeToken-Intel stack (this repo)

Rationale for each layer: [stack.md](stack.md).

```
Python (ft)
  → engine / scheduler / MoE policy
    → torch.xpu
    → Triton-Intel kernels
    → in-tree SYCL (icpx)
    → oneDNN / IPEX GEMM (optional)
    → oneCCL (multi-card)
  → host RAM expert banks + CPU AMX/AVX-512 overflow
```

## Driver / runtime pitfalls

* Prefer a **single** Level Zero ICD (Arc client driver). Nested
  Data-Center ICDs inside containers can split-brain device discovery.
* Pin `ZE_AFFINITY_MASK` / `ONEAPI_DEVICE_SELECTOR` in benchmarks.
* PCIe 4.0 x8 (OCuLink) boxes have less host↔device bandwidth than x16;
  `ft bench bw` must be run on the actual machine before trusting `hybrid`.
