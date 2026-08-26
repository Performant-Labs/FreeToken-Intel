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

## Running Qwen3-30B-A3B with host-offloaded experts (#54)

ADR 0002 moves the MoE expert weights out of XPU VRAM into pinned host RAM
plus a small LRU slot pool, so a 30B-A3B MoE runs on a B70 whose 32 GB cannot
hold every expert. The dense weights (embeddings, attention, norms, router,
`lm_head`) stay on the XPU; only the expert `gate`/`up`/`down` projections are
offloaded.

* **Checkpoint is ~61 GB.** Qwen3-30B-A3B ships as bf16 safetensors shards;
  download the full set before the first run (the loader streams the shards,
  but the offload banks are built from the expert shards on the first
  `load_model`, so a partial download stalls there). `ft serve` should surface
  the expected download / build time up front.
* **Host RAM budget.** The expert banks are pinned host RAM; the working set
  is the whole offload bank set (tens of GB), so the box needs that much free
  RAM in addition to what the OS / driver hold.
* **Divergence watchlist.** The offload forward must reproduce the in-VRAM
  forward's greedy tokens exactly (the slot pool is a *transport*, not a math
  change). Watch for:
  - **Prefill/decode mix-up.** If the prompt step is tagged `decode`, the
    model writes only the last prompt token's K/V to the pool and attends over
    the unwritten (garbage) rows — logits blow up and the output is
    non-deterministic run-to-run. `engine.step` must tag the prompt step
    `prefill`.
  - **LRU eviction of a still-needed expert.** With `S < E` slots a decode
    step can evict an expert another routed token still needs in the same
    step; the forward must re-stream it (the miss-stats counters in
    `OffloadMoeCache` should stay near zero for a warm prompt).
  - **Slot / layer-map mismatch.** A wrong `moe_layer_id` mapping reads the
    wrong layer's bank — outputs stay finite but silently wrong.

The CPU tests in `tests/test_moe_offload_forward.py` encode this contract on a
tiny fabricated Qwen3-MoE (in-VRAM reference vs. offload, plus a
determinism check), so a regression in any of the above fails the forward
gate.
