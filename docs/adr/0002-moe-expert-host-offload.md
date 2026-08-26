# ADR 0002: MoE experts live in host RAM, streamed into an XPU LRU slot pool

- Status: Accepted
- Date: 2026-08-25
- Issues: [#14](https://github.com/Performant-Labs/FreeToken-Intel/issues/14) (engine loop), [#17](https://github.com/Performant-Labs/FreeToken-Intel/issues/17) (loader), `moe-offload` / `moe-cpu` (backends, to be filed)

## Context

A Qwen3-30B-A3B MoE layer holds 128 experts; the active set per token is 8.
The *full* expert weight set is ~28 GB in bf16 and does not fit on the
B70's 32 GB alongside the KV pool, the dense weights (embed, attention,
norms, lm_head ~4 GB), and the runtime. Upstream FreeToken solves exactly
this: the entire expert set lives in **host RAM** (pinned banks) and the
GPU holds only a small fixed pool of "slots," streaming in just the experts
each token routes to via a timestamp LRU.

Our port already carries the upstream shapes. The #17 loader fabricates
per-layer banks `gate_up [E, 2I, H]` (gate+up fused on dim 1) and
`down [E, H, I]` — **bit-for-bit the upstream `bf16` bank layout**
(`offload_cache.py` `_BANK_SCHEMAS`). So the loader side needs no redesign,
only a fix to repack the checkpoint's per-expert keys into that packed form.

The open question is the *model* side: where do the expert `nn.Module`s
live during a forward pass?

## Options

1. **XPU-resident experts** — build all 128 `_Qwen3Expert` modules per layer
   on the XPU and gather over them in the forward. Simple, but ~28 GB of
   experts is VRAM we do not have; it also contradicts the loader, which
   routes expert weights to host.
2. **Host-offload + XPU LRU slot pool (upstream design)** — experts stay in
   pinned host banks; the XPU holds `S` slots plus the dense weights; each
   decode step streams only the routed (missed) experts from host into the
   pool, evicting the least-recently-used. Mirrors FreeToken module-for-module
   (per ADR 0001) and fits the 32 GB budget.

## Decision

Option 2. `_Qwen3MoE` does **not** own XPU-resident expert modules; the
forward reads the host banks through the engine's slot/LRU mapping. The
dense path (embed, attention, router, norms, lm_head) stays XPU-resident.
This replaces the interim #14 model, which built XPU-resident experts as a
correct-but-memory-hungry reference.

Concretely:
- **Loader** (`models/weight.py`): repack the checkpoint's per-expert
  `gate_proj` / `up_proj` / `down_proj` keys into the packed
  `[E, 2I, H]` / `[E, H, I]` banks (concatenate gate+up on dim 1). The dummy
  path already emits the packed form and stays unchanged.
- **Backends** (`moe/`, now stubs): port the upstream `OffloadMoeCache`
  timestamp LRU (start from the in-repo pure-torch CPU mirror of
  `lru_ensure`), `HostBanks` (pinned, shapes as above), and `copy_missing`
  (the one genuinely B70-specific piece: `cudaMemcpyBatchAsync` → oneAPI
  `queue.memcpy` / USM between pinned host and XPU). A pure-torch gather over
  the resident slots is the correct reference forward until a grouped-GEMM
  (Triton-XPU) kernel lands.
- **Prefill** materializes a whole layer (no LRU) into a double buffer;
  decode streams per-step misses. Slot pool sized `S ≥ 2E` for the prefill
  buffers plus the resident decode region.

## Consequences

- The #17 bank layout is the contract: any later quantized format (fp8 /
  mxfp4 / nvfp4) must produce the same logical `[E, 2I, H]` / `[E, H, I]`
  row shapes (packed), or the slot pool and LRU must be taught a new schema.
- The LRU bookkeeping and bank shapes port almost directly from upstream;
  the **copy engine is the real XPU work** and is the scope of the
  `moe-offload` issue.
- The #14 XPU-resident gather is kept only as a CPU-testable reference; it
  is not a serving path and must not be re-adopted for the 30B.
- Multi-card (oneCCL, ADR 0001) later splits the host banks and the slot
  pool across devices; this ADR's single-B70 layout is the `S`, `E`, `2E`
  baseline that path generalizes.
