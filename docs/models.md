# Supported models

Intel port. FreeToken-Intel loads HF safetensors and GGUF checkpoints
(architecture-name lookup in `models/register.py` picks the model class
either way). The hero model is Qwen3.5/3.6-35B-A3B because it is the
laptop/desktop MoE upstream FreeToken tunes first, and 32 GB is enough with
offload.

**Status** below is derived mechanically: *running* means the registered
class's `forward` does not raise `NotYetImplemented`; *stub* means it always
does. There is no in-between case in this tree today — a package is either a
real forward or a 20-line placeholder — so nothing here is marked *partial*.
"Running" is not the same as "validated against a real checkpoint": see the
Qwen3.5/3.6 row.

| Model | Registered as (`models/register.py`) | Status | Issue |
| --- | --- | --- | --- |
| Qwen3.5 / Qwen3.6 hybrid-attention MoE | `Qwen3_5MoeForConditionalGeneration`, `Qwen3_5ForConditionalGeneration` | running — config + weights + forward (linear attention + full attention + MoE, offload/cpu/hybrid). **Validated on a real GGUF checkpoint** (`Qwen3.6-35B-A3B` `q4_k_m`, live on B70, coherent generation, [#274](https://github.com/Performant-Labs/FreeToken-Intel/issues/274) closed) via `offload`/`cpu`. Known real gaps: `hybrid` is not yet available for GGUF ([#290](https://github.com/Performant-Labs/FreeToken-Intel/issues/290), [#302](https://github.com/Performant-Labs/FreeToken-Intel/issues/302)); real-checkpoint decode is slow and unprofiled at 0.68 tok/s ([#303](https://github.com/Performant-Labs/FreeToken-Intel/issues/303)); context is capped well below the checkpoint's native 262K tokens to avoid a VRAM OOM ([#304](https://github.com/Performant-Labs/FreeToken-Intel/issues/304)); no concurrent-request testing yet ([#305](https://github.com/Performant-Labs/FreeToken-Intel/issues/305)). | `models-qwen35` (#18) |
| Qwen3-MoE | `Qwen3MoeForCausalLM` | running | `models-qwen3-moe` (#19) |
| Qwen1.5-MoE-A2.7B | `Qwen2MoeForCausalLM` | running | `models-qwen2moe-attn` (#221) |
| Qwen3.8-Flash-Next (hyper-connections + PLE + QSA) | `Qwen4ExpForCausalLM`, `Qwen4ExpForConditionalGeneration` | running | epic #198 (`models-qwen4-e2e` #209) |
| GLM-4.7 MoE | `Glm4MoeForCausalLM` | running | `models-glm` (#22) |
| GLM-5.2 (DSA) MoE | `GlmMoeDsaForCausalLM` | running | `models-glm` (#22) |
| gpt-oss | `GptOssForCausalLM` | running | `models-gpt-oss` (#23) |
| DeepSeek-V4 | `DeepseekV4ForCausalLM` | running | `models-dsv4` (#21) |
| DeepSeek-Coder-V2-Lite | `DeepseekV2ForCausalLM` | running | epic #216 (`models-dsv2lite-mla` #217, `models-dsv2lite-moe` #218) |
| OLMoE-1B-7B | `OlmoeForCausalLM` | running | epic #223 (`models-olmoe-attn` #224) |
| Mellum2-12B-A2.5B | `MellumForCausalLM` | running | `models-mellum-attn` (#227) |
| LFM2(.5)-8B-A1B | `Lfm2MoeForCausalLM` | running | epic #229 |
| Qwen3 dense | `Qwen3ForCausalLM` | running | `models-dense` (#20) |
| Qwen2, Llama, Mistral, Gemma-4, MiniMax-M2, MiniMax-M3, MuseGlimmer dense/sparse | `Qwen2ForCausalLM`, `LlamaForCausalLM`, `MistralForCausalLM`, `Gemma4ForCausalLM`, `Gemma4ForConditionalGeneration`, `MiniMaxM2ForCausalLM`, `MiniMaxM3SparseForCausalLM`, `MuseGlimmerForConditionalGeneration` | stub — `forward` raises `NotYetImplemented` | `models-dense` (#20) |

## MoE backends

`ft serve --moe-backend {auto,cpu,offload,hybrid}` (`--moe-strategy` is an
accepted alias, issue #289). **`fused`** is not a CLI choice: it is an
internal `auto`-resolution outcome for a dense model with an XPU available
(`resolve_moe_backend`), and every dense architecture above is currently a
stub, so no served model resolves to it today.

* **offload** — experts in host RAM, LRU expert slots on XPU; misses stream
  over PCIe. The default `auto` outcome for a MoE.
* **cpu** — misses computed on the CPU via the native thread-pool GEMM;
  `--moe-cpu-threads` reaches it
  ([#291](https://github.com/Performant-Labs/FreeToken-Intel/issues/291), merged).
* **hybrid** — per step, fetch some misses over PCIe and compute the rest on
  CPU, overlapped (native GIL-free pool, bf16 only). Calibrate with
  `ft bench bw`. On GPTQ/FP8/MXFP4/INT8/GGUF checkpoints `hybrid` currently
  silently runs as `offload` instead of erroring
  ([#290](https://github.com/Performant-Labs/FreeToken-Intel/issues/290));
  making it actually work for GGUF (not just fail loudly) is
  [#302](https://github.com/Performant-Labs/FreeToken-Intel/issues/302).
* **auto** — dense → fused (moot while dense models are stubs); MoE →
  offload, upgraded to hybrid when a cached `ft bench bw` profile
  recommends it.

Quantization on Xe2: BF16, FP8, MXFP4, INT8/INT4, and GGUF K-quant. NVFP4
CUDA kernels are not used; see `quant-xpu`.
