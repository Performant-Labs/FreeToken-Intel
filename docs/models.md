# Supported models

Intel port. Known-good checkpoints will be listed here as they land on B70.

FreeToken-Intel loads HF safetensors (plus GGUF later). The first target is
Qwen3.6-35B-A3B because it is the laptop/desktop MoE that upstream FreeToken
tunes first, and 32 GB is enough with offload.

| Model | Status | Issue |
| --- | --- | --- |
| Qwen3.6 / Qwen3.5 MoE (`Qwen3_5Moe*`) | config + weights (forward pending) | `models-qwen35` |
| Qwen3-MoE | stub | `models-qwen3-moe` |
| Qwen3 / Qwen2 / Llama / Mistral / Gemma-4 dense | stub | `models-dense` |
| gpt-oss | stub | `models-gpt-oss` |
| GLM-4.7 / GLM-5.2 MoE | stub | `models-glm` |
| DeepSeek-V4 | stub | `models-dsv4` |

## MoE backends

`ft serve --moe-backend {auto,fused,offload,cpu,hybrid}`:

* **fused** — experts resident on XPU VRAM (needs the 32 GB to cover the
  active set). Never auto-selected for huge MoEs.
* **offload** — experts in host RAM, LRU expert slots on XPU; misses stream
  over PCIe.
* **cpu** — misses computed on the CPU (AVX-512 / AMX) instead of fetched.
* **hybrid** — per step, fetch some misses over PCIe and compute the rest on
  CPU, overlapped. Calibrate with `ft bench bw`.
* **auto** — dense → fused; MoE → offload, upgraded to hybrid when a cached
  bandwidth profile recommends it.

Quantization on Xe2: BF16, FP8, MXFP4, INT8/INT4. NVFP4 CUDA kernels are not
used; see `quant-xpu`.
