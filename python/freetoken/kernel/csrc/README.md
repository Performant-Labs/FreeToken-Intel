# SYCL / CPU kernel sources

Upstream FreeToken keeps CUDA under `python/freetoken/kernel/csrc/`. This port
uses:

| Path | Role | Replaces |
| --- | --- | --- |
| `sycl/` | DPC++ kernels (`icpx -fsycl`) | `*.cu` / `*.cuh` |
| `cpu_moe/` | AVX-512 / AMX expert GEMM | `cpu_moe/cpu_moe_ext.cpp` |
| `include/freetoken/` | Shared headers | CUDA utils / tensor.h |

Fill in under issues `kernel-sycl` and `moe-cpu`.
