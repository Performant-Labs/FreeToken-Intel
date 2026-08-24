# benchmarks

Run from the repo root. Pin to one XPU (`ZE_AFFINITY_MASK=0` or
`ONEAPI_DEVICE_SELECTOR=level_zero:0`).

| Script | Purpose | Issue |
| --- | --- | --- |
| `bench_decode_moe.py` | bs=1 decode tok/s of a served MoE | `benchmarks` |
| `bench_load_weight_generic.py` | expert-bank load time | `ftw-checkpoint` |
| `bench_offload_cache_copy.py` | synthetic expert copy cost | `moe-offload` |

For host RAM vs PCIe vs XPU HBM, use `ft bench bw` (`moe-hybrid`).
