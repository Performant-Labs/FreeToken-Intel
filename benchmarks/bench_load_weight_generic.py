"""Expert bank load timing: raw safetensors vs FTW (issue `ftw-checkpoint`, #11).

Builds a small fabricated MoE checkpoint (many small per-expert tensors --
the shape that pays safetensors' per-tensor/per-shard header-parse overhead
the most), converts it to FTW, and times ``iter_safetensors`` reading each
form. Run: ``python benchmarks/bench_load_weight_generic.py [num_experts]``.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import time

import torch
from safetensors.torch import save_file

from freetoken.checkpoint.convert import convert_checkpoint
from freetoken.models.weight import iter_safetensors


def _fabricate_checkpoint(path: str, *, num_experts: int, hidden: int = 256, inter: int = 512) -> None:
    torch.manual_seed(0)
    tensors = {}
    for e in range(num_experts):
        tensors[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] = torch.randn(inter, hidden)
        tensors[f"model.layers.0.mlp.experts.{e}.up_proj.weight"] = torch.randn(inter, hidden)
        tensors[f"model.layers.0.mlp.experts.{e}.down_proj.weight"] = torch.randn(hidden, inter)
    save_file({k: v.contiguous() for k, v in tensors.items()}, f"{path}/model.safetensors")


def _time_load(model_path: str) -> float:
    t0 = time.perf_counter()
    n = 0
    for _name, tensor in iter_safetensors(model_path, torch.device("cpu")):
        n += tensor.numel()
    return time.perf_counter() - t0


def main() -> None:
    num_experts = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    tmp = tempfile.mkdtemp(prefix="ft_bench_load_weight_")
    src = f"{tmp}/safetensors_ckpt"
    dst = f"{tmp}/ftw_ckpt"
    try:
        import os

        os.makedirs(src, exist_ok=True)
        _fabricate_checkpoint(src, num_experts=num_experts)
        convert_checkpoint(src, dst)

        # One warm read each to bring both fully into the page cache, so the
        # comparison is parse/open overhead, not disk I/O noise.
        _time_load(src)
        _time_load(dst)

        st_time = min(_time_load(src) for _ in range(3))
        ftw_time = min(_time_load(dst) for _ in range(3))
        print(f"experts={num_experts}  safetensors={st_time * 1e3:.2f}ms  ftw={ftw_time * 1e3:.2f}ms  "
              f"speedup={st_time / ftw_time:.2f}x")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
