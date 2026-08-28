# Install

The full list of developer installs (CPU vs XPU, optional, do-not-install)
is [dev-requirements.md](dev-requirements.md). Machine walkthrough,
including venvs: [dev-setup.md](dev-setup.md).

## Requirements

* Linux x86_64
* Intel Arc Pro B70 (or another Arc Xe2 GPU) with a current compute driver — XPU work only
* Intel oneAPI (DPC++ `icpx` on `PATH`) and Level Zero — XPU work only
* Python >= 3.10; [uv](https://docs.astral.sh/uv/) recommended

## Method 1: editable install (this repo)

```bash
git clone https://github.com/Performant-Labs/FreeToken-Intel.git
cd FreeToken-Intel
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

Install a **PyTorch XPU** wheel into `.venv-xpu` from Intel's XPU index. The
verified pins (Jupiter, Ubuntu 26.04, oneAPI 2026.1.1):

| Piece | Pin | Why |
| --- | --- | --- |
| oneAPI DPC++ compiler | `intel-oneapi-compiler-dpcpp-cpp` **2026.1.1** | builds SYCL; `icpx` on PATH after `setvars.sh` |
| PyTorch XPU wheel | `torch==2.13.0+xpu` from `https://download.pytorch.org/whl/xpu` | the wheel that makes `torch.xpu.is_available()` True |
| Triton (Intel) | `triton==3.7.2` | Triton-Intel backend for the kernels |

```bash
python3.11 -m venv .venv-xpu && source .venv-xpu/bin/activate
pip install -U pip
pip install -e ".[dev]"
pip install torch --index-url https://download.pytorch.org/whl/xpu
```

Then verify the device layer (the goal of the `device-layer` issue):

```bash
source /opt/intel/oneapi/setvars.sh
ft --version
ft device
# expect: torch.xpu available: True, device 0 name: Intel(R) Graphics,
#         xe2/battlemage: True, Level Zero driver: 1.14, VRAM: 32 GB
```

`ft device` now reads the **Level Zero driver** version (the Intel equivalent of
upstream's CUDA UMD version) and the GPU's **total VRAM** straight from the
device, rather than printing a fixed spec. On a CPU-only box the same command
still runs and reports `torch.xpu available: False` with a `Level Zero driver:
(not exposed)` line — it never crashes.

## Method 2: from PyPI

Not published yet. Track the packaging issue on the epic.

## Verify (once serve is implemented)

```bash
ft serve --model ~/path/to/Qwen3.6-35B-A3B
curl http://127.0.0.1:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B","messages":[{"role":"user","content":"hi"}]}'
```
