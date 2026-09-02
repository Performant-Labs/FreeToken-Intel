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

## Method 2: from a wheel (or PyPI)

The core package is **pure Python and torch-free** — installing it needs no GPU
and no oneAPI. The Intel XPU stack (oneAPI, Level Zero, the `torch` XPU wheel,
Triton, TBB, etc.) ships as the **optional `[xpu]` extra** (25 pinned
packages; `[accel]` is an alias that pulls `[xpu]`). Core deps therefore never
change between a CPU box and an XPU box, and the dual-venv contract from Method
1 is preserved at the packaging layer: a CPU consumer installs the base package
and never sees torch; an XPU consumer adds the extra.

Install on an **XPU box** (oneAPI + Level Zero + the `icpx` compiler must
already be present per the [Requirements](#requirements) above):

The `[xpu]` extra pins each package to the **concrete version** the XPU nightly
in [ci.md](ci.md) verifies: `torch==2.13.0+xpu` plus the oneAPI runtimes
(oneMKL-SYCL, TBB, UMF, pyzes, tcmlib, triton-xpu, all `2026.0.0`). Note the two
different source mechanisms:

* `torch` is a **pip index** install — the `+xpu` wheel lives only on Intel's
  PyTorch XPU index, so pip needs that index URL (same as Method 1).
* the oneAPI runtimes are **`apt`/repo** packages on Intel's oneAPI repo,
  installed with the oneAPI toolchain (see [dev-setup.md](dev-setup.md)) —
  they are not pip artifacts. So on a fully set-up oneAPI box, `pip install
  "freetoken-intel[xpu]"` resolves the `torch` pin from the XPU index and the
  oneAPI pins from the repo already on the box's index path.

```bash
python3.11 -m venv .venv-xpu && source .venv-xpu/bin/activate
pip install -U pip
# torch==2.13.0+xpu from the PyTorch XPU index; the oneAPI pins resolve from
# the oneAPI repo this box is already configured with (dev-setup.md).
pip install "freetoken-intel[xpu]" \
  --index-url https://download.pytorch.org/whl/xpu
```

On a **CPU-only box**, install the base package with **no extra**:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install "freetoken-intel"        # no [xpu] -- torch-free by construction
```

### Where the wheels come from

CI builds the CPU sdist + wheel on **every push/PR** (the `build-wheel` job in
[ci.md](ci.md)) and uploads it as a run artifact (`freetoken-intel-wheel`).
That is for verification — it is a CPU wheel, and it is **not** a publish.
Publishing to PyPI is a **manual, key-gated release step** (see the packaging
issue on the epic), so "pip install freetoken-intel[xpu]" from the public index
is not available until the first release is cut. Until then, use Method 1
(editable) on a dev box, or build a wheel locally with
`scripts/build-release-wheels.sh` (CPU) / `... xpu` (on a oneAPI box) and
`pip install dist/*.whl`.

## Verify (once serve is implemented)

```bash
ft serve --model ~/path/to/Qwen3.6-35B-A3B
curl http://127.0.0.1:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B","messages":[{"role":"user","content":"hi"}]}'
```
