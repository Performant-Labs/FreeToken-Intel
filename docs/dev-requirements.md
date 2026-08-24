# Software required to develop FreeToken-Intel

Inventory of tools to install on a development machine. *Why* each
layer exists is in [stack.md](stack.md). Ordered machine setup (venvs):
[dev-setup.md](dev-setup.md). Package install: [install.md](install.md).

Two tracks:

| Track | What you can do | What you need |
| --- | --- | --- |
| **CPU** | Edit Python, run unit tests, use `ft --help` / `--version` | Linux x86_64, Git, Python 3.10+, this repo’s `[dev]` extra |
| **XPU** | Compile SYCL, run kernels, serve a model on Arc Pro B70 | Everything in CPU, plus driver, Level Zero, oneAPI DPC++, PyTorch XPU |

Windows and macOS are not v1 development targets.

## 1. Always (CPU track)

| Software | Version | Why | Check |
| --- | --- | --- | --- |
| Linux x86_64 | Ubuntu 24.04 LTS or similar | Matches Intel’s oneAPI / Level Zero matrix and upstream FreeToken | `uname -m` → `x86_64` |
| Git | any recent | Clone, PRs | `git --version` |
| Python | **≥ 3.10** (3.12 used in CI) | Runtime | `python3 --version` |
| [uv](https://docs.astral.sh/uv/) | current | Recommended installer; `pip` + venv is fine | `uv --version` |
| C compiler (`gcc`/`g++`) | system | Native extensions later; harmless on CPU-only | `g++ --version` |
| `make` | system | Build scripts | `make --version` |

Create the CPU venv and install this repo (see [dev-setup.md](dev-setup.md)):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -m "not xpu and not slow"
ft --version
```

### Python packages from this repo

`uv pip install -e ".[dev]"` pulls the `[project.dependencies]` plus
`pytest`. You do **not** install these by hand unless you are debugging
resolution:

| Package | Role |
| --- | --- |
| `fastapi`, `uvicorn`, `pydantic` | HTTP server (once `server-openai` lands) |
| `transformers`, `safetensors`, `huggingface_hub`, `gguf` | Checkpoints |
| `numpy`, `einops`, `tqdm` | Numerics / progress |
| `pyzmq`, `msgpack` | Frontend ↔ engine |
| `prompt_toolkit` | `ft shell` |
| `openai`, `partial-json-parser` | Client helpers / tool-call parse |
| `pytest` | Tests (`[dev]`) |

`torch` is **not** in the default extra. CPU tests must keep passing
without it. Install a PyTorch **XPU** wheel only on the XPU track
(below). Do not install a CUDA wheel on this machine.

## 2. XPU track (B70 kernels and serving)

Install on the machine that has the GPU. The exact, **verified** Ubuntu
26.04 install sequence (Level Zero ICD via apt → Intel APT → DPC++ 2026.1
→ `.venv-xpu` with torch 2.13.0+xpu) is in
[dev-setup.md](dev-setup.md) §3 and §6. Version pins for the in-tree
kernels still land in [#2](https://github.com/Performant-Labs/FreeToken-Intel/issues/2)
and [#3](https://github.com/Performant-Labs/FreeToken-Intel/issues/3).

> **APT gotcha:** `apt.repos.intel.com` is S3-backed with no public
> ListBucket, so *directory* URLs (`/oneapi/`, `/keyring/`, …) return
> `403 AccessDenied` while the real objects underneath are `200`. Don’t
> mistake that for a dead repo — and note the GPG key filename is
> **uppercase** `GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB`.

| Software | Why | Check |
| --- | --- | --- |
| Intel Arc **compute** driver | Device ownership for Battlemage | `lspci` shows the B70; `ft device` after PyTorch XPU |
| **Level Zero** loader + Arc ICD | Kernel submit, USM, copies. One Arc ICD only — do not mix Data Center GPU ICDs ([intel-b70.md](intel-b70.md)) | `ls /usr/lib/libze_loader.so*` or `sycl-ls` |
| **oneAPI DPC++** (`icpx`) | Compile in-tree SYCL (`icpx -fsycl`) | `icpx --version` after `source /opt/intel/oneapi/setvars.sh` |
| SYCL headers / Level Zero headers | Shipped with the Base Toolkit | `icpx -fsycl` compiles `kernel/csrc/sycl/placeholder.cpp` |
| CMake ≥ 3.18 and Ninja | Out-of-tree SYCL extension builds | `cmake --version`, `ninja --version` |
| **PyTorch XPU** wheel | `torch.xpu` — not the CUDA index | `python -c "import torch; print(torch.xpu.is_available())"` → `True` |
| **Triton** with Intel GPU backend | Default attention / MoE kernels | `python -c "import triton; print(triton.__version__)"` |

Typical oneAPI enable (path may differ):

```bash
source /opt/intel/oneapi/setvars.sh
export ONEAPI_DEVICE_SELECTOR=level_zero:0   # or ZE_AFFINITY_MASK=0
ft device
```

PyTorch XPU install is **not** `pip install torch` from PyPI’s CUDA
default. Use the XPU index documented by Intel/PyTorch at the time you
set up the box, then confirm `torch.xpu.is_available()`.

## 3. Optional (install when you hit the matching issue)

| Software | Issue | Why |
| --- | --- | --- |
| `intel-extension-for-pytorch` (IPEX) | GEMM boost | Optional; runtime probe `is_ipex_installed()`. Not required to import `freetoken`. |
| **oneCCL** | [#29](https://github.com/Performant-Labs/FreeToken-Intel/issues/29) | Multi-B70 tensor parallel. Skip on a single card. |
| Intel VTune / **unitrace** / ITT | profiling | Replaces NVTX. Only for performance work. |
| `gh` | GitHub CLI | Issues/PRs; not a build dependency. |
| `curl` | HTTP smoke | `install.md` verify recipe. |
| Hugging Face CLI / `huggingface_hub` (already a dep) | checkpoints | Needed when you load real weights (`FREETOKEN_TEST_MODEL`). |

## 4. Do not install

| Software | Why |
| --- | --- |
| **IPEX-LLM** | Archived January 2026. |
| CUDA toolkit, `nvcc`, FlashInfer, sglang-kernel, NCCL | NVIDIA stack. |
| PyTorch **CUDA** wheels on the B70 box | They shadow `torch.xpu`. |
| ZLUDA or CUDA-on-Intel shims | Unsupported; hides Xe2 features. |
| OpenVINO as the engine | Not this project’s runtime. |
| A second Level Zero ICD from Intel Data Center GPU containers | Split-brain device discovery. |

## 5. Hardware (not software, but required for XPU)

| Item | Notes |
| --- | --- |
| Intel Arc Pro B70 (or other Arc Xe2) | 32 GB GDDR6 target SKU |
| Host RAM | Expert offload banks; plan tens of GB beyond the OS for large MoEs |
| PCIe | x16 preferred; x8 (OCuLink) works but calibrate `ft bench bw` |

## 6. Quick self-check

```bash
# CPU track
python3 --version          # >= 3.10
git --version
source .venv/bin/activate
ft --version
pytest -m "not xpu and not slow"

# XPU track (in addition)
source /opt/intel/oneapi/setvars.sh
which icpx
python -c "import torch; print('xpu', torch.xpu.is_available(), torch.xpu.device_count())"
ft device                  # expect Battlemage / B70 / Xe2
```

`ft device` exiting non-zero with `torch.xpu available: False` means the
CPU track is fine and the XPU track is not ready.
