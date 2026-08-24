# Set up a machine for development

Walkthrough for a Linux x86_64 box. Software *names* are in
[dev-requirements.md](dev-requirements.md). This page is the order of
operations. After setup, run the agent prompt
[tools/preflight-check.md](../tools/preflight-check.md) to verify the
machine.

**Put every Python package in a venv.** Do not `pip install` FreeToken,
PyTorch, Triton, IPEX, or pytest into the system interpreter. Drivers,
Level Zero, and `icpx` cannot live in a venv — those stay on the OS.

Two venvs are enough:

| Venv | Path | Use |
| --- | --- | --- |
| CPU | `.venv` | Edit Python, `pytest`, `ft --help` / `--version`. No PyTorch. |
| XPU | `.venv-xpu` | Kernels, `torch.xpu`, `ft device`, later `ft serve`. |

Keep them separate so a CUDA or CPU `torch` wheel cannot shadow the XPU
build, and so CI-equivalent tests stay runnable without a GPU.

## 0. What goes where

| Lives on the **system** | Lives in a **venv** |
| --- | --- |
| Kernel, Intel Arc compute driver | This repo (`pip install -e ".[dev]"`) |
| Level Zero loader + Arc ICD | PyTorch XPU, Triton-Intel |
| oneAPI DPC++ (`icpx`), SYCL headers | Optional IPEX, oneCCL Python bindings |
| `gcc`/`g++`, CMake, Ninja, Git, Python 3.10+ | pytest and the rest of `[dev]` |

## 1. Operating system

Use Ubuntu 24.04 LTS (or another current x86_64 Linux Intel documents
for Arc + oneAPI). Confirm:

```bash
uname -m    # must be x86_64
```

Windows and macOS are not v1 targets.

## 2. System packages

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  git \
  python3 \
  python3-venv \
  python3-pip \
  cmake \
  ninja-build \
  curl \
  pkg-config
python3 --version    # 3.10 or newer (CI uses 3.12)
```

`python3-venv` is required for `python3 -m venv`. Without it Ubuntu
raises `ensurepip is not available`.

Optional: [uv](https://docs.astral.sh/uv/) if you prefer `uv venv` /
`uv pip` to `python3 -m venv` / `pip`. Both create the same kind of
venv; the rest of this page uses the stdlib so it works with only apt.

## 3. GPU driver and oneAPI (XPU machines only)

Skip this section on a CPU-only laptop. You can still develop Python
and run unit tests in `.venv`.

These installs are **system-wide**. They will not fit in a venv.

1. Install Intel’s **Arc compute driver** for Battlemage (B70) from
   Intel’s current Linux graphics/compute guide for your distro.
2. Confirm the card: `lspci | grep -i -E 'VGA|Display'` should mention
   Arc / B70 / Battlemage.
3. Install **oneAPI Base Toolkit** so `icpx` and Level Zero headers
   exist. Typical layout: `/opt/intel/oneapi/`.
4. Use **one** Level Zero ICD (the Arc client). Do not also load Intel
   Data Center GPU ICDs from containers — see [intel-b70.md](intel-b70.md).

After install, in a **new** shell:

```bash
source /opt/intel/oneapi/setvars.sh    # path may differ
which icpx && icpx --version
sycl-ls                                # should list a Level Zero GPU
```

You will `source setvars.sh` in every shell that compiles SYCL or talks
to the XPU. It does not replace activating a venv; do both.

## 4. Clone

```bash
git clone https://github.com/Performant-Labs/FreeToken-Intel.git
cd FreeToken-Intel
```

`.venv` and `.venv-xpu` are gitignored.

## 5. CPU venv (`.venv`)

Use this for everyday Python work. It must **not** contain `torch`.

```bash
cd /path/to/FreeToken-Intel
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

`python3 -m venv` already copies pip into `.venv`. Always invoke
`python` / `pip` after `source .venv/bin/activate` so you do not hit
`/usr/bin/python3`.

Check:

```bash
which python                 # .../FreeToken-Intel/.venv/bin/python
python -c "import freetoken; print(freetoken.__version__)"
ft --version
pytest -m "not xpu and not slow"
deactivate
```

Equivalent with uv:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## 6. XPU venv (`.venv-xpu`)

Second venv, same repo, **plus** PyTorch built for XPU. Create it only
on a machine that completed step 3.

```bash
cd /path/to/FreeToken-Intel
python3 -m venv .venv-xpu
source .venv-xpu/bin/activate
source /opt/intel/oneapi/setvars.sh
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Install **PyTorch XPU** into this venv only. Do not use the default
PyPI CUDA wheel.

```bash
# Index URL follows Intel/PyTorch's current XPU instructions.
# Confirm on https://pytorch.org (XPU / Intel GPU) before pinning.
python -m pip install torch --index-url https://download.pytorch.org/whl/xpu
```

If Triton does not come in with that wheel, install the Intel GPU
Triton build into the **same** venv (not the CPU venv). IPEX, when you
want it, is also `pip install` into `.venv-xpu` — never into `.venv`.

Check (venv activated **and** `setvars.sh` sourced):

```bash
which python                 # .../.venv-xpu/bin/python
python -c "import torch; print(torch.__version__, torch.xpu.is_available())"
ft device                    # expect Battlemage / B70; exit 0
```

If `torch.xpu.is_available()` is false: driver/ICD/oneAPI, not pip.
If `import torch` fails or you got a `+cu` build: wrong index; recreate
the venv rather than stacking wheels.

`uv venv .venv-xpu` then `uv pip install …` is equivalent.

## 7. Daily workflow

CPU tests and docs/Python-only issues:

```bash
cd /path/to/FreeToken-Intel
source .venv/bin/activate
pytest -m "not xpu and not slow"
ft --help
```

Kernel / device / serve work:

```bash
cd /path/to/FreeToken-Intel
source .venv-xpu/bin/activate
source /opt/intel/oneapi/setvars.sh
export ONEAPI_DEVICE_SELECTOR=level_zero:0   # optional; pin one card
ft device
```

One shell, one venv. `deactivate` before switching. Do not
`source .venv/bin/activate` on top of `.venv-xpu` (PATH will mix).

Optional `~/.bashrc` helpers (edit the repo path):

```bash
fti-cpu() {
  source /path/to/FreeToken-Intel/.venv/bin/activate
}
fti-xpu() {
  source /path/to/FreeToken-Intel/.venv-xpu/bin/activate
  source /opt/intel/oneapi/setvars.sh
  export ONEAPI_DEVICE_SELECTOR=level_zero:0
}
```

## 8. Recreate a venv

If a wheel is wrong or the venv is stale:

```bash
deactivate 2>/dev/null || true
rm -rf .venv          # or .venv-xpu
# then repeat section 5 or 6
```

Faster than trying to uninstall a CUDA `torch` and replace it in place.

## 9. What not to do

- `sudo pip install …` or `pip install --user` for this project
- One venv with both CUDA and XPU PyTorch
- Putting `icpx`, the driver, or `libze_loader` inside the venv
- Installing IPEX-LLM (archived) or FlashInfer / sglang-kernel
- Running `ft device` in `.venv` and treating a missing XPU as a driver
  bug — that venv is not supposed to have `torch`
