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
python3 --version    # 3.10 or newer (CI uses 3.11)
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

1. Add the user to the `render` and `video` groups, then **re-login**:

   ```bash
   sudo usermod -aG render,video "$USER"   # then log out and back in
   ```

2. Install the **Level Zero loader + Intel Arc ICD** from Ubuntu (the
   Battlemage ICD ships as `libze-intel-gpu*`):

   ```bash
   sudo apt update
   sudo apt install -y libze1 libze-dev libze-intel-gpu1
   ```

   This puts `libze_loader.so.1` and `libze_intel_gpu.so` on the system.
   The OpenCL `intel-opencl-icd` may also be present from the graphics
   driver — that is fine and expected.

3. Add **Intel’s APT repo** and install the **DPC++/C++ compiler**.
   Verified on **Ubuntu 26.04 (resolute)**:

   ```bash
   # Intel GPG key. NOTE: the filename is UPPERCASE .PUB — the lowercase
   # .pub 404/403s. The APT server is S3-backed with no public ListBucket,
   # so *directory* paths (e.g. /oneapi/, /keyring/) return 403
   # "AccessDenied" even though the real objects under them are 200.
   # Don't read that 403 as "repo is down" — test real file paths.
   wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
     | sudo gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg

   echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
     | sudo tee /etc/apt/sources.list.d/oneAPI.list
   sudo apt update

   # The compiler only (what FreeToken-Intel needs to build SYCL).
   sudo apt install -y intel-oneapi-compiler-dpcpp-cpp
   ```

   This lands the real binary at `/opt/intel/oneapi/compiler/2026.1/bin/icpx`
   and the official env script at `/opt/intel/oneapi/setvars.sh`.
   (If you also want the debugger, TBB, oneMKL, etc., install the
   `intel-oneapi-base-toolkit` metapackage instead.)

   The repo carries every version from 2023.2.2 through the current
   2026.1.1; apt enforces the signed Release/Packages metadata, so there
   is no separate artifact hash to record.

4. Use **one** Level Zero ICD (the Arc client). Do not also load Intel
   Data Center GPU ICDs from containers — see [intel-b70.md](intel-b70.md).

After install, in a **new** shell:

```bash
source /opt/intel/oneapi/setvars.sh --force
which icpx && icpx --version
sycl-ls                                  # should list a Level Zero GPU
```

You will `source setvars.sh` in every shell that compiles SYCL or talks
to the XPU. It does not replace activating a venv; do both.

**Heads-up (this project):** the repo also ships
[setvars-xpu.sh](../setvars-xpu.sh), which sets
`ONEAPI_DEVICE_SELECTOR=level_zero:0` and then delegates to the official
`setvars.sh`. Use that when working in `.venv-xpu` so the B70 is pinned
if an AMD iGPU Level-Zero ICD ever shadows it (on the reference machine
`torch.xpu.device_count` is 1, so the pin is currently a no-op guard).

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
python3.11 -m venv .venv-xpu      # match the interpreter to the wheel tag
source .venv-xpu/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Install **PyTorch XPU** into this venv only. Do **not** use the default
PyPI CUDA wheel, and install it with `--no-deps` (see below). Verified on
Ubuntu 26.04 with Python 3.11:

```bash
python -m pip install --no-deps torch==2.13.0 \
  --index-url https://download.pytorch.org/whl/xpu
```

**Why `--no-deps`, and the follow-ups.** The XPU wheel's metadata pulls
`nvidia-*` CUDA packages on a plain resolve, which must not land on a B70
box. Install the Intel-side deps explicitly instead:

```bash
# Intel runtimes/oneCCL/Triton-XPU come from the same XPU index.
python -m pip install \
  oneccl==2022.0.0 oneccl-devel==2022.0.0 tbb==2023.0.0 \
  umf==1.1.0 pyzes==0.1.1 tcmlib==1.5.0 triton-xpu==3.7.2 \
  --index-url https://download.pytorch.org/whl/xpu

# torch also wants sympy>=1.13.3; the [dev] resolve may leave an older one.
python -m pip install "sympy>=1.13.3"

# oneMKL + a couple of torch deps resolve from PyPI.
python -m pip install \
  onemkl-license==2026.0.0 onemkl-sycl-blas==2026.0.0 \
  onemkl-sycl-dft==2026.0.0 onemkl-sycl-lapack==2026.0.0 \
  onemkl-sycl-rng==2026.0.0 onemkl-sycl-sparse==2026.0.0 \
  jinja2 networkx

# Pin the Intel runtimes back to the 2026.0.x set torch 2.13.0+xpu wants.
# (The onemkl step above drags intel-*-rt up to 2026.1.x, which then makes
#  `pip check` complain that torch requires the 2026.0.x pins.)
python -m pip install --force-reinstall --no-deps \
  intel-cmplr-lib-rt==2026.0.0 intel-cmplr-lib-ur==2026.0.0 \
  intel-cmplr-lic-rt==2026.0.0 intel-sycl-rt==2026.0.0 \
  intel-opencl-rt==2026.0.0 intel-openmp==2026.0.0 \
  dpcpp-cpp-rt==2026.0.0 intel-pti==0.17.0
```

Confirm with `python -m pip check` (should report no broken requirements).
IPEX, when you want it, is also `pip install` into `.venv-xpu` — never
into `.venv`. (IPEX is out of scope for this project; skip unless needed.)

**One subtlety:** `sycl-ls` only discovers the B70 once the `.venv-xpu`
runtime packages are on `PATH`/`LD_LIBRARY_PATH`. That is expected — the
pip-shipped `intel-*-rt` bundles the Level-Zero runtime the loader binds
to. In a bare system shell (no venv) `sycl-ls` can report "No platforms
found" even though the GPU is fine; always run it with `.venv-xpu` active.

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

## 8. Local validation

Before you push, run the local mirror of the CI gates:

```bash
cd /path/to/FreeToken-Intel
bash tools/validate.sh            # or: bash tools/validate.sh origin/main
```

It runs the same commands the CI jobs run (workflow lint, secret scan, the CPU
venv contract, the CLI smoke test, the CPU test suite) and prints one
`PASS`/`FAIL`/`SKIP` line per gate, then a verdict. It exits non-zero if any
gate `FAIL`s, so it is a useful pre-push gate.

- A `SKIP` is **not** a failure: a gate that cannot run on this box (no local
  `actionlint`, or a check that needs the XPU fleet) skips rather than failing
  and still runs in CI.
- The `ci:*` gates run inside `.venv`, so create it (section 5) first; without
  it those gates `SKIP`. `gitleaks` and `actionlint` are optional local
  installs — install either to un-skip its gate.
- Optional `base-ref` argument sets the secret-scan range (default: the PR base
  when on a PR branch, else the tip commit).

`ci.md` (the CI walkthrough) documents each gate in detail; this command is the
local stand-in.

## 9. Recreate a venv

If a wheel is wrong or the venv is stale:

```bash
deactivate 2>/dev/null || true
rm -rf .venv          # or .venv-xpu
# then repeat section 5 or 6
```

Faster than trying to uninstall a CUDA `torch` and replace it in place.

## 10. What not to do

- `sudo pip install …` or `pip install --user` for this project
- One venv with both CUDA and XPU PyTorch
- Putting `icpx`, the driver, or `libze_loader` inside the venv
- Installing IPEX-LLM (archived) or FlashInfer / sglang-kernel
- Running `ft device` in `.venv` and treating a missing XPU as a driver
  bug — that venv is not supposed to have `torch`
