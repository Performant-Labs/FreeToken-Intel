# FreeToken-Intel

Edge-native Mixture-of-Experts serving on **Intel Arc Pro B70** (Xe2 / Battlemage).

This repository is an Intel-stack equivalent of
[FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken). The NVIDIA
original maps computation across GPU, CPU, host memory, and PCIe. This port
keeps that design and swaps CUDA for **SYCL / Level Zero / PyTorch XPU**.

Status: **pre-alpha scaffolding**. The package layout, CLI, registries, and
tests are in place. Kernels, engine, and server raise `NotYetImplemented`
until the matching GitHub issues land.

Backlog: [Epic #1](https://github.com/Performant-Labs/FreeToken-Intel/issues/1)
(31 child tasks). Slug → issue map: [docs/architecture.md](docs/architecture.md).

## Why B70

| | Arc Pro B70 |
| --- | --- |
| VRAM | 32 GB GDDR6 |
| Bandwidth | 608 GB/s |
| Xe-cores | 32 (Xe2-HPG) |
| TBP | ~230 W |

That VRAM is enough to serve Qwen3.6-35B-A3B-class MoEs on-device, with the
same expert-offload path for larger models. Lower HBM bandwidth than
high-end NVIDIA cards makes the `hybrid` CPU–XPU policy first-class.

## Layout

Mirrors upstream FreeToken:

* `python/freetoken/` — engine, MoE, attention, KV cache, server, CLI
* `python/freetoken/kernel/csrc/sycl/` — DPC++ kernels (stubs)
* `docs/` — [index](docs/README.md): stack rationale, ADRs, install, architecture map
* `tests/` — CPU-only unit tests for the scaffold

See [docs/stack.md](docs/stack.md) for why each Intel layer was chosen, and
[docs/architecture.md](docs/architecture.md) for the CUDA → Intel name map.

## Install

```bash
git clone https://github.com/Performant-Labs/FreeToken-Intel.git
cd FreeToken-Intel
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
ft --version
```

Machine setup (two venvs, driver, oneAPI): [docs/dev-setup.md](docs/dev-setup.md).
Software list: [docs/dev-requirements.md](docs/dev-requirements.md).

## CI

Four checks gate this repo: `pre-flight` (workflow lint), `secret-scan`
(gitleaks, hard-fail), `ci` (CPU tests + CLI smoke, torch-free venv),
and a nightly `xpu-nightly` that runs the torch/XPU suite on the B70
fleet. An optional bot review runs in parallel and is advisory. Run the local
mirror before pushing with `tools/validate.sh`.
See [docs/ci.md](docs/ci.md) for what each check guards and why.

## CLI

```
ft serve | shell | ctl | daemon | launch | checkpoint | bench | device
```

Default API port (planned): `127.0.0.1:1919`, OpenAI + Anthropic routes.

## License

Apache License 2.0. Inspired by FreeToken (Apache 2.0) and the projects it
credits (SGLang, vLLM, FlashInfer, llama.cpp, and others).
