# Contributing to FreeToken-Intel

## Scope

This is the Intel Arc Pro B70 port of
[FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken). Prefer
filling an existing stub over adding a parallel tree. Each stub names a
GitHub issue slug; implement against that issue.

## Setup

Follow [docs/dev-setup.md](docs/dev-setup.md) (system packages on the OS,
Python in `.venv` / `.venv-xpu`). Short version:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -m "not xpu and not slow"
```

XPU tests need a B70 (or other Xe2) plus oneAPI. Report driver +
`ft device` output on XPU bugs.

## Pull requests

* One change per PR; link the issue.
* Conventional Commits: `feat(moe): ...`, `fix(kernel): ...`.
* Do not copy CUDA kernels and `#ifdef` them. Write SYCL / Triton-Intel /
  oneDNN paths.
* Do not depend on IPEX-LLM (archived).
* Stubs must keep raising `freetoken._stub.NotYetImplemented` until the
  feature actually works.

## License

Contributions are Apache 2.0 (see `LICENSE`).
