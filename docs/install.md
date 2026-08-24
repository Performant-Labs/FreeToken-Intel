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

Install a **PyTorch XPU** wheel from Intel/PyTorch's XPU index (version pins
live in later `device-layer` work). Then:

```bash
ft --version
ft device
```

## Method 2: from PyPI

Not published yet. Track the packaging issue on the epic.

## Verify (once serve is implemented)

```bash
ft serve --model ~/path/to/Qwen3.6-35B-A3B
curl http://127.0.0.1:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B","messages":[{"role":"user","content":"hi"}]}'
```
