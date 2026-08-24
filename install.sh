#!/usr/bin/env bash
# Minimal bootstrap. Full oneAPI / XPU setup is documented in docs/install.md.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
python3 -m pip install -e ".[dev]"
echo "Installed freetoken-intel. Run: ft --version && ft device"
