#!/usr/bin/env bash
#
# build-release-wheels.sh -- build the FreeToken-Intel CPU sdist/wheel (+ optional
# XPU wheel) for release.
#
# Usage:
#   scripts/build-release-wheels.sh cpu                 # CPU sdist + wheel (CI default)
#   scripts/build-release-wheels.sh xpu                 # XPU wheel (needs oneAPI on PATH)
#   scripts/build-release-wheels.sh all                 # both
#
# The CPU wheel is pure-Python (no compiled kernels) and is the artifact CI builds
# on ubuntu-latest and uploads. It is what `pip install freetoken-intel` gives a CPU
# box; it must NOT bundle torch (the dual-venv contract).
#
# The XPU wheel additionally needs the oneAPI runtime (icpx on PATH) to build any
# SYCL AOT kernels, and is built in the manylinux container with the PyTorch XPU index
# so the embedded torch tag is correct. Today the package has no compiled XPU kernels
# yet (see #10 quant-xpu / #15 engine-graph), so the XPU wheel is built from the same
# pure-Python source -- the script is here so the manylinux/SYCL path is wired up and
# exercised (the issue's "SYCL AOT wheel scripts") the moment a kernel lands.
#
# Artifacts land in dist/ and are uploaded by ci.yml as build artifacts (we do NOT
# publish to PyPI from CI -- that is a manual, key-gated step; see docs/install.md).
#
# Build tooling: `python -m build` is the canonical PEP 517 builder. It is NOT a
# dependency of the package (the repo venvs do not carry it), so if the requested
# interpreter lacks it we create a throwaway isolated venv and `pip install build`
# there (needs network to PyPI -- true on CI, and on a dev box with internet).
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
DIST="$REPO_ROOT/dist"
PYTHON="${PYTHON:-python3}"

# --- ensure a `build`-capable interpreter ------------------------------------------
ensure_build() {
  if "$PYTHON" -m build --version >/dev/null 2>&1; then
    return 0
  fi
  echo "   '$PYTHON' has no 'build' module -- creating an isolated build venv..."
  local bv
  bv="$(mktemp -d "${TMPDIR:-/tmp}/ftbuild.XXXXXX")"
  "$PYTHON" -m venv "$bv"
  "$bv/bin/python" -m pip install -q --upgrade pip
  "$bv/bin/python" -m pip install -q build
  PYTHON="$bv/bin/python"
  echo "   using isolated build venv: $PYTHON"
}

# --- which wheels to build ---------------------------------------------------------
MODE="${1:-cpu}"
case "$MODE" in
  cpu|all) WANT_CPU=1 ;;
  *) WANT_CPU=0 ;;
esac
case "$MODE" in
  xpu|all) WANT_XPU=1 ;;
  *) WANT_XPU=0 ;;
esac

[ -d "$DIST" ] || mkdir -p "$DIST"

# --- CPU: sdist + pure wheel -------------------------------------------------------
build_cpu() {
  echo "==> Building CPU sdist + wheel (pure Python, no torch, no kernels)..."
  ensure_build
  # build isolation ON: the manylinux/CI image may not have a compatible setuptools;
  # build pulls its own. No torch is required (the contract is that the CPU wheel
  # imports without torch).
  "$PYTHON" -m build --sdist --wheel --outdir "$DIST" .
  echo "   artifacts:"
  ls -1 "$DIST"/freetoken_intel-*.whl "$DIST"/freetoken_intel-*.tar.gz 2>/dev/null || ls -1 "$DIST"
}

# --- XPU: wheel built with the oneAPI runtime present ------------------------------
build_xpu() {
  echo "==> Building XPU wheel (oneAPI runtime required on PATH)..."
  if ! command -v icpx >/dev/null 2>&1; then
    echo "   icpx not on PATH. Source the oneAPI setvars first, e.g.:"
    echo "     source /opt/intel/oneapi/setvars.sh"
    echo "   (On a manylinux build, the XPU toolchain is pre-baked into the image.)"
    exit 1
  fi
  ensure_build
  # The XPU wheel is the same pure-Python source today; the distinction is the build
  # environment (oneAPI present) so that when compiled SYCL AOT kernels land (#10),
  # the wheel recipe here already carries the right toolchain. The embedded torch is
  # NOT rebuilt here -- consumers install the XPU torch via the [xpu] extra / the
  # PyTorch XPU index, never from this wheel (see pyproject [xpu]).
  "$PYTHON" -m build --wheel --outdir "$DIST" .
  echo "   XPU wheel built with oneAPI $("$PYTHON" -c 'import intel_sycl_rt' 2>/dev/null && echo present || echo 'rt not importable (ok)'):"
  ls -1 "$DIST"/freetoken_intel-*.whl
}

if [ "$WANT_CPU" = 1 ]; then build_cpu; fi
if [ "$WANT_XPU" = 1 ]; then build_xpu; fi

echo
echo "==> Done. Artifacts in $DIST:"
ls -lh "$DIST"
