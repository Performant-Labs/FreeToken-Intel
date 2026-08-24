# FreeToken-Intel XPU environment
# Source this in the shell that compiles SYCL or talks to the XPU:
#   source setvars-xpu.sh
# Use together with the XPU venv:  source .venv-xpu/bin/activate
#
# ONEAPI_ROOT points at the oneAPI install at /opt/intel/oneapi, which is the
# Intel oneAPI DPC++/C++ Compiler 2026.1 installed from Intel's apt repo
# (package intel-oneapi-compiler-dpcpp-cpp, see docs/dev-setup.md).
# The real env is set by the official oneAPI script; we just pin the device
# and defer to it.
export ONEAPI_ROOT=/opt/intel/oneapi
# Pin a single Level Zero device if an AMD iGPU shadows the Battlemage.
# (level_zero:0 is the B70 on this machine; adjust index if devices reorder.)
export ONEAPI_DEVICE_SELECTOR=level_zero:0
# Official oneAPI environment: puts icpx/icx/sycl-ls on PATH + sets library paths.
. "$ONEAPI_ROOT/setvars.sh" --force
