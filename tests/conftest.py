"""Shared pytest policy for the suite: the "no torch -> no xpu tests" rule.

The load-bearing rule is the **dual-venv contract**: the CPU venv (``.venv``,
which the per-PR CI job builds) must *never* contain ``torch``; the torch (XPU)
build lives only in ``.venv-xpu``. The CI job enforces the install side of that
contract with a hard ``import torch`` guard. This conftest is the
collection-time half: when torch is absent (i.e. we are in the CPU venv), any
``xpu``-marked test is *deselected with a clear reason* instead of erroring at
import/call time.

Today the torch-dependent modules (``test_engine_loop``, ``test_models_loader``,
``test_moe_offload_*``) already self-skip via a module-level
``pytest.importorskip("torch")``, and no test carries the ``xpu`` marker yet
(those live on the XPU nightly, #50). This hook is the single authoritative place
for the policy and future-proofs the CPU venv against a future xpu-marked test
that does *not* import torch at module scope -- which ``importorskip`` alone
would not catch.
"""

from __future__ import annotations

import errno
import fcntl
import importlib.util
import sys
import time

import pytest

# Jupiter has exactly one Intel GPU (Battlemage, 0000:03:00.0), shared with the
# qwen38 vLLM production service (llama-swap). An XPU test that aborts mid-kernel
# corrupts the shared Level-Zero queue and takes qwen38 down for every other
# process on the box for several minutes -- confirmed root cause of repeated
# production incidents on 2026-08-30 (see ~/ai-work/logs/freetoken-hang-incidents.md).
#
# This follows the house GPU-lock convention already proven on Io
# (model-runners/l7-gpu-contention/, ingress/tests/live/test_contention_live.py):
# a named flock per host, taken by any job that touches the GPU. /var/lock ->
# /run/lock (tmpfs) here, matching /var/lock/io_gpu.lock's shape.
#
# IMPORTANT -- this is HALF the fix. It serializes XPU tests against each other,
# Landed 2026-08-30: qwen38's start command (model-runners' config.yaml,
# via deploy/vllm-xpu/gpu-lock-gate.sh) now takes this same lock before
# starting, so a cold start correctly waits instead of racing a running test.
# Known residual gap (accepted, Jupiter-infra's call): the gate only covers
# the START moment, not qwen38's full runtime -- a test can still starve an
# ALREADY-RUNNING qwen38's generation requests by holding the GPU, which will
# surface as a watchdog probe timeout/restart, not corruption. Confirmed live
# the same day. Full per-generate-step locking would close that gap too but
# was deliberately not built (higher engineering cost, the corruption sites
# that actually motivated this were already patched in #91/#94).
_GPU_LOCK_PATH = "/var/lock/jupiter_gpu.lock"
_GPU_LOCK_POLL_S = 5.0

# --- Invocation convention (cannot be enforced from inside conftest.py --
# this is about how a shell/agent LAUNCHES pytest, before this file ever
# runs) -----------------------------------------------------------------
# Prefer `timeout <N> .venv-xpu/bin/python -m pytest ...` over bare
# `nohup ... &` for any xpu-marked test run, especially from an agent
# session. `timeout` is kernel-enforced and self-cleaning regardless of
# whether anything remembers to check on it later. `nohup ... &` is NOT: if
# the launching shell/session ends without the agent capturing the PID
# (`echo $!`), the backgrounded process is orphaned (reparented to init) with
# no way for ANY later session -- including a resumed one -- to find and kill
# it. Confirmed live 2026-08-30: two separate xpu test runs
# (tests/test_layers_norm.py) were launched via bare `nohup ... &`, both
# orphaned when their OpenCode task was stopped/restarted, each burned the
# shared GPU at 100% CPU for up to ~1 hour before being found and killed by
# direct process inspection -- one of the two orphans was found only because
# a LATER, separate session came back to check the redirected log file and,
# having no PID on record, launched a SECOND independent nohup'd run on top
# of the still-alive first one rather than being able to clean it up.
# If backgrounding genuinely is necessary (a run expected to outlive a single
# tool call), capture the PID at launch so a later turn can act on it:
#   nohup .venv-xpu/bin/python -m pytest ... > /tmp/xpu_test.log 2>&1 &
#   echo $! > /tmp/xpu_test.pid
# See ~/ai-work/logs/freetoken-hang-incidents.md for the full incident.

# Stash of the full collected item list, populated by the tryfirst hookwrapper
# below and consumed by the non-wrapper hook, so a single module-level name
# bridges the two hooks without relying on `config` identity.
_all_items: list = []


def pytest_collection_modifyitems(config, items):
    if importlib.util.find_spec("torch") is not None:
        # torch present (e.g. .venv-xpu on a B70 box) -> run everything.
        return
    deselected = [item for item in items if item.get_closest_marker("xpu") is not None]
    if not deselected:
        return
    config.hook.pytest_deselected(items=deselected)
    items[:] = [item for item in items if item.get_closest_marker("xpu") is None]


    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def _populate_all_items(config, items):
        _all_items.clear()
        _all_items.extend(items)
        yield


@pytest.fixture(autouse=True)
def _gpu_lock(request):
    """Serialize xpu-marked tests against each other via the house flock
    convention, so this suite never runs two GPU-heavy jobs concurrently on
    Jupiter's single Intel GPU. See module docstring above for what this does
    NOT yet cover (qwen38's own start command).
    """
    if request.node.get_closest_marker("xpu") is None:
        yield
        return

    fh = open(_GPU_LOCK_PATH, "a")
    waited = 0.0
    try:
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if waited == 0.0:
                    print(
                        f"\n[gpu-lock] {_GPU_LOCK_PATH} held by another job -- waiting "
                        "(this test needs Jupiter's Intel GPU exclusively)",
                        file=sys.stderr,
                    )
                time.sleep(_GPU_LOCK_POLL_S)
                waited += _GPU_LOCK_POLL_S
                if waited % 60 == 0:
                    print(f"[gpu-lock] still waiting ({waited:.0f}s)", file=sys.stderr)
        yield
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


@pytest.fixture(autouse=True)
def _reset_generation_hook():
    # create_app installs a process-global tokenizer hook (the #95 message
    # frontend resolver) into generation._frontend_tokenizer_hook. Without a
    # reset, one test's hook (and its lazily-loaded model) would leak into every
    # later test that drives the generation seam. Clear it around each test.
    from freetoken.server import generation

    generation.set_frontend_tokenizer_hook(None)
    yield
    generation.set_frontend_tokenizer_hook(None)
