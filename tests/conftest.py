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

import importlib.util

import pytest

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
def _reset_generation_hook():
    # create_app installs a process-global tokenizer hook (the #95 message
    # frontend resolver) into generation._frontend_tokenizer_hook. Without a
    # reset, one test's hook (and its lazily-loaded model) would leak into every
    # later test that drives the generation seam. Clear it around each test.
    from freetoken.server import generation

    generation.set_frontend_tokenizer_hook(None)
    yield
    generation.set_frontend_tokenizer_hook(None)
