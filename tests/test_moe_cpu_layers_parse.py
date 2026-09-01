"""Unit tests for the ``--moe-cpu-layers`` spec parser (issue #8, ADR 0002).

The parser (``freetoken.moe.parse_moe_cpu_layers``) is **pure Python**: it lives
in the ``freetoken.moe`` package, which the CPU venv imports without torch, so
this test carries no torch import and runs in *both* venvs (the CPU per-PR job
and the XPU nightly). It pins the spec grammar the loader resolves at model-build
time and the blocks dispatch on per layer:

* ``None`` / ``""`` -> **no CPU override**: no MoE layer is steered to the CPU,
  so the resolved MoE backend runs entirely as chosen (``--moe-backend auto`` on
  an XPU -> ``offload``, the ADR 0002 LRU slot pool -- the 32 GB B70 default).
  This is the serve default, so an *unspecified* spec must not pull layers onto
  the CPU (that would flip the offload invariant the #18 tests pin).
* ``"auto"`` -> explicit opt-in to the CPU path: **all** MoE layers on the CPU
  (the ``--moe-backend cpu`` default; the #9 bandwidth profile that would select a
  head+tail subset is not online yet).
* ``"0"`` -> no MoE layers on the CPU (all offload); the explicit form of the
  ``None``/``""`` default.
* an integer ``N`` -> the first ``N`` MoE layers (``N >= total`` -> all).
* a fraction ``F`` in ``(0, 1]`` -> the first ``ceil(F * total)`` MoE layers.
* an explicit id list ``"3,7"`` -> exactly those (de-duplicated, ascending,
  in-bounds validated; an out-of-range id raises).

A silent ignore of a bad id would mis-partition the model (a layer routed to the
wrong backend reads the wrong expert bytes -> divergent logits), so the parser
must reject out-of-bounds ids loudly.
"""
from __future__ import annotations

import pytest

from freetoken.moe import parse_moe_cpu_layers


def test_parse_moe_cpu_layers_specs():
    total = 8
    # "no CPU override" (the serve default): None / empty -> no MoE layer is
    # steered to the CPU, so the resolved backend (auto -> offload) runs as-is.
    assert parse_moe_cpu_layers(None, total) == []
    assert parse_moe_cpu_layers("", total) == []
    # Explicit opt-in to the CPU path: "auto" -> every MoE layer on the CPU
    # (the --moe-backend cpu default; the #9 bandwidth profile that would select
    # a head+tail subset is not online yet).
    assert parse_moe_cpu_layers("auto", total) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert parse_moe_cpu_layers("AUTO", total) == [0, 1, 2, 3, 4, 5, 6, 7]
    # "0": no MoE layers on the CPU (all offload) -- the explicit form of the
    # None/"" default.
    assert parse_moe_cpu_layers("0", total) == []
    # Integer count: the first N MoE layers; N >= total -> all (== None).
    assert parse_moe_cpu_layers("2", total) == [0, 1]
    assert parse_moe_cpu_layers("7", total) == [0, 1, 2, 3, 4, 5, 6]
    assert parse_moe_cpu_layers("8", total) is None  # N == total -> all
    assert parse_moe_cpu_layers("9", total) is None  # N > total -> all
    # Fraction: the first ceil(F * total) MoE layers; ceil hits total -> all.
    assert parse_moe_cpu_layers("0.5", total) == [0, 1, 2, 3]
    assert parse_moe_cpu_layers("0.75", total) == [0, 1, 2, 3, 4, 5]  # ceil(0.75*8)=6
    assert parse_moe_cpu_layers("0.999", total) is None  # ceil(0.999*8) == 8
    assert parse_moe_cpu_layers("1.0", total) is None  # F == 1 -> all
    # Explicit id list: de-duplicated, ascending, in-bounds validated.
    assert parse_moe_cpu_layers("3,7", total) == [3, 7]
    assert parse_moe_cpu_layers("7,3", total) == [3, 7]
    assert parse_moe_cpu_layers("3,3,7", total) == [3, 7]
    assert parse_moe_cpu_layers(" 1 , 5 ", total) == [1, 5]


def test_parse_moe_cpu_layers_rejects_bad_specs():
    # Out-of-bounds / negative / non-numeric specs must raise, not silently
    # mis-partition (a silent ignore routes a layer to the wrong backend ->
    # the wrong expert bytes -> divergent logits, which the forward tests catch
    # end to end but the parser must reject up front).
    with pytest.raises(ValueError):
        parse_moe_cpu_layers("3,8", total := 8)  # 8 is out of [0, 8)
    with pytest.raises(ValueError):
        parse_moe_cpu_layers("-1", 8)
    with pytest.raises(ValueError):
        parse_moe_cpu_layers("not-a-spec", 8)
    with pytest.raises(ValueError):
        parse_moe_cpu_layers("1.5", 8)  # > 1.0 and not an integer count


def test_parse_moe_cpu_layers_zero_layers():
    # A model with no MoE layers (a dense model) has nothing to partition: every
    # spec resolves to [] ("no CPU layers"). The block's _is_cpu_layer gates on the
    # resolved backend first (a dense model is never cpu/hybrid), so this value is
    # inert for dense models.
    assert parse_moe_cpu_layers("3", 0) == []
    assert parse_moe_cpu_layers(None, 0) == []
    assert parse_moe_cpu_layers("auto", 0) == []
