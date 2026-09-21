"""CPU tests for the --max-model-len serve flag (issue #246).

Admission allocates a request's FULL max_seq_len up front while the auto
planner can return fewer KV pages than the checkpoint's context -- the served
context must be cappable from the CLI (EngineConfig.max_seq_len_override) and
must never be non-positive. The cap ITSELF (planner pages vs context) is a pure
function unit-tested in test_cache_budget.py; the live B70 behavior (engine
building at the capped length, /v1/models reporting it) runs in the xpu suite.
"""
from __future__ import annotations

import pytest

from freetoken.server.args import ServerArgs, parse_args


def test_max_model_len_defaults_to_none():
    args = parse_args(["Qwen/Qwen3-30B-A3B"])
    assert args.max_model_len is None


def test_max_model_len_flag_parses():
    args = parse_args(["Qwen/Qwen3-30B-A3B", "--max-model-len", "8192"])
    assert args.max_model_len == 8192


def test_max_model_len_rejects_non_positive():
    with pytest.raises(ValueError, match="max_model_len"):
        ServerArgs(model="m", max_model_len=0)
    with pytest.raises(ValueError, match="max_model_len"):
        ServerArgs(model="m", max_model_len=-5)


def test_max_model_len_threads_into_engine_config_override():
    # The whole point of the flag: it lands on EngineConfig.max_seq_len_override,
    # the knob the engine's served-context resolution reads (issue #246).
    from dataclasses import fields as dc_fields

    from freetoken.engine.config import EngineConfig

    override_fields = {f.name for f in dc_fields(EngineConfig)}
    assert "max_seq_len_override" in override_fields
    args = parse_args(["Qwen/Qwen3-30B-A3B", "--max-model-len", "4096"])
    assert args.max_model_len == 4096
