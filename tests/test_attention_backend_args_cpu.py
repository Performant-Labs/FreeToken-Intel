"""CPU tests for the --attention-backend serve flag (issue #293).

The torch-dependent half (create_attention_backend resolution, the flag
reaching EngineConfig through the real engine holder) lives in
test_attention_backend_selection.py, gated behind pytest.importorskip("torch")
per tests/conftest.py's dual-venv contract.
"""
from __future__ import annotations

from dataclasses import fields as dc_fields

from freetoken.engine.config import EngineConfig
from freetoken.server.args import parse_args
from freetoken.server.launch import EXIT_OK


def test_attention_backend_defaults_to_auto():
    server_args = parse_args(["m"])
    assert server_args.attention_backend == "auto"


def test_attention_backend_flag_parses_registered_names():
    for choice in ("auto", "torch", "triton", "sycl"):
        server_args = parse_args(["m", "--attention-backend", choice])
        assert server_args.attention_backend == choice


def test_attention_backend_rejects_upstream_cuda_names():
    for bogus in ("trtllm", "fa,fi", "fi"):
        try:
            parse_args(["m", "--attention-backend", bogus])
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError(f"{bogus!r} must not be an accepted --attention-backend choice")


def test_serve_help_documents_attention_backend(capsys):
    from freetoken.server import launch_server

    capsys.readouterr()
    assert launch_server(["--help"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "--attention-backend" in out
    assert "pure-PyTorch GQA" in out


def test_engine_config_has_attention_backend_field():
    assert "attention_backend" in {f.name for f in dc_fields(EngineConfig)}
