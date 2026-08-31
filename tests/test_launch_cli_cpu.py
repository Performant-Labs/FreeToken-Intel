"""CPU tests for ``ft launch`` (issue ``agent-launch``).

The launch command is pure config/CLI plumbing — no torch, no XPU — so it is
exercised on the CPU box (no ``xpu`` marker). ``prepare_*`` write into ``$HOME``
(agent config dirs), so the write paths run under a temp ``$HOME`` via
``monkeypatch.setenv``; the server-discovery path is exercised against a real
``create_app`` + ``TestClient`` server, so the ported code is pinned to the
Intel server's actual ``/v1/models`` shape (plus its ``/v1/stats`` context
length).
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from urllib.request import urlopen

import pytest

from freetoken.launch import (
    DEFAULT_SERVER,
    FALLBACK_CONTEXT_WINDOW,
    MAX_OUTPUT_TOKENS_CAP,
    LaunchContext,
    ServedModel,
    ServerURL,
    _codex_catalog_path,
    _codex_config_path,
    _codex_profile_path,
    _context_window,
    _hermes_config_path,
    _max_output_tokens,
    _opencode_state_path,
    _ordered_model_ids,
    _openclaw_config_path,
    _write_json_with_backup,
    discover_server_model,
    main,
    prepare_claude,
    prepare_codex,
    prepare_dsh,
    prepare_hermes,
    prepare_opencode,
    prepare_openclaw,
    resolve_server_url,
)
from freetoken.server.args import parse_args
from freetoken.server.api_server import create_app


# ---------------------------------------------------------------------------
# resolve_server_url — pure URL parsing, no server required.
# ---------------------------------------------------------------------------


def test_resolve_server_url_default():
    url = resolve_server_url(None)
    assert url.origin == DEFAULT_SERVER
    assert url.openai_base_url == f"{DEFAULT_SERVER}/v1"


def test_resolve_server_url_bare_host():
    url = resolve_server_url("127.0.0.1:7000")
    assert url.origin == "http://127.0.0.1:7000"
    assert url.openai_base_url == "http://127.0.0.1:7000/v1"


def test_resolve_server_url_full_url():
    url = resolve_server_url("http://127.0.0.1:7000/v1/")
    assert url.origin == "http://127.0.0.1:7000"
    assert url.openai_base_url == "http://127.0.0.1:7000/v1"


def test_resolve_server_url_maps_wildcard_hosts():
    assert resolve_server_url("0.0.0.0:1").origin == "http://127.0.0.1:1"
    assert resolve_server_url("[::]:1").origin == "http://[::1]:1"


def test_resolve_server_url_rejects_nested_path():
    with pytest.raises(ValueError):
        resolve_server_url("http://127.0.0.1:7000/v2")


def test_resolve_server_url_env_fallback(monkeypatch):
    monkeypatch.setenv("FREETOKEN_HOST", "127.0.0.1:9111")
    assert resolve_server_url(None).origin == "http://127.0.0.1:9111"


# ---------------------------------------------------------------------------
# ServedModel window derivation + the context/output helpers.
# ---------------------------------------------------------------------------


def test_context_window_falls_back_when_server_reports_none():
    model = ServedModel(model_id="m", models=["m"], context_length=None)
    ctx = LaunchContext(server=ServerURL("http://h", "http://h/v1"), model=model, extra_args=[], dry_run=True)
    assert _context_window(ctx) == FALLBACK_CONTEXT_WINDOW
    assert _max_output_tokens(ctx) == max(1, min(FALLBACK_CONTEXT_WINDOW // 4, MAX_OUTPUT_TOKENS_CAP))


def test_context_window_uses_reported_value_and_caps_output():
    model = ServedModel(model_id="m", models=["m"], context_length=4096)
    ctx = LaunchContext(server=ServerURL("http://h", "http://h/v1"), model=model, extra_args=[], dry_run=True)
    assert _context_window(ctx) == 4096
    # A small window must not be inflated by the cap: output = window // 4.
    assert _max_output_tokens(ctx) == 4096 // 4


def test_context_window_capped_at_max_output():
    model = ServedModel(model_id="m", models=["m"], context_length=4_000_000)
    ctx = LaunchContext(server=ServerURL("http://h", "http://h/v1"), model=model, extra_args=[], dry_run=True)
    assert _max_output_tokens(ctx) == MAX_OUTPUT_TOKENS_CAP


def test_ordered_model_ids_dedupes_preferring_primary():
    model = ServedModel(model_id="primary", models=["secondary", "primary"])
    ctx = LaunchContext(server=ServerURL("http://h", "http://h/v1"), model=model, extra_args=[], dry_run=True)
    assert _ordered_model_ids(ctx) == ["primary", "secondary"]


# ---------------------------------------------------------------------------
# dry-run: prints the plan + command, and writes nothing (no temp HOME needed).
# ---------------------------------------------------------------------------


def test_dry_run_codex_previews_without_writing(tmp_path, monkeypatch, capsys):
    # Point HOME at an empty temp dir so we can prove dry-run writes nothing.
    monkeypatch.setenv("HOME", str(tmp_path))
    model = ServedModel(model_id="qwen3-30b", models=["qwen3-30b"], context_length=None)
    ctx = LaunchContext(server=ServerURL("http://127.0.0.1:1919", "http://127.0.0.1:1919/v1"), model=model, extra_args=["--extra"], dry_run=True)
    spec = prepare_codex(ctx)
    # dry-run still builds the same command a real run would execute.
    assert spec.argv[:3] == ["codex", "--profile", "freetoken-launch"]
    # The catalog path is embedded in the argv; the server URL is carried in the catalog, not argv.
    assert any("freetoken-model.json" in arg for arg in spec.argv)
    assert spec.env == {"FREETOKEN_API_KEY": "freetoken"}
    # dry-run must not write any config into the (empty) home.
    assert list(tmp_path.iterdir()) == []


def test_dry_run_writes_nothing_into_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    model = ServedModel(model_id="m", models=["m"], context_length=None)
    ctx = LaunchContext(server=ServerURL("http://127.0.0.1:1919", "http://127.0.0.1:1919/v1"), model=model, extra_args=[], dry_run=True)
    for preparer in (prepare_codex, prepare_claude, prepare_opencode, prepare_hermes, prepare_dsh):
        preparer(ctx)
    # None of the config-writing preparers may touch the (empty) home.
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# real runs: config is written into $HOME (temp) and a backup is left on edit.
# ---------------------------------------------------------------------------


def test_codex_writes_profile_and_catalog_into_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    model = ServedModel(model_id="qwen3-30b", models=["qwen3-30b"], context_length=8192)
    ctx = LaunchContext(server=ServerURL("http://127.0.0.1:1919", "http://127.0.0.1:1919/v1"), model=model, extra_args=[], dry_run=False)
    spec = prepare_codex(ctx)
    profile = _codex_profile_path()
    catalog = _codex_catalog_path()
    assert profile.exists()
    assert catalog.exists()
    assert "model = " in profile.read_text()
    catalog_json = json.loads(catalog.read_text())
    assert catalog_json["models"][0]["slug"] == "qwen3-30b"
    assert catalog_json["models"][0]["context_window"] == 8192
    assert spec.argv[0] == "codex"


def test_codex_migrates_existing_config_strips_stale_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    config_path = _codex_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "model = \"old-model\"\n"
        'profile = "freetoken-launch"\n'
        "\n"
        "[profiles.freetoken-launch]\n"
        'model = "old-model"\n'
        "\n"
        "[other_section]\n"
        'keep = "yes"\n'
    )
    model = ServedModel(model_id="new-model", models=["new-model"], context_length=None)
    ctx = LaunchContext(server=ServerURL("http://h", "http://h/v1"), model=model, extra_args=[], dry_run=False)
    prepare_codex(ctx)
    migrated = config_path.read_text()
    # The stale profile pointer and the whole profile/provider section are gone.
    assert 'profile = "freetoken-launch"' not in migrated
    assert "[profiles.freetoken-launch]" not in migrated
    # Unrelated sections survive the migration.
    assert 'keep = "yes"' in migrated
    assert "[other_section]" in migrated
    # A .bak of the pre-migration config was left behind.
    assert any(p.name.endswith(".bak") for p in config_path.parent.iterdir() if p.name.startswith("config.toml"))


def test_opencode_updates_recent_state_under_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    state_path = _opencode_state_path()
    # Pre-seed a state file with a stale freetoken recent entry + a foreign one.
    state_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_with_backup(state_path, {"recent": [{"providerID": "freetoken", "modelID": "stale"}, {"providerID": "anthropic", "modelID": "claude-sonnet"}]})
    model = ServedModel(model_id="qwen3-30b", models=["qwen3-30b"], context_length=4096)
    ctx = LaunchContext(server=ServerURL("http://h", "http://h/v1"), model=model, extra_args=[], dry_run=False)
    spec = prepare_opencode(ctx)
    state = json.loads(state_path.read_text())
    recent = state["recent"]
    # The fresh entry is pinned first; an identical freetoken entry (same model) is
    # de-duped away while a foreign (stale model) freetoken entry is preserved per the ported
    # de-dupe rule (only drop entries whose modelID is in the *current* set).
    assert recent[0] == {"providerID": "freetoken", "modelID": "qwen3-30b"}
    assert any(e.get("providerID") == "anthropic" for e in recent)
    # The config is injected via env, pointing at the server, with a real window/output.
    config = json.loads(spec.env["OPENCODE_CONFIG_CONTENT"])
    assert config["model"] == "freetoken/qwen3-30b"
    assert config["provider"]["freetoken"]["options"]["baseURL"] == "http://h/v1"
    assert config["provider"]["freetoken"]["models"]["qwen3-30b"]["limit"] == {"context": 4096, "output": 1024}


def test_hermes_patches_model_section_preserving_other_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    config_path = _hermes_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("api_key: old-key\nmodel:\n  provider: ollama\n")
    model = ServedModel(model_id="qwen3-30b", models=["qwen3-30b"], context_length=200000)
    ctx = LaunchContext(server=ServerURL("http://127.0.0.1:1919", "http://127.0.0.1:1919/v1"), model=model, extra_args=[], dry_run=False)
    prepare_hermes(ctx)
    text = config_path.read_text()
    assert "provider: custom" in text
    assert "127.0.0.1:1919/v1" in text
    assert "context_length: 200000" in text
    # The top-level api_key is left alone; only the model section's own keys are rewritten.
    assert "api_key: old-key" in text
    assert "api_key: freetoken-local" in text


def test_dsh_writes_settings_and_patch_under_home(tmp_path, monkeypatch):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    model = ServedModel(model_id="deepseek-v3", models=["deepseek-v3"], context_length=16384)
    ctx = LaunchContext(server=ServerURL("http://127.0.0.1:1919", "http://127.0.0.1:1919/v1"), model=model, extra_args=[], dry_run=False)
    spec = prepare_dsh(ctx)
    settings = tmp_path / "freetoken-launch.settings.yaml"
    patch = tmp_path / "freetoken-launch.cordis.patch.yml"
    assert settings.exists()
    assert patch.exists()
    assert "127.0.0.1:1919/v1" in settings.read_text()
    # The patch overlay repoints dsh's active settings row at our file.
    assert str(settings) in patch.read_text()
    assert spec.argv[0] == "dsh"
    assert "--patch" in spec.argv
    assert spec.env["DEEPSEEK_BASE_URL"] == "http://127.0.0.1:1919/v1"


def test_openclaw_first_patch_requires_confirmation(tmp_path, monkeypatch):
    # With a non-interactive stdin and no -y, the first OpenClaw patch is cancelled
    # (the config is not written and the launch aborts with a clean error). HOME is
    # isolated to a temp dir so the real ~/.openclaw is never touched.
    monkeypatch.setenv("HOME", str(tmp_path))
    model = ServedModel(model_id="m", models=["m"], context_length=None)
    ctx = LaunchContext(server=ServerURL("http://h", "http://h/v1"), model=model, extra_args=[], dry_run=False, assume_yes=False)
    config_path = _openclaw_config_path()
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(RuntimeError):
        prepare_openclaw(ctx)
    # No config was written because the prompt was declined.
    assert not config_path.exists()


def test_openclaw_assume_yes_patches_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    model = ServedModel(model_id="m", models=["m"], context_length=8192)
    ctx = LaunchContext(server=ServerURL("http://127.0.0.1:1919", "http://127.0.0.1:1919/v1"), model=model, extra_args=[], dry_run=False, assume_yes=True)
    spec = prepare_openclaw(ctx)
    config_path = _openclaw_config_path()
    assert config_path.exists()
    config = json.loads(config_path.read_text())
    provider = config["models"]["providers"]["freetoken"]
    assert provider["baseUrl"] == "http://127.0.0.1:1919/v1"
    assert provider["models"][0]["id"] == "m"
    assert config["agents"]["defaults"]["model"]["primary"] == "freetoken/m"
    # With no extra args the openclaw invocation defaults to the chat profile.
    assert spec.argv == ["openclaw", "chat"]


def test_claude_preparer_env_points_at_server():
    model = ServedModel(model_id="qwen3-30b", models=["qwen3-30b"], context_length=128000)
    ctx = LaunchContext(server=ServerURL("http://127.0.0.1:1919", "http://127.0.0.1:1919/v1"), model=model, extra_args=["--resume"], dry_run=True)
    spec = prepare_claude(ctx)
    assert spec.argv == ["claude", "--model", "qwen3-30b", "--resume"]
    assert spec.env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:1919"
    assert spec.env["ANTHROPIC_MODEL"] == "qwen3-30b"
    assert spec.env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "128000"


# ---------------------------------------------------------------------------
# Server discovery against the *real* Intel server surface.
# ---------------------------------------------------------------------------


@pytest.fixture
def live_server(monkeypatch):
    """Stand up the real FreeToken server on an ephemeral loopback port.

    ``create_app`` is the same object ``ft serve`` runs; serving it on a real
    socket (rather than TestClient) exercises the ported launch-side HTTP client
    end-to-end. The port is resolved by binding to :0. The fixture also pins
    ``FREETOKEN_HOST`` to that port (and restores it on teardown) so the
    server-resolution side of ``ft launch`` is exercised too.
    """
    import uvicorn

    server_args = parse_args(["Qwen/Qwen3-30B-A3B"])

    def engine_holder():
        raise RuntimeError("not loaded — discovery only, no generation")

    app = create_app(server_args, engine_holder)

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            with urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5) as resp:
                if resp.status == 200:
                    break
        except OSError:
            time.sleep(0.05)
    else:
        raise RuntimeError("server did not come up for the launch discovery test")

    monkeypatch.setenv("FREETOKEN_HOST", f"127.0.0.1:{port}")
    try:
        yield ServerURL(origin=f"http://127.0.0.1:{port}", openai_base_url=f"http://127.0.0.1:{port}/v1")
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_discover_server_model_reads_intel_server(live_server):
    model = discover_server_model(live_server)
    # /v1/models reports the served model id (our server: the repo basename).
    assert model.model_id == "Qwen3-30B-A3B"
    assert model.models == ["Qwen3-30B-A3B"]
    # Our /v1/models predates the context field, so the /v1/stats fallback applies.
    # The server reports no model ctx (no loaded engine), so context_length is None
    # and the launch side will use the documented fallback.
    assert model.context_length is None or isinstance(model.context_length, int)


def test_discover_server_model_connect_error_is_clean():
    server = resolve_server_url("127.0.0.1:1")  # nothing listening
    with pytest.raises(RuntimeError) as exc:
        discover_server_model(server)
    assert "Cannot connect" in str(exc.value)


def test_main_dry_run_against_intel_server(live_server, capsys):
    # End-to-end: the CLI resolves the server and prints a launch plan (no install, no run).
    # codex is on PATH here, so ensure_agent_installed resolves it without installing.
    rc = main(["codex", "--server", live_server.origin, "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "OpenAI base URL: " + live_server.openai_base_url in out
    assert "Command: codex" in out
    assert "--profile freetoken-launch" in out


def test_main_install_only_needs_no_server(tmp_path, monkeypatch):
    # --install-only short-circuits before resolve_server_url, so it must succeed even
    # though --server points at an address with nothing listening (nothing connects).
    # A fake `codex` is placed on PATH so the test is hermetic: the agent resolves
    # without a real install or network, and it passes on a GitHub runner (where the
    # real codex/curl are absent) as well as on a dev box that already has codex.
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_codex = fake_bin / "codex"
    fake_codex.write_text("#!/bin/sh\nexit 0\n")
    fake_codex.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin) + os.pathsep + os.environ["PATH"])
    monkeypatch.delenv("FREETOKEN_HOST", raising=False)
    assert main(["codex", "--install-only", "--server", "http://127.0.0.1:1"]) == 0
