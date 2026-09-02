from freetoken.cli import main


def test_help_exits_zero():
    assert main(["--help"]) == 0


def test_version_exits_zero(capsys):
    assert main(["--version"]) == 0
    out = capsys.readouterr().out
    assert "freetoken-intel version" in out


def test_unknown_command_exits_two():
    assert main(["not-a-command"]) == 2


def test_no_args_exits_two():
    assert main([]) == 2


def test_checkpoint_help_exits_zero():
    # Regression: `_run_checkpoint` calls `checkpoint.__main__.main(argv,
    # prog=...)` -- a prior version of that main() did not accept `prog` and
    # raised TypeError for every `ft checkpoint` invocation, `--help`
    # included (caught during shell-daemon (#27) follow-up work, PR #127).
    assert main(["checkpoint", "--help"]) == 0


def test_checkpoint_convert_round_trips_through_the_top_level_cli(tmp_path, capsys):
    import json

    torch = __import__("pytest").importorskip("torch")
    from safetensors.torch import save_file

    src = tmp_path / "src_ckpt"
    src.mkdir()
    save_file({"w": torch.randn(4, 4).contiguous()}, str(src / "model.safetensors"))
    (src / "config.json").write_text(json.dumps({"architectures": ["Foo"]}))
    dst = tmp_path / "dst_ckpt"

    assert main(["checkpoint", "convert", str(src), str(dst)]) == 0
    assert (dst / "ftw_index.json").is_file()
    assert "Wrote FTW archive" in capsys.readouterr().out
