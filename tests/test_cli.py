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
