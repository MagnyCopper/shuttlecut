from shuttlecut import __version__
from shuttlecut.cli import main


def test_version_flag(capsys):
    try:
        main(["--version"])
    except SystemExit as e:
        assert e.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_subcommand_shows_help(capsys):
    assert main([]) == 1
    assert "process" in capsys.readouterr().out
