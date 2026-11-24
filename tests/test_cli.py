"""The command line."""

import json

import pytest

from bythebooktest import __version__
from bythebooktest.cli import main, parser


def test_every_command_is_registered():
    commands = parser()._subparsers._group_actions[0].choices  # type: ignore[union-attr]
    assert set(commands) == {"generate", "prune", "run", "status", "metrics", "analysis", "probe"}


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0 and __version__ in capsys.readouterr().out


def test_invalid_configuration_is_reported_without_traceback(tmp_path, capsys):
    path = tmp_path / "gen.toml"
    path.write_text("seed = 1\n[level.1]\nkbs = 0\nclaims_per_cell = 4\nnoise_ratio = 0.5\ngroups = []\n")
    assert main(["generate", "--config", str(path), "--out", str(tmp_path / "out")]) == 2
    assert "invalid configuration" in capsys.readouterr().err


def test_status_command(tmp_path, capsys):
    path = tmp_path / "status.json"
    assert main(["status", str(path)]) == 1
    from bythebooktest.status import StatusWriter

    StatusWriter(path, plan_total=4, plan_label="claims").update(phase="x", step="y", step_done=1, step_total=2)
    assert main(["status", str(path)]) == 0
    assert "Plan" in capsys.readouterr().out
    assert json.loads(path.read_text())["phase"] == "x"
