"""输入校验与错误信息测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shuttlecut.cli import _check_input, main


def test_missing_input_clean_error(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    missing = str(tmp_path / "nope.MP4")
    assert main(["process", missing]) == 1
    err = capsys.readouterr().err
    assert "输入视频不存在" in err
    assert "Traceback" not in err


def test_calibrate_missing_input(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    missing = str(tmp_path / "nope.MP4")
    assert main(["calibrate", missing]) == 1
    assert "输入视频不存在" in capsys.readouterr().err
