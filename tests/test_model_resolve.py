"""模型解析优先级测试(_resolve_models)。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shuttlecut.cli import _resolve_models


def test_no_model_found(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _resolve_models("vid", None) == []


def test_official_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "shuttlecut.pt").write_bytes(b"x")
    assert _resolve_models("vid", None) == ["models/shuttlecut.pt"]


def test_calibrated_model_wins(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "shuttlecut.pt").write_bytes(b"x")
    (tmp_path / "models" / "shuttlecut-vid.pt").write_bytes(b"x")
    assert _resolve_models("vid", None) == ["models/shuttlecut-vid.pt"]


def test_explicit_arg_passthrough(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _resolve_models("vid", "a.pt, b.pt ,") == ["a.pt", "b.pt"]
