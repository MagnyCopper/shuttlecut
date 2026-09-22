"""模型解析优先级测试(_resolve_models;产品原则:自动解析均匀,无按 stem 特例)。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shuttlecut.cli import _resolve_models


def test_no_model_found(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _resolve_models(None) == []


def test_official_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "shuttlecut.pt").write_bytes(b"x")
    assert _resolve_models(None) == ["models/shuttlecut.pt"]


def test_calibrated_model_not_auto_picked(tmp_path, monkeypatch):
    """stem 命名校准模型不参与自动解析(产品均匀性);仅 --model 显式使用。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "shuttlecut.pt").write_bytes(b"x")
    (tmp_path / "models" / "shuttlecut-vid.pt").write_bytes(b"x")
    assert _resolve_models(None) == ["models/shuttlecut.pt"]


def test_explicit_arg_passthrough(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _resolve_models("a.pt, b.pt ,") == ["a.pt", "b.pt"]
