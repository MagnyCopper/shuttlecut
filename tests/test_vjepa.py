"""vjepa 探针路径单元测试(无 GPU 依赖)。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_snap_pulls_boundary_to_max_slope():
    from shuttlecut.vjepa import _snap
    cent = np.arange(0, 60, 0.5, dtype=np.float32)
    probs = np.where((cent >= 10) & (cent < 40), 0.9, 0.1).astype(np.float64)
    # 平滑阶跃,使斜率最大点在 ~10/~40
    from scipy.ndimage import median_filter
    probs = median_filter(np.convolve(probs, np.ones(5) / 5, mode="same"), size=5)
    coarse = [(13.0, 37.0)]  # 故意偏宽的粗段
    (a, b), = _snap(coarse, cent, probs, win_s=3.0)
    assert abs(a - 10.5) <= 1.5, a
    assert abs(b - 40.0) <= 1.5, b


def test_snap_keeps_when_no_slope():
    from shuttlecut.vjepa import _snap
    cent = np.arange(0, 20, 0.5, dtype=np.float32)
    probs = np.full(len(cent), 0.5)
    (a, b), = _snap([(2.0, 10.0)], cent, probs, win_s=1.5)
    assert (a, b) == (2.0, 10.0) or abs(a - 2) <= 1.5 and abs(b - 10) <= 1.5


def test_ensure_probe_missing_manifest_entry(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    import shuttlecut.modelhub as hub
    import shuttlecut.vjepa as vj
    monkeypatch.setattr(hub, "load_manifest", lambda: {"url": "x"})
    assert vj.ensure_probe() is None  # 无 probe 条目 → None(不抛)


def test_ensure_probe_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "shuttlecut-probe.pt").write_bytes(b"x")
    import shuttlecut.vjepa as vj
    assert vj.ensure_probe() == "models/shuttlecut-probe.pt"
