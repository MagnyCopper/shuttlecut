"""boundary_vote 单元测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shuttlecut.temporal import boundary_vote


def test_vote_anchors_on_first_model():
    a = [(10.0, 20.0), (30.0, 40.0)]
    b = [(12.0, 19.0), (29.0, 44.0)]
    c = [(11.0, 21.0), (31.0, 39.0)]
    out = boundary_vote([a, b, c])
    assert len(out) == 2
    s, e = out[0]
    assert abs(s - 11.0) < 0.01 and abs(e - 20.0) < 0.01  # 中位数


def test_vote_merges_near_segments():
    a = [(10.0, 20.0), (20.5, 30.0)]
    out = boundary_vote([a])
    assert len(out) == 1 and out[0][1] == 30.0


def test_vote_empty():
    assert boundary_vote([]) == []
    assert boundary_vote([[]]) == []


def test_vote_drops_short():
    a = [(10.0, 11.0)]  # 投票后 <2.5s 丢弃
    assert boundary_vote([a, a]) == []
