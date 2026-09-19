"""calibrate 组装逻辑测试:条带判决 → 回合段。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def assemble(strips):
    """与 cli.calibrate_cmd run 阶段同语义的最小复制(单测锚点)。"""
    segs = []
    for s in strips:
        v = s["verdict"].upper().replace(" ", "")
        step = max((s["times"][-1] - s["times"][0]) / max(len(v) - 1, 1), 4.5)
        for k, ch in enumerate(v):
            if ch == "Y":
                t = s["times"][0] + k * step
                segs.append((t - step / 2, t + step / 2))
    merged = []
    for a, b in sorted(segs):
        if merged and a <= merged[-1][1] + 1.0:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= 2.5]


def test_adjacent_y_merges():
    s = [{"times": [10.0, 13.0, 16.0, 19.0, 22.0, 25.0], "verdict": "YYYYNN"}]
    out = assemble(s)
    assert len(out) == 1
    a, b = out[0]
    assert abs(a - 7.75) < 0.1 and abs(b - 25.75) < 0.1


def test_gap_splits():
    s = [{"times": [10.0, 13.0, 16.0, 19.0, 22.0, 25.0], "verdict": "YYNNYY"}]
    out = assemble(s)
    assert len(out) == 2


def test_all_n_empty():
    s = [{"times": [10.0, 13.0, 16.0, 19.0, 22.0, 25.0], "verdict": "NNNNNN"}]
    assert assemble(s) == []


def test_min_width_floor():
    # 短跨度:步长下限 2.2 → 单 Y 段宽 2.2*2=4.4s ≥ 2.5 保留
    s = [{"times": [10.0, 10.5, 11.0, 11.5, 12.0, 12.5], "verdict": "YNNNNN"}]
    out = assemble(s)
    assert len(out) == 1
    assert out[0][1] - out[0][0] >= 4.4
