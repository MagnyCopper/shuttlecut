from shuttlecut.activity import EnergySeries
from shuttlecut.segmenter import SegParams, segment

FPS = 5


def series_from(level_fn, duration_s):
    n = int(duration_s * FPS)
    return EnergySeries([i / FPS for i in range(n)], [level_fn(i / FPS) for i in range(n)])


def test_two_rallies_detected():
    # 0-10s 高能量,10-18s 低,18-28s 高,28-30s 低
    s = series_from(lambda t: 10.0 if (t < 10 or 18 <= t < 28) else 1.0, 30)
    rallies = segment(s, SegParams())
    assert len(rallies) == 2
    assert abs(rallies[0].start - 0.0) < 1.5 and abs(rallies[0].end - 10.0) < 1.5
    assert abs(rallies[1].start - 18.0) < 1.5 and abs(rallies[1].end - 28.0) < 1.5


def test_short_blip_merged_not_rally():
    # 间歇中 1s 的小尖峰不应产生碎片回合
    def lvl(t):
        return 10.0 if 5.0 <= t < 6.0 else 1.0
    s = series_from(lvl, 20)
    assert segment(s, SegParams()) == []


def test_min_idle_merge():
    # 两个 8s 高能段间隔 1s(< min_idle 2.5)应合并为一个回合
    def lvl(t):
        return 10.0 if (t < 8 or 9 <= t < 17) else 1.0
    s = series_from(lvl, 22)
    rallies = segment(s, SegParams())
    assert len(rallies) == 1
    assert rallies[0].end - rallies[0].start > 15


def test_short_rally_dropped():
    def lvl(t):
        return 10.0 if 5 <= t < 6.5 else 1.0  # 1.5s < min_rally 3s
    s = series_from(lvl, 15)
    assert segment(s, SegParams()) == []


def test_max_rally_split():
    def lvl(t):
        return 10.0 if 5.0 <= t < 295.0 else 1.0  # 290s 连续高能量(首尾留间歇避免 hi<=lo)
    s = series_from(lvl, 300)
    rallies = segment(s, SegParams())
    assert all(r.end - r.start <= 120.0 + 5.0 for r in rallies)
    assert len(rallies) >= 2
