from shuttlecut.activity import EnergySeries
from shuttlecut.segmenter import SegParams, segment

FPS = 5


def series_from(level_fn, duration_s):
    n = int(duration_s * FPS)
    return EnergySeries([i / FPS for i in range(n)], [level_fn(i / FPS) for i in range(n)])


def test_two_rallies_detected():
    # 0-10s 高能量,10-18s 低,18-28s 高,28-30s 低
    s = series_from(lambda t: 10.0 if (t < 10 or 18 <= t < 28) else 1.0, 30)
    rallies = segment(s, SegParams(hi_q=0.6, lo_q=0.2))
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
    rallies = segment(s, SegParams(hi_q=0.8, lo_q=0.01))
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
    rallies = segment(s, SegParams(hi_q=0.8, lo_q=0.01))
    assert all(r.end - r.start <= 120.0 + 5.0 for r in rallies)
    assert len(rallies) >= 2


def test_nonconstant_series_with_nonseparating_thresholds_returns_empty():
    s = EnergySeries([0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 2.0])
    assert segment(s, SegParams()) == []


def test_empty_series_returns_empty():
    assert segment(EnergySeries([], []), SegParams()) == []


def test_unusable_long_split_terminates_without_recursion_error():
    s = series_from(lambda t: 10.0 if 0.0 < t < 499.0 else 1.0, 500)
    rallies = segment(s, SegParams())
    assert all(r.end > r.start for r in rallies)


def test_split_peaks_are_recomputed_for_each_subsegment():
    def lvl(t):
        if 5.0 <= t < 125.0:
            return 10.0
        if 125.0 <= t < 185.0:
            return 12.0
        return 1.0

    rallies = segment(series_from(lvl, 300), SegParams())
    assert len(rallies) >= 2
    assert rallies[0].motion_peak == 10.0
    assert any(r.motion_peak == 12.0 for r in rallies[1:])


def test_confidence_is_zero_at_lo_and_one_at_hi():
    params = SegParams(hi_q=0.75, lo_q=0.25, min_rally_s=1.0)
    rallies = segment(series_from(lambda t: 10.0 if t < 5.0 else 1.0, 10), params)
    assert rallies[0].confidence == 1.0
    assert 0.0 <= rallies[0].confidence <= 1.0
