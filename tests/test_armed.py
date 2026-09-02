import pytest

from shuttlecut.armed import ArmedParams, segment_armed

# 合成特征约定:帧率 5fps(步长 0.2s),基线腕速 10 → MAD=0 回退 1.0,
# 挥拍帧腕速 ≥480 → z_w 远超阈值。就位= n_by_side 双侧 ≥1。


def _feat(t: float, n: tuple = (1, 1), wrist: float = 10.0, ready: bool = True) -> dict:
    return {"t": t, "n_by_side": n, "wrist_peak": wrist, "any_ready": ready}


def _timeline(end: float, events: dict, ready_off_from: float | None = None,
              n: tuple = (1, 1)) -> list[dict]:
    rows = []
    t = 0.0
    while t <= end + 1e-9:
        rows.append(_feat(round(t, 1), n=n, wrist=events.get(round(t, 1), 10.0),
                          ready=ready_off_from is None or t < ready_off_from))
        t += 0.2
    return rows


def test_standard_rally_boundaries_follow_backtrack_rules() -> None:
    events = {2.0: 500.0, 2.6: 600.0, 3.2: 550.0, 3.8: 520.0, 4.0: 510.0}
    features = _timeline(7.0, events, ready_off_from=4.2)
    transients = [2.02, 2.62, 3.22, 3.82, 4.02]
    rallies = segment_armed(features, transients)
    assert len(rallies) == 1
    r = rallies[0]
    # 进入 RALLY 回溯 0.5s(事件帧 2.0),结束=最后事件 4.0 + 0.5s
    assert r.start == 1.5
    assert r.end == 4.5
    assert r.motion_peak == 600.0
    assert r.confidence == 1.0
    assert r.hits == 5


def test_short_rally_kept_when_meeting_min_duration() -> None:
    features = _timeline(5.0, {2.0: 500.0, 2.8: 480.0}, ready_off_from=3.0)
    rallies = segment_armed(features, [2.02, 2.82])
    assert len(rallies) == 1
    assert (rallies[0].start, rallies[0].end) == (1.5, 3.3)
    assert rallies[0].hits == 2


def test_single_event_rally_dropped_below_min_duration() -> None:
    features = _timeline(4.0, {2.0: 500.0}, ready_off_from=2.2)
    assert segment_armed(features, [2.02]) == []


def test_single_event_rally_kept_with_lower_min_rally_param() -> None:
    features = _timeline(4.0, {2.0: 500.0}, ready_off_from=2.2)
    rallies = segment_armed(features, [2.02], ArmedParams(min_rally_s=0.8))
    assert len(rallies) == 1
    assert (rallies[0].start, rallies[0].end) == (1.5, 2.5)
    assert rallies[0].hits == 1


def test_adjacent_court_audio_pollution_produces_no_segment() -> None:
    # 邻场:音频瞬态密集但腕速始终基线 → 不得进入 RALLY
    features = _timeline(8.0, {})
    transients = [round(2.0 + 0.3 * k, 2) for k in range(14)]
    assert segment_armed(features, transients) == []


def test_no_readiness_produces_no_segment() -> None:
    # 无就位:腕速峰高但一侧始终无人 → 不产生段
    features = _timeline(6.0, {2.0: 500.0, 3.0: 520.0}, n=(2, 0))
    assert segment_armed(features, []) == []


def test_nearby_rallies_merged_once() -> None:
    events = {5.0: 500.0, 5.6: 560.0, 7.2: 520.0, 7.8: 540.0}
    rows = []
    t = 0.0
    while t <= 10.0 + 1e-9:
        ready = not (5.8 <= t <= 6.6 or 8.0 <= t <= 8.8)
        rows.append(_feat(round(t, 1), wrist=events.get(round(t, 1), 10.0), ready=ready))
        t += 0.2
    transients = [5.02, 5.62, 7.22, 7.82]
    rallies = segment_armed(rows, transients)
    # 两段 [4.5,6.1] 与 [6.7,8.3] 间隙 0.6s < min_idle_s → 合并一次
    assert len(rallies) == 1
    assert (rallies[0].start, rallies[0].end) == (4.5, 8.3)
    assert rallies[0].motion_peak == 560.0
    assert rallies[0].hits == 4


def test_event_after_confirm_timeout_still_detected() -> None:
    # ARMED 超 confirm_s 无事件回 IDLE;就位持续则重新 ARMED,迟来的发球仍检出
    features = _timeline(8.0, {6.0: 500.0}, ready_off_from=6.2)
    rallies = segment_armed(features, [6.02], ArmedParams(min_rally_s=0.8))
    assert len(rallies) == 1
    assert (rallies[0].start, rallies[0].end) == (5.5, 6.5)


def test_empty_features_return_empty() -> None:
    assert segment_armed([], [1.0]) == []
