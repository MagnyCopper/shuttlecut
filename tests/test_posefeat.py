import cv2
import numpy as np

from shuttlecut.court import CourtCal
from shuttlecut.posefeat import court_players, frame_features, stance_ready, wrist_speed
from shuttlecut.poses import FramePose, PersonKps


def _cal() -> CourtCal:
    target = np.asarray([(0, 13.4), (3.05, 13.4), (3.05, 0), (0, 0), (1.525, 6.7)], dtype=np.float32)
    image = np.asarray([(200, 200), (1000, 200), (1200, 600), (100, 600)], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(target[:4], image)
    points = cv2.perspectiveTransform(target[None], transform)[0]
    return CourtCal([tuple(map(float, point)) for point in points[:4]], tuple(map(float, points[4])))


def _person(x: float, y: float, *, wrist: float = 0, ready: bool = False) -> PersonKps:
    kps = np.full((17, 2), np.nan)
    kps[11] = (x, y - 10 if ready else y - 80)
    kps[12] = (x + 10, y - 10 if ready else y - 80)
    kps[13] = (x, y - 40)
    kps[14] = (x + 10, y - 40)
    kps[15] = (x, y)
    kps[16] = (x + 10, y)
    kps[7] = (x - 20, y - 60)
    kps[8] = (x + 30, y - 60)
    kps[9] = (x - 30 + wrist, y - 60)
    kps[10] = (x + 40 + wrist, y - 60)
    return PersonKps(kps, 1.0)


def test_court_players_filters_outside_and_assigns_side() -> None:
    cal = _cal()
    inside = _person(650, 500)
    outside = _person(50, 50)
    result = court_players([FramePose(1.0, [inside, outside])], cal)
    assert result[0].t == 1.0
    assert [(person, side) for person, side in result[0].players] == [(inside, 0)]


def test_wrist_speed_detects_large_motion() -> None:
    prev = _person(650, 500)
    current = _person(650, 500, wrist=100)
    steady = _person(650, 500, wrist=1)
    assert wrist_speed(current, prev) > wrist_speed(steady, prev) * 3


def test_stance_ready_distinguishes_bent_and_upright() -> None:
    assert stance_ready(_person(650, 500, ready=True))
    assert not stance_ready(_person(650, 500))


def test_frame_features_reports_counts_peak_and_readiness() -> None:
    cal = _cal()
    previous = _person(650, 500)
    current = _person(650, 500, wrist=100, ready=True)
    rows = frame_features([FramePose(0.0, [previous]), FramePose(0.2, [current])], cal)
    assert rows[0] == {"t": 0.0, "n_by_side": (1, 0), "wrist_peak": 0.0, "any_ready": False}
    assert rows[1]["n_by_side"] == (1, 0)
    assert rows[1]["wrist_peak"] > 400
    assert rows[1]["any_ready"] is True


def test_frame_features_matches_people_by_standing_point_when_order_changes() -> None:
    cal = _cal()
    waving = _person(500, 500)
    still = _person(800, 500)
    waving_current = _person(500, 500, wrist=100)
    still_current = _person(800, 500)
    rows = frame_features(
        [FramePose(0.0, [waving, still]), FramePose(0.2, [still_current, waving_current])], cal
    )
    assert rows[1]["wrist_peak"] > 400


def test_court_players_keeps_four_people_nearest_to_net() -> None:
    cal = _cal()
    people = [_person(650, 300 + index * 50) for index in range(5)]
    result = court_players([FramePose(0.0, people)], cal)
    assert len(result[0].players) == 4
    assert all(person is not people[4] for person, _ in result[0].players)


def test_missing_arm_points_use_available_points_and_stance_requires_all_groups() -> None:
    prev = _person(650, 500)
    current = _person(650, 500, wrist=100)
    current.kps[7] = np.nan
    current.kps[8] = np.nan
    assert wrist_speed(current, prev) > 400
    current.kps[11:17] = np.nan
    assert not stance_ready(current)


def test_court_players_applies_inclusive_x_boundary() -> None:
    cal = _cal()
    target = np.asarray([(-0.10, 6.7), (-0.30, 6.7)], dtype=np.float32)
    image = np.asarray([(200, 200), (1000, 200), (1200, 600), (100, 600)], dtype=np.float32)
    source = np.asarray([(0, 13.4), (3.05, 13.4), (3.05, 0), (0, 0)], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source, image)
    points = cv2.perspectiveTransform(target[None], transform)[0]
    inside = _person(*points[0])
    outside = _person(*points[1])
    result = court_players([FramePose(0.0, [inside, outside])], cal)
    assert len(result[0].players) == 1


def test_frame_features_skips_unmatched_far_person_between_frames() -> None:
    # 上一帧在 A 处,当前帧同一个人瞬移 >150px → 未匹配,腕速不计入峰值
    cal = _cal()
    near = FramePose(0.0, [_person(650, 300)])
    far = FramePose(0.2, [_person(650, 300 + 180, wrist=200)])
    feats = frame_features([near, far], cal)
    assert feats[1]["wrist_peak"] == 0.0


def test_court_players_applies_inclusive_y_boundary() -> None:
    cal = _cal()
    # 直接构造映射到 Y=-0.10 与 Y=13.50(界内)及 Y=-0.30、Y=13.80(界外)的立足点
    import cv2 as _cv2
    target = np.asarray(
        [(1.5, -0.10), (1.5, 13.50), (1.5, -0.30), (1.5, 13.80)], dtype=np.float32
    )
    std = np.asarray([(0, 13.4), (3.05, 13.4), (3.05, 0), (0, 0)], dtype=np.float32)
    image = np.asarray([(200, 200), (1000, 200), (1200, 600), (100, 600)], dtype=np.float32)
    h = _cv2.getPerspectiveTransform(std, image)
    pts = _cv2.perspectiveTransform(target[None], h)[0]
    poses = FramePose(0.0, [_person(float(x), float(y)) for x, y in pts])
    result = court_players([poses], cal)
    assert len(result[0].players) == 2  # 仅 -0.10 与 13.50 保留
