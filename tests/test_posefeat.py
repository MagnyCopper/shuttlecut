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
    result = court_players([FramePose(1.0, [inside, outside])], cal, (1300, 700))
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
    rows = frame_features([FramePose(0.0, [previous]), FramePose(0.2, [current])], cal, (1300, 700))
    assert rows[0] == {"t": 0.0, "n_by_side": (1, 0), "wrist_peak": 0.0, "any_ready": False}
    assert rows[1]["n_by_side"] == (1, 0)
    assert rows[1]["wrist_peak"] > 400
    assert rows[1]["any_ready"] is True
