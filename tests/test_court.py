import json

import cv2
import numpy as np
import pytest

from shuttlecut.court import CourtCal, load_cal, save_cal, side_of, to_court_xy


@pytest.fixture
def cal() -> CourtCal:
    target = np.asarray(
        [(0, 13.4), (3.05, 13.4), (3.05, 0), (0, 0), (1.525, 6.7)],
        dtype=np.float32,
    )
    image = np.asarray(
        [(200, 200), (1000, 200), (1200, 600), (100, 600)], dtype=np.float32
    )
    transform = cv2.getPerspectiveTransform(target[:4], image)
    points = cv2.perspectiveTransform(target[None], transform)[0]
    return CourtCal(
        corners=[tuple(float(value) for value in point) for point in points[:4]],
        net_mid=tuple(float(value) for value in points[4]),
    )


def test_to_court_xy_maps_corners_and_net(cal: CourtCal) -> None:
    assert to_court_xy(cal, 100, 600) == pytest.approx((0, 0), abs=0.3)
    assert to_court_xy(cal, 1200, 600) == pytest.approx((3.05, 0), abs=0.3)
    assert to_court_xy(cal, 200, 200) == pytest.approx((0, 13.4), abs=0.3)
    assert to_court_xy(cal, 1000, 200) == pytest.approx((3.05, 13.4), abs=0.3)
    assert to_court_xy(cal, *cal.net_mid) == pytest.approx((1.525, 6.7), abs=0.3)


def test_side_of_distinguishes_near_and_far_halves(cal: CourtCal) -> None:
    assert side_of(cal, 650, 500) == 0
    assert side_of(cal, 650, 300) == 1


def test_save_load_roundtrip(cal: CourtCal, tmp_path) -> None:
    path = tmp_path / "court.json"
    save_cal(cal, path)
    assert load_cal(path) == cal


def test_load_rejects_wrong_corner_count(tmp_path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"corners": [[0, 0]], "net_mid": [0, 0]}))
    with pytest.raises(ValueError):
        load_cal(path)
