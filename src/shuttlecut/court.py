import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import matplotlib.pyplot as plt
import numpy as np

_COURT_WIDTH: Final = 3.05
_COURT_LENGTH: Final = 13.4


@dataclass(frozen=True, slots=True)
class CourtCal:
    """Manual image-space calibration of a badminton court."""

    corners: list[tuple[float, float]]
    net_mid: tuple[float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.corners, (list, tuple)) or len(self.corners) != 4:
            raise ValueError("court calibration must contain four corners")
        points = [*self.corners, self.net_mid]
        try:
            valid = all(
                isinstance(point, (list, tuple))
                and len(point) == 2
                and all(math.isfinite(float(value)) for value in point)
                for point in points
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError("court calibration coordinates must be finite pairs")


def save_cal(cal: CourtCal, path: str | Path) -> None:
    """Save a court calibration as JSON."""
    data = {"corners": cal.corners, "net_mid": cal.net_mid}
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def load_cal(path: str | Path) -> CourtCal:
    """Load and validate a court calibration from JSON."""
    data = json.loads(Path(path).read_text())
    try:
        corners = data["corners"]
        net_mid = data["net_mid"]
        return CourtCal(
            corners=[(float(x), float(y)) for x, y in corners],
            net_mid=(float(net_mid[0]), float(net_mid[1])),
        )
    except (KeyError, TypeError, ValueError, IndexError, OverflowError) as error:
        raise ValueError("invalid court calibration") from error


def to_court_xy(cal: CourtCal, x: float, y: float) -> tuple[float, float]:
    """Map an image pixel to standard badminton court coordinates."""
    source = np.asarray(cal.corners, dtype=np.float32)
    target = np.asarray(
        [(0, _COURT_LENGTH), (_COURT_WIDTH, _COURT_LENGTH), (_COURT_WIDTH, 0), (0, 0)],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, target)
    point = cv2.perspectiveTransform(np.asarray([[[x, y]]], dtype=np.float32), transform)[0, 0]
    return (float(point[0]), float(point[1]))


def side_of(cal: CourtCal, x: float, y: float) -> int:
    """Return 0 for the near half and 1 for the far half of the court."""
    return int(to_court_xy(cal, x, y)[1] >= _COURT_LENGTH / 2)


def pick_court(frame_path: str, out_path: str) -> CourtCal:
    """Interactively select court corners and net midpoint from a video frame."""
    frame = plt.imread(frame_path)
    figure, axis = plt.subplots()
    axis.imshow(frame)
    corners: list[tuple[float, float]] = []
    for label in ("左上", "右上", "右下", "左下"):
        axis.set_title(f"点击{label}")
        selected = plt.ginput(1, timeout=-1)
        if not selected:
            plt.close(figure)
            raise ValueError("标定取消")
        corners.append((float(selected[0][0]), float(selected[0][1])))
    axis.set_title("点击网线中点")
    selected = plt.ginput(1, timeout=-1)
    if not selected:
        plt.close(figure)
        raise ValueError("标定取消")
    cal = CourtCal(corners=corners, net_mid=(float(selected[0][0]), float(selected[0][1])))
    save_cal(cal, out_path)
    plt.close(figure)
    return cal
