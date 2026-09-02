from dataclasses import dataclass
from typing import Final, TypedDict

import numpy as np

from shuttlecut.court import CourtCal, side_of, to_court_xy
from shuttlecut.poses import FramePose, PersonKps

_ANKLES: Final = (15, 16)
_HIPS: Final = (11, 12)
_KNEES: Final = (13, 14)
_ARMS: Final = (7, 8, 9, 10)


@dataclass(frozen=True, slots=True)
class FramePlayers:
    t: float
    players: list[tuple[PersonKps, int]]


class FrameFeatures(TypedDict):
    t: float
    n_by_side: tuple[int, int]
    wrist_peak: float
    any_ready: bool


def _mean_point(kps: np.ndarray, indices: tuple[int, ...]) -> tuple[float, float] | None:
    points = kps[list(indices)]
    valid = np.isfinite(points).all(axis=1)
    if not valid.any():
        return None
    mean = points[valid].mean(axis=0)
    return (float(mean[0]), float(mean[1]))


def court_players(
    poses: list[FramePose], cal: CourtCal, frame_wh: tuple[int, int]
) -> list[FramePlayers]:
    """Select up to four in-court people, preferring those nearest the net."""
    del frame_wh
    result: list[FramePlayers] = []
    for frame in poses:
        candidates: list[tuple[float, PersonKps, int]] = []
        for person in frame.persons:
            foot = _mean_point(person.kps, _ANKLES) or _mean_point(person.kps, _HIPS)
            if foot is None:
                continue
            court_x, court_y = to_court_xy(cal, *foot)
            if -0.15 <= court_x <= 3.20 and -0.15 <= court_y <= 13.55:
                candidates.append((abs(court_y - 6.7), person, side_of(cal, *foot)))
        candidates.sort(key=lambda candidate: candidate[0])
        result.append(FramePlayers(frame.t, [(person, side) for _, person, side in candidates[:4]]))
    return result


def wrist_speed(cur: PersonKps, prev: PersonKps, dt: float = 0.2) -> float:
    """Return the fastest valid wrist/elbow displacement between two frames."""
    displacements: list[float] = []
    for index in _ARMS:
        current = cur.kps[index]
        previous = prev.kps[index]
        if np.isfinite(current).all() and np.isfinite(previous).all():
            displacements.append(float(np.linalg.norm(current - previous) / dt))
    return max(displacements, default=0.0)


def stance_ready(p: PersonKps) -> bool:
    """Detect a broad, flexed stance from hip, knee, and ankle heights."""
    hip = _mean_point(p.kps, _HIPS)
    knee = _mean_point(p.kps, _KNEES)
    ankle = _mean_point(p.kps, _ANKLES)
    if hip is None or knee is None or ankle is None:
        return False
    return hip[1] > knee[1] and hip[1] - knee[1] >= 0.12 * abs(ankle[1] - hip[1])


def frame_features(
    poses: list[FramePose], cal: CourtCal, frame_wh: tuple[int, int]
) -> list[FrameFeatures]:
    """Calculate occupancy, peak arm speed, and readiness for each frame."""
    selected = court_players(poses, cal, frame_wh)
    features: list[FrameFeatures] = []
    previous: list[PersonKps] = []
    for index, frame in enumerate(selected):
        by_side = (sum(side == 0 for _, side in frame.players), sum(side == 1 for _, side in frame.players))
        peak = 0.0
        if index > 0:
            peak = max((wrist_speed(person, old) for (person, _), old in zip(frame.players, previous)), default=0.0)
        previous = [person for person, _ in frame.players]
        features.append({"t": frame.t, "n_by_side": by_side, "wrist_peak": peak, "any_ready": any(stance_ready(person) for person, _ in frame.players)})
    return features
