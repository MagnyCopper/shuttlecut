from dataclasses import dataclass

import numpy as np

from shuttlecut.detector import FramePersons


@dataclass
class EnergySeries:
    times: list[float]
    values: list[float]


def _large_persons(row: FramePersons, frame_h: float, min_h_ratio: float,
                   roi: tuple[float, float, float, float] | None) -> list[tuple[float, float]]:
    rx, ry, rw, rh = roi if roi else (0.0, 0.0, float("inf"), float("inf"))
    out = []
    for p in row.persons:
        if p.h < min_h_ratio * frame_h:
            continue
        if not (rx <= p.cx <= rx + rw and ry <= p.cy <= ry + rh):
            continue
        out.append((p.cx, p.cy))
    return out


def auto_roi(rows: list[FramePersons], frame_w: float, frame_h: float,
             min_h_ratio: float = 0.12) -> tuple[float, float, float, float]:
    pts = np.array([pt for r in rows for pt in _large_persons(r, frame_h, min_h_ratio, None)])
    if len(pts) == 0:
        return (0.0, 0.0, float(frame_w), float(frame_h))
    x0, y0 = np.percentile(pts[:, 0], 5), np.percentile(pts[:, 1], 5)
    x1, y1 = np.percentile(pts[:, 0], 95), np.percentile(pts[:, 1], 95)
    return (float(x0), float(y0), float(x1 - x0), float(y1 - y0))


def _chamfer(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float((d.min(axis=1).mean() + d.min(axis=0).mean()) / 2)


def motion_energy(rows: list[FramePersons], frame_h: float,
                  roi: tuple[float, float, float, float] | None = None,
                  min_h_ratio: float = 0.12) -> EnergySeries:
    times, values = [], []
    prev: np.ndarray | None = None
    for row in rows:
        cur = np.array(_large_persons(row, frame_h, min_h_ratio, roi))
        times.append(row.t)
        values.append(_chamfer(cur, prev) if prev is not None else 0.0)
        prev = cur
    return EnergySeries(times, values)


def smooth(series: EnergySeries, window_s: float, fps: float) -> EnergySeries:
    k = max(1, int(round(window_s * fps)))
    values = np.array(series.values)
    pad = k // 2
    padded = np.pad(values, (pad, k - 1 - pad), mode="edge")
    v = np.convolve(padded, np.ones(k) / k, mode="valid")
    return EnergySeries(series.times, v.tolist())
