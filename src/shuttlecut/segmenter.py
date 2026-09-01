from dataclasses import dataclass, field

import numpy as np

from shuttlecut.activity import EnergySeries


@dataclass
class SegParams:
    hi_q: float = 0.60
    lo_q: float = 0.30
    min_rally_s: float = 3.0
    min_idle_s: float = 2.5
    max_rally_s: float = 120.0
    smooth_s: float = 2.0
    pre_roll_s: float = 1.5
    post_roll_s: float = 2.0


@dataclass
class Rally:
    start: float
    end: float
    motion_peak: float
    confidence: float
    hits: int = 0


def _raw_segments(values: np.ndarray, times: list[float],
                  hi: float, lo: float) -> list[tuple[float, float, float]]:
    """滞回扫描:>=hi 进入,<=lo 退出。返回 (start, end, peak)。"""
    segs: list[tuple[float, float, float]] = []
    in_rally = False
    start = peak = 0.0
    for t, v in zip(times, values):
        if not in_rally and v >= hi:
            in_rally, start, peak = True, t, float(v)
        elif in_rally:
            peak = max(peak, float(v))
            if v <= lo:
                segs.append((start, t, peak))
                in_rally = False
    if in_rally:
        segs.append((start, times[-1], peak))
    return segs


def _merge_by_min_idle(segs: list[tuple[float, float, float]],
                       min_idle_s: float) -> list[tuple[float, float, float]]:
    merged: list[tuple[float, float, float]] = []
    for seg in segs:
        if merged and seg[0] - merged[-1][1] < min_idle_s:
            s0, _, p0 = merged[-1]
            merged[-1] = (s0, seg[1], max(p0, seg[2]))
        else:
            merged.append(seg)
    return merged


def _split_long(seg: tuple[float, float, float], max_rally_s: float,
                values: np.ndarray, times: list[float]) -> list[tuple[float, float, float]]:
    s0, s1, peak = seg
    if s1 - s0 <= max_rally_s:
        return [seg]
    # 在中点附近找能量最低的切分点
    mid = (s0 + s1) / 2
    mask = (np.array(times) >= mid - 5) & (np.array(times) <= mid + 5)
    idx = np.where(mask)[0]
    cut_t = float(times[idx[np.argmin(values[idx])]])
    left, right = (s0, cut_t, peak), (cut_t, s1, peak)
    return _split_long(left, max_rally_s, values, times) + _split_long(right, max_rally_s, values, times)


def segment(series: EnergySeries, params: SegParams) -> list[Rally]:
    values = np.array(series.values)
    hi, lo = float(np.percentile(values, params.hi_q * 100)), float(np.percentile(values, params.lo_q * 100))
    if hi <= lo and float(np.min(values)) == float(np.max(values)):
        return []
    if hi <= lo:
        hi, lo = float(np.max(values)), float(np.min(values))
    segs = _raw_segments(values, series.times, hi, lo)
    segs = [s for s in segs if s[1] - s[0] >= params.min_rally_s]
    segs = _merge_by_min_idle(segs, params.min_idle_s)
    segs = [s for s in segs if s[1] - s[0] >= params.min_rally_s]
    segs = [p for s in segs for p in _split_long(s, params.max_rally_s, values, series.times)]
    med = float(np.median(values))
    return [Rally(start=s, end=e, motion_peak=p,
                  confidence=float(np.clip((p - lo) / max(hi - lo, 1e-9), 0.0, 1.0))
                  )
            for s, e, p in segs]
