from dataclasses import dataclass

import numpy as np

from shuttlecut.segmenter import Rally

_IDLE, _ARMED, _RALLY = 0, 1, 2
_AUDIO_WINDOW_S = 0.25
_SERVE_BACKTRACK_S = 0.5
_END_BACKTRACK_S = 0.5


@dataclass
class ArmedParams:
    arm_s: float = 1.0
    confirm_s: float = 3.0
    wrist_mad: float = 3.0
    wrist_mad_audio: float = 1.5
    end_gap_s: tuple[float, float] = (0.4, 1.2)
    min_rally_s: float = 1.5
    min_idle_s: float = 1.2


def _wrist_z(features: list[dict]) -> np.ndarray:
    """腕速按中位数/MAD 自适应归一;MAD 退化(<1e-6)时用 1.0。"""
    peaks = np.asarray([f["wrist_peak"] for f in features], dtype=float)
    median = float(np.median(peaks))
    mad = float(np.median(np.abs(peaks - median)))
    if mad < 1e-6:
        mad = 1.0
    return (peaks - median) / mad


def _is_event(z: float, feature: dict, transients: list[float],
              params: ArmedParams) -> bool:
    """动作事件:高腕速无音频共现,或音频共现时降阈触发。"""
    near_audio = any(abs(tr - feature["t"]) <= _AUDIO_WINDOW_S for tr in transients)
    if z >= params.wrist_mad and not near_audio:
        return True
    return z >= params.wrist_mad_audio and near_audio


def segment_armed(features: list[dict], transients: list[float],
                  params: ArmedParams = ArmedParams()) -> list[Rally]:
    """IDLE→ARMED(双侧就位 arm_s)→RALLY(腕速±音频事件),结束谷收段并回溯边界。"""
    if not features:
        return []
    z = _wrist_z(features)
    state = _IDLE
    arm_run_start: float | None = None
    armed_since = 0.0
    start = last_event_t = 0.0
    collapse_seen = False
    raw: list[tuple[float, float]] = []

    for zi, feat in zip(z, features):
        t = feat["t"]
        both_sides = min(feat["n_by_side"]) >= 1
        # 就位游程跨状态统一维护:任一侧离场即清零,重新累计 arm_s
        if both_sides:
            arm_run_start = t if arm_run_start is None else arm_run_start
        else:
            arm_run_start = None

        if state == _IDLE:
            if arm_run_start is not None and t - arm_run_start >= params.arm_s:
                state, armed_since = _ARMED, t
        elif state == _ARMED:
            if _is_event(zi, feat, transients, params):
                state = _RALLY
                start = max(0.0, t - _SERVE_BACKTRACK_S)
                last_event_t, collapse_seen = t, False
            elif not both_sides:
                state = _IDLE  # 就位丢失
            elif t - armed_since > params.confirm_s:
                state = _IDLE  # 确认超时;就位仍在则下一帧重新 ARMED
        else:  # RALLY:新事件刷新 last_event_t(半场交替语义由 n_by_side 双侧保证)
            if _is_event(zi, feat, transients, params):
                last_event_t, collapse_seen = t, False
                continue
            if not feat["any_ready"] or min(feat["n_by_side"]) == 0:
                collapse_seen = True
            if collapse_seen and t - last_event_t >= params.end_gap_s[1]:
                raw.append((start, last_event_t + _END_BACKTRACK_S))
                state = _IDLE

    if state == _RALLY:  # 特征流结束仍未现结束谷:按同一回溯收段
        raw.append((start, last_event_t + _END_BACKTRACK_S))

    kept = [(s, e) for s, e in raw if e - s >= params.min_rally_s]
    merged: list[tuple[float, float]] = []
    i = 0
    while i < len(kept):  # 只合并一次相邻对,不递归长合并
        if i + 1 < len(kept) and kept[i + 1][0] - kept[i][1] < params.min_idle_s:
            merged.append((kept[i][0], kept[i + 1][1]))
            i += 2
        else:
            merged.append(kept[i])
            i += 1

    peaks = np.asarray([f["wrist_peak"] for f in features], dtype=float)
    times = np.asarray([f["t"] for f in features], dtype=float)
    rallies: list[Rally] = []
    for s, e in merged:
        inside = (times >= s) & (times <= e)
        peak = float(peaks[inside].max()) if inside.any() else 0.0
        hits = sum(1 for tr in transients if s <= tr <= e)
        rallies.append(Rally(start=s, end=e, motion_peak=peak,
                             confidence=1.0, hits=hits))
    return rallies
