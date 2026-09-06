"""Tests for temporal segmentation logic (no GPU/model needed)."""
import cv2
import numpy as np

from shuttlecut.temporal import (
    _median,
    build_diffs,
    segment_curve,
    split_merged,
    two_scale_segments,
)


def _curve(segs, total=100.0, dt=0.2, high=0.95, low=0.02):
    t = np.arange(0, total, dt)
    p = np.full_like(t, low)
    for a, b in segs:
        p[(t >= a) & (t <= b)] = high
    return t, p


def test_segment_curve_finds_blocks() -> None:
    t, p = _curve([(10, 20), (40, 50)])
    out = segment_curve(t, p, hi=0.5, lo=0.1, min_len_s=1.5)
    assert [(round(a), round(b)) for a, b in out] == [(10, 20), (40, 50)]


def test_segment_curve_drops_short_blocks() -> None:
    t, p = _curve([(10, 11), (40, 50)])  # 1s block < min_len 1.5
    out = segment_curve(t, p, hi=0.5, lo=0.1, min_len_s=1.5)
    assert len(out) == 1 and abs(out[0][0] - 40) < 0.6


def test_split_merged_cuts_at_deep_valley() -> None:
    # 合并段 10-30,细曲线在 20 处有 0.05 深谷、两侧高原
    t = np.arange(10, 30, 0.2)
    p = np.where(np.abs(t - 20) < 0.6, 0.05, 0.9)
    out = split_merged(t, p, 10.0, 30.0, th=0.2, min_sub_s=1.5)
    assert len(out) == 2
    assert abs(out[0][1] - 20) < 1.2 and abs(out[1][0] - 20) < 1.2


def test_split_merged_ignores_shallow_dip() -> None:
    t = np.arange(10, 30, 0.2)
    p = np.where(np.abs(t - 20) < 0.6, 0.5, 0.9)  # 浅谷 0.5 > th 0.2
    out = split_merged(t, p, 10.0, 30.0, th=0.2, min_sub_s=1.5)
    assert len(out) == 1


def test_two_scale_without_fine_returns_base() -> None:
    t, p = _curve([(5, 15)])
    out = two_scale_segments(t, p, None, None)
    assert len(out) == 1 and abs(out[0][0] - 5) < 0.6


def test_median_smooths_spike() -> None:
    x = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    assert _median(x, 3)[2] == 0.0


def test_build_diffs_static_frames_zero_and_moving_nonzero() -> None:
    rng = np.random.default_rng(3)
    base = cv2.GaussianBlur(rng.integers(0, 256, (90, 160)).astype(np.uint8), (5, 5), 0)
    static = np.stack([base] * 4)
    assert build_diffs(static)[1:].max() < 2.0
    moved = np.stack([base, np.roll(base, -12, axis=1), np.roll(base, -12, axis=1), base])
    assert build_diffs(moved)[1:].max() > 5.0
