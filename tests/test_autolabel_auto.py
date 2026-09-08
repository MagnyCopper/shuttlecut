import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "autolabel"))

from auto import pick_indices, segment_frames  # noqa: E402


def test_pick_indices_uniform_and_bounded():
    assert pick_indices(0) == []
    assert pick_indices(5) == [0, 1, 2, 3, 4]
    idx = pick_indices(100)
    assert len(idx) == 9 and idx[0] == 0 and idx[-1] == 99
    assert all(b - a >= 10 for a, b in zip(idx, idx[1:]))  # 均匀覆盖
    assert all(0 <= i < 100 for i in idx)


def test_segment_frames_fps_convert():
    assert segment_frames([(0.0, 1.0), (10.5, 12.2)]) == [(0, 15), (158, 183)]


def test_pick_indices_single_frame():
    assert pick_indices(1) == [0]
