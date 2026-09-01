from shuttlecut.activity import EnergySeries, auto_roi, motion_energy, smooth
from shuttlecut.detector import FramePersons, PersonBox

H = 720.0


def frame(t, pts):
    return FramePersons(t, [PersonBox(x, y, 50, 0.2 * H, 0.9) for x, y in pts])


def test_motion_zero_when_static():
    rows = [frame(0.0, [(100, 400)]), frame(0.2, [(100, 400)])]
    s = motion_energy(rows, H)
    assert s.values[1] == 0.0


def test_motion_displacement():
    rows = [frame(0.0, [(100, 400)]), frame(0.2, [(130, 400)])]
    s = motion_energy(rows, H)
    assert abs(s.values[1] - 30.0) < 1e-6


def test_small_persons_filtered():
    rows = [FramePersons(0.0, [PersonBox(100, 400, 50, 0.2 * H, 0.9),
                                PersonBox(600, 100, 30, 0.05 * H, 0.9)]),
            FramePersons(0.2, [PersonBox(100, 400, 50, 0.2 * H, 0.9),
                                PersonBox(900, 100, 30, 0.05 * H, 0.9)])]
    s = motion_energy(rows, H)
    assert s.values[1] == 0.0


def test_roi_filters_outside():
    rows = [FramePersons(0.0, [PersonBox(100, 400, 50, 0.2 * H, 0.9),
                                PersonBox(1100, 400, 50, 0.2 * H, 0.9)]),
            FramePersons(0.2, [PersonBox(100, 400, 50, 0.2 * H, 0.9),
                                PersonBox(1100, 700, 50, 0.2 * H, 0.9)])]
    s = motion_energy(rows, H, roi=(0, 200, 800, 500))
    assert s.values[1] == 0.0


def test_auto_roi_bounds():
    rows = [frame(0.0, [(200, 300), (900, 600), (640, 360)])]
    x, y, w, h = auto_roi(rows, 1280, H)
    assert 0 <= x < x + w <= 1280 and 0 <= y < y + h <= H


def test_auto_roi_empty_returns_full_frame():
    assert auto_roi([], 1280, H) == (0.0, 0.0, 1280.0, H)


def test_smooth_constant_series_unchanged():
    s = EnergySeries([0.0, 0.2, 0.4], [5.0, 5.0, 5.0])
    sm = smooth(s, window_s=0.4, fps=5)
    assert sm.values == [5.0, 5.0, 5.0]


def test_smooth_edge_padding_behavior():
    # k=3(fps=5, window_s=0.6 取整)时边缘用 edge 填充:[1,2,3] → [4/3, 2.0, 8/3]
    s = EnergySeries([0.0, 0.2, 0.4], [1.0, 2.0, 3.0])
    sm = smooth(s, window_s=0.6, fps=5)
    assert abs(sm.values[0] - 4 / 3) < 1e-9
    assert abs(sm.values[1] - 2.0) < 1e-9
    assert abs(sm.values[2] - 8 / 3) < 1e-9
