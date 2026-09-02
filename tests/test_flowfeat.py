import cv2
import numpy as np

from shuttlecut.flowfeat import global_shift, residual_action


def _texture() -> np.ndarray:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (180, 240), dtype=np.uint8)
    return cv2.GaussianBlur(image, (5, 5), 0)


def test_global_translation_is_estimated_and_removed() -> None:
    previous = _texture()
    current = np.roll(previous, 12, axis=1)

    tx, ty = global_shift(previous, current)

    assert 10 <= abs(tx) <= 14
    assert abs(ty) < 2
    assert residual_action(previous, current, [(30, 30, 100, 100)]) < 4


def test_local_motion_remains_after_global_motion_is_removed() -> None:
    previous = _texture()
    current = previous.copy()
    x, y, width, height = 60, 50, 80, 80
    current[y : y + height, x + 8 : x + width] = previous[y : y + height, x : x + width - 8]
    current[y : y + height, x : x + 8] = previous[y : y + height, x + width - 8 : x + width]

    tx, ty = global_shift(previous, current)

    assert abs(tx) < 2
    assert abs(ty) < 2
    assert 5 <= residual_action(previous, current, [(x, y, width, height)]) <= 11


def test_static_frames_have_no_residual_action() -> None:
    previous = _texture()

    assert residual_action(previous, previous.copy(), [(20, 20, 100, 100)]) < 1.5
    assert residual_action(previous, previous, []) == 0.0
