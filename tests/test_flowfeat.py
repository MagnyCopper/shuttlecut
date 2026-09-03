import cv2
import numpy as np

from shuttlecut.flowfeat import body_residual, global_shift, local_flow_vec, residual_action


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


def test_local_flow_vec_and_body_residual_cancel_box_translation() -> None:
    # 仅 bbox 内纹理左移 10px、背景静止(全局平移≈0):
    # 位移=(−10,0)(框随内容走) → 残差≈0;位移=(0,0)(躯干不动肢体动) → 残差≈10
    rng = np.random.default_rng(7)
    prev = cv2.GaussianBlur(rng.random((120, 160)).astype(np.float32), (5, 5), 0)
    cur = prev.copy()
    y, x, h, w = 30, 40, 60, 60
    cur[y:y + h, x:x + w] = np.roll(prev[y:y + h, x:x + w], -10, axis=1)
    bbox = (x, y, w, h)
    vx, vy = local_flow_vec(prev, cur, bbox)
    assert -12 <= vx <= -8 and abs(vy) < 3
    r_moved = body_residual(prev, cur, bbox, (-10.0, 0.0))
    r_still = body_residual(prev, cur, bbox, (0.0, 0.0))
    assert r_moved < 3.5
    assert r_still > 7
