"""Pure optical-flow features for separating camera and local motion."""

import cv2
import numpy as np


def global_shift(prev_gray: np.ndarray, cur_gray: np.ndarray) -> tuple[float, float]:
    """Estimate the dominant frame translation with LK flow and RANSAC."""
    points = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=300, qualityLevel=0.01, minDistance=15
    )
    if points is None or len(points) < 8:
        return 0.0, 0.0
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, points, None)
    if tracked is None or status is None:
        return 0.0, 0.0
    valid = status.ravel().astype(bool)
    previous_points = points.reshape(-1, 2)[valid]
    current_points = tracked.reshape(-1, 2)[valid]
    if len(previous_points) < 8:
        return 0.0, 0.0
    affine, _ = cv2.estimateAffinePartial2D(
        previous_points, current_points, method=cv2.RANSAC
    )
    if affine is None:
        return 0.0, 0.0
    return float(affine[0, 2]), float(affine[1, 2])


def local_flow_mag(
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> float:
    """Return mean Farneback flow magnitude inside a clamped bounding box."""
    frame_height, frame_width = prev_gray.shape[:2]
    x, y, width, height = bbox
    left = max(0, min(x, frame_width))
    top = max(0, min(y, frame_height))
    right = max(left, min(x + width, frame_width))
    bottom = max(top, min(y + height, frame_height))
    if right <= left or bottom <= top:
        return 0.0
    previous_crop = prev_gray[top:bottom, left:right]
    current_crop = cur_gray[top:bottom, left:right]
    flow = cv2.calcOpticalFlowFarneback(
        previous_crop,
        current_crop,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    return float(np.linalg.norm(flow, axis=2).mean())


def local_flow_vec(
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[float, float]:
    """Mean Farneback flow VECTOR inside a clamped box — 方向性位移场均值。"""
    frame_height, frame_width = prev_gray.shape[:2]
    x, y, width, height = bbox
    left = max(0, min(x, frame_width))
    top = max(0, min(y, frame_height))
    right = max(left, min(x + width, frame_width))
    bottom = max(top, min(y + height, frame_height))
    if right <= left or bottom <= top:
        return 0.0, 0.0
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray[top:bottom, left:right], cur_gray[top:bottom, left:right], None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    return float(flow[..., 0].mean()), float(flow[..., 1].mean())


def body_residual(
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
    bbox: tuple[int, int, int, int],
    displacement: tuple[float, float],
) -> float:
    """肢体相对运动残差 = ‖框内平均流向量 − bbox 自身位移向量‖(同在相机系)。

    走动/被跟拍:框内流≈框位移 → 残差≈0;挥拍:肢体相对躯干运动 → 残差高。
    不减全局平移:跟拍时 v≈disp≈0,减 global 反而注入伪残差。"""
    vx, vy = local_flow_vec(prev_gray, cur_gray, bbox)
    rx, ry = vx - displacement[0], vy - displacement[1]
    return float(np.hypot(rx, ry))


def residual_action(
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
) -> float:
    """Return the strongest local flow after subtracting global motion."""
    if not bboxes:
        return 0.0
    tx, ty = global_shift(prev_gray, cur_gray)
    global_magnitude = float(np.hypot(tx, ty))
    return max(
        max(local_flow_mag(prev_gray, cur_gray, bbox) - 0.8 * global_magnitude, 0.0)
        for bbox in bboxes
    )


def box_flow_dispersion(
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> float:
    """框内 Farneback 幅值的 P95−中位数:肢体快动的高分散统计量。

    刚体平移(走动/跟拍/整体移动)→ 框内幅值近乎一致 → 离散度≈0;
    挥拍/蹬跨 → 手臂局部幅值远超躯干 → 高分散。"""
    frame_height, frame_width = prev_gray.shape[:2]
    x, y, width, height = bbox
    left = max(0, min(x, frame_width))
    top = max(0, min(y, frame_height))
    right = max(left, min(x + width, frame_width))
    bottom = max(top, min(y + height, frame_height))
    if right <= left or bottom <= top:
        return 0.0
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray[top:bottom, left:right], cur_gray[top:bottom, left:right], None,
        pyr_scale=0.5, levels=3, winsize=25, iterations=7, poly_n=7, poly_sigma=1.2, flags=0,
    )
    magnitudes = np.linalg.norm(flow, axis=2).ravel()
    if magnitudes.size < 20:
        return 0.0
    return float(np.percentile(magnitudes, 95) - np.median(magnitudes))
