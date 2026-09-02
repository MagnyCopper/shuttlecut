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
