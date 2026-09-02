"""Cached frame extraction, person detection, and residual-flow pipeline."""

import json
from pathlib import Path
from typing import TypedDict

import cv2

from shuttlecut.detector import PersonBox, detect_persons
from shuttlecut.flowfeat import residual_action
from shuttlecut.sampler import extract_frames


class FlowRow(TypedDict):
    t: float
    residual: float
    n_persons: int


def bbox_of(person: PersonBox) -> tuple[int, int, int, int]:
    """Convert a centered person box to integer top-left dimensions."""
    return (
        int(person.cx - person.w / 2),
        int(person.cy - person.h / 2),
        int(person.w),
        int(person.h),
    )


def run_flow(
    video: str,
    out_jsonl: str,
    fps: float = 15.0,
    width: int = 960,
    device: str = "auto",
    max_frames: int | None = None,
) -> list[FlowRow]:
    """Run person-guided residual optical flow and persist one row per frame."""
    stem = Path(video).stem
    frame_paths = extract_frames(video, f"temp/work/{stem}/frames15", fps=fps, width=width)
    if max_frames is not None:
        frame_paths = frame_paths[:max_frames]
    detections = detect_persons(frame_paths, frame_fps=fps, device=device)
    rows: list[FlowRow] = []
    previous = None
    for frame_path, detection in zip(frame_paths, detections, strict=True):
        current = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if current is None:
            raise OSError(f"Unable to read extracted frame: {frame_path}")
        frame_height = current.shape[0]
        selected = [p for p in detection.persons if p.h >= 0.10 * frame_height]
        selected.sort(key=lambda p: p.w * p.h, reverse=True)
        bboxes = [bbox_of(p) for p in selected[:4]]
        value = 0.0 if previous is None else residual_action(previous, current, bboxes)
        rows.append({"t": round(detection.t, 3), "residual": round(value, 2), "n_persons": int(len(detection.persons))})
        previous = current
    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return rows


def load_flow(path: str) -> list[FlowRow]:
    """Load flow rows from JSONL."""
    with open(path) as handle:
        return [json.loads(line) for line in handle]
