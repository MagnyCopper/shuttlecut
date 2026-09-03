"""Cached frame extraction, person detection, and body-residual flow pipeline."""

import json
import os
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np

from shuttlecut.detector import PersonBox, detect_persons, load_persons_jsonl
from shuttlecut.flowfeat import body_residual
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


def _frames_or_extract(video: str, fps: float, width: int) -> list[str]:
    """帧目录缓存:meta 匹配且帧数符合预期时免重复抽帧。"""
    stem = Path(video).stem
    d = Path(f"temp/work/{stem}/frames15")
    meta_path = d / "meta.json"
    from shuttlecut.sampler import probe
    expected = int(probe(video).duration_s * fps)
    key = {"video": video, "mtime": os.path.getmtime(video), "fps": fps, "width": width}
    if meta_path.exists() and json.loads(meta_path.read_text()) == key:
        frames = sorted(str(p) for p in d.glob("frame_*.jpg"))
        if abs(len(frames) - expected) <= 2:
            print(f"[cache] 复用帧缓存 {d}({len(frames)} 帧)")
            return frames
    frames = extract_frames(video, str(d), fps=fps, width=width)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(key))
    return frames


def _detect_or_load(frame_paths: list[str], fps: float, device: str, video: str) -> list:
    """15fps 检测缓存(temp/work/<stem>/persons15.jsonl,键 mtime):重算光流时不重复检测。"""
    stem = Path(video).stem
    cache = Path(f"temp/work/{stem}/persons15.jsonl")
    meta_path = Path(f"temp/work/{stem}/persons15.meta.json")
    key = {"video": video, "mtime": os.path.getmtime(video), "fps": fps, "n": len(frame_paths)}
    if cache.exists() and meta_path.exists() and json.loads(meta_path.read_text()) == key:
        _, rows = load_persons_jsonl(str(cache))
        print(f"[cache] 复用 15fps 检测缓存 {cache}")
        return rows
    rows = detect_persons(frame_paths, frame_fps=fps, device=device, out_jsonl=str(cache))
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(key))
    return rows
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
    frame_paths = _frames_or_extract(video, fps, width)
    if max_frames is not None:
        frame_paths = frame_paths[:max_frames]
    detections = _detect_or_load(frame_paths, fps, device, video)
    rows: list[FlowRow] = []
    previous_gray = None
    prev_players: list[tuple[float, float, tuple[int, int, int, int]]] = []  # (cx, cy, bbox)
    for frame_path, detection in zip(frame_paths, detections, strict=True):
        current = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if current is None:
            raise OSError(f"Unable to read extracted frame: {frame_path}")
        frame_height = current.shape[0]
        selected = [p for p in detection.persons if p.h >= 0.10 * frame_height]
        selected.sort(key=lambda p: p.w * p.h, reverse=True)
        cur_players = [(p.cx, p.cy, bbox_of(p)) for p in selected[:4]]
        value = 0.0
        if previous_gray is not None:
            # 最近邻配对(≤150px),残差=框内流向量−bbox位移−全局平移
            for cx, cy, bbox in cur_players:
                if not prev_players:
                    break
                px, py, _ = min(prev_players, key=lambda q: (q[0] - cx) ** 2 + (q[1] - cy) ** 2)
                if (px - cx) ** 2 + (py - cy) ** 2 <= 150 ** 2:
                    r = body_residual(previous_gray, current, bbox, (cx - px, cy - py))
                    value = max(value, r)
        rows.append({"t": round(detection.t, 3), "residual": round(value, 2), "n_persons": int(len(detection.persons))})
        previous_gray = current
        prev_players = cur_players
    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return rows


def load_flow(path: str) -> list[FlowRow]:
    """Load flow rows from JSONL."""
    with open(path) as handle:
        return [json.loads(line) for line in handle]
