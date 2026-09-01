import json
from dataclasses import dataclass
from pathlib import Path

import torch
from ultralytics import YOLO


@dataclass
class PersonBox:
    cx: float
    cy: float
    w: float
    h: float
    conf: float


@dataclass
class FramePersons:
    t: float
    persons: list[PersonBox]


def resolve_device(pref: str) -> str:
    if pref != "auto":
        return pref
    return "mps" if torch.backends.mps.is_available() else "cpu"


def detect_persons(frame_paths: list[str], frame_fps: float,
                   model_path: str = "models/yolo11n.pt", device: str = "auto",
                   batch: int = 64, out_jsonl: str | None = None) -> list[FramePersons]:
    dev = resolve_device(device)
    model = YOLO(model_path)
    rows: list[FramePersons] = []
    frame_w = frame_h = 0
    for i in range(0, len(frame_paths), batch):
        chunk = frame_paths[i : i + batch]
        results = model.predict(chunk, device=dev, imgsz=1280, classes=[0],
                                conf=0.25, verbose=False)
        for j, r in enumerate(results):
            frame_w, frame_h = int(r.orig_shape[1]), int(r.orig_shape[0])
            persons = []
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                persons.append(PersonBox((x1 + x2) / 2, (y1 + y2) / 2,
                                         x2 - x1, y2 - y1, float(box.conf[0])))
            rows.append(FramePersons(t=(i + j) / frame_fps, persons=persons))
    if out_jsonl:
        Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(out_jsonl, "w") as f:
            f.write(json.dumps({"meta": {"frame_w": frame_w, "frame_h": frame_h,
                                         "fps": frame_fps}}) + "\n")
            for row in rows:
                f.write(json.dumps({
                    "t": row.t,
                    "persons": [vars(p) for p in row.persons],
                }) + "\n")
    return rows


def load_persons_jsonl(path: str) -> tuple[dict, list[FramePersons]]:
    meta: dict = {}
    rows: list[FramePersons] = []
    with open(path) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            if i == 0 and "meta" in d:
                meta = d["meta"]
                continue
            rows.append(FramePersons(
                t=d["t"],
                persons=[PersonBox(**p) for p in d["persons"]],
            ))
    return meta, rows
