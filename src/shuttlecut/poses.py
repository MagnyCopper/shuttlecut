import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TORCH_HOME", "models/rtmlib")

import cv2


@dataclass(frozen=True, slots=True)
class PersonKps:
    kps: list[list[float]]
    score: float


@dataclass(frozen=True, slots=True)
class FramePose:
    t: float
    persons: list[PersonKps]


_BODY = None


def estimate_poses(
    frames: list[str],
    out_jsonl: str | None = None,
    mode: str = "lightweight",
) -> list[FramePose]:
    """Estimate COCO-17 keypoints for frames and optionally cache them as JSONL."""
    global _BODY
    if _BODY is None:
        from rtmlib import Body

        _BODY = Body(mode=mode, device="cpu")

    rows: list[FramePose] = []
    for index, frame_path in enumerate(frames):
        image = cv2.imread(frame_path)
        keypoints, scores = _BODY(image)
        persons = [
            PersonKps(
                kps=keypoints[person_index].tolist(),
                score=float(scores[person_index].mean()),
            )
            for person_index in range(len(keypoints))
        ]
        rows.append(FramePose(t=index / 5.0, persons=persons))

    if out_jsonl is not None:
        path = Path(out_jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            for row in rows:
                json.dump(
                    {
                        "t": row.t,
                        "persons": [
                            {"kps": person.kps, "score": person.score}
                            for person in row.persons
                        ],
                    },
                    handle,
                )
                handle.write("\n")
    return rows


def load_poses_jsonl(path: str) -> list[FramePose]:
    """Load frame poses from the JSONL cache format."""
    rows: list[FramePose] = []
    with Path(path).open() as handle:
        for line in handle:
            data = json.loads(line)
            rows.append(
                FramePose(
                    t=float(data["t"]),
                    persons=[
                        PersonKps(
                            kps=person["kps"],
                            score=float(person["score"]),
                        )
                        for person in data["persons"]
                    ],
                )
            )
    return rows
