import json

import cv2
import numpy as np

from shuttlecut.detector import FramePersons
from shuttlecut.flowpipe import bbox_of, load_flow, run_flow


def test_bbox_of_converts_center_box_to_integer_corners() -> None:
    assert bbox_of(type("P", (), {"cx": 20.5, "cy": 15.5, "w": 9.0, "h": 7.0})()) == (16, 12, 9, 7)


def test_run_flow_writes_thirty_rows_and_loads_roundtrip(tmp_path, monkeypatch) -> None:
    video = str(tmp_path / "synth.mp4")
    writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (96, 64))
    for index in range(300):
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        cv2.rectangle(frame, (index % 70, 20), (index % 70 + 20, 50), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()

    monkeypatch.setattr("shuttlecut.flowpipe.detect_persons", lambda frames, frame_fps, device: [
        FramePersons(t=index / frame_fps, persons=[]) for index in range(len(frames))
    ])
    output = str(tmp_path / "flow.jsonl")
    rows = run_flow(video, output, device="cpu", max_frames=30)

    assert len(rows) == 30
    assert all(set(row) == {"t", "residual", "n_persons"} for row in rows)
    assert all(row["residual"] >= 0 and np.isfinite(row["residual"]) for row in rows)
    assert all(isinstance(row["n_persons"], int) for row in rows)
    assert all(abs(rows[i]["t"] - i / 15) < 0.001 for i in range(30))
    with open(output) as handle:
        assert [json.loads(line) for line in handle] == rows
    assert load_flow(output) == rows
