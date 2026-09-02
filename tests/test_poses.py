import json
from pathlib import Path

import numpy as np

from shuttlecut.poses import estimate_poses, load_poses_jsonl
from shuttlecut.sampler import extract_frames


def test_estimate_poses_returns_timed_keypoints_and_roundtrips_jsonl(
    synth_video: str, tmp_path: Path,
) -> None:
    frames = extract_frames(synth_video, str(tmp_path / "frames"), fps=1.0)[:3]

    poses = estimate_poses(frames, str(tmp_path / "poses.jsonl"))

    assert len(poses) == 3
    assert [pose.t for pose in poses] == [0.0, 0.2, 0.4]
    for pose in poses:
        for person in pose.persons:
            assert isinstance(person.kps, np.ndarray)
            assert person.kps.shape == (17, 2)

    jsonl_lines = (tmp_path / "poses.jsonl").read_text().splitlines()
    assert len(jsonl_lines) == 3
    for line in jsonl_lines:
        data = json.loads(line)
        assert set(data) == {"t", "persons"}
        assert isinstance(data["t"], float)
        assert isinstance(data["persons"], list)
        for person in data["persons"]:
            assert set(person) == {"kps", "score"}
            assert isinstance(person["kps"], list)
            assert len(person["kps"]) == 17
            assert all(
                isinstance(point, list)
                and len(point) == 2
                and all(isinstance(value, (int, float)) for value in point)
                for point in person["kps"]
            )
            assert isinstance(person["score"], float)

    loaded = load_poses_jsonl(str(tmp_path / "poses.jsonl"))
    assert [pose.t for pose in loaded] == [pose.t for pose in poses]
    for loaded_pose, pose in zip(loaded, poses):
        for loaded_person, person in zip(loaded_pose.persons, pose.persons):
            assert np.array_equal(loaded_person.kps, person.kps)
