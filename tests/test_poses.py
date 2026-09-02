from pathlib import Path

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
            assert len(person.kps) == 17
            assert all(len(point) == 2 for point in person.kps)
    assert load_poses_jsonl(str(tmp_path / "poses.jsonl")) == poses
