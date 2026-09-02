import json
from pathlib import Path

from shuttlecut.cli import main


def test_process_end_to_end(synth_video, tmp_path, capsys):
    rc = main(["process", synth_video, "--out", str(tmp_path), "--device", "cpu",
               "--no-reel"])
    assert rc == 0
    out = tmp_path / "synth"
    assert (out / "persons.jsonl").exists()
    data = json.loads((out / "rallies.json").read_text())
    assert data["video"] == "synth"
    assert isinstance(data["rallies"], list)  # testsrc 无真实回合,允许空
    # 自动 ROI 路径:testsrc 无人 → auto_roi 空输入返回全画面,roi.txt 与 params 一致
    roi_content = [float(v) for v in (out / "roi.txt").read_text().split(",")]
    assert roi_content == [0.0, 0.0, 1280.0, 720.0]
    assert data["params"]["roi"] == roi_content
    assert "[summary]" in capsys.readouterr().out


def test_process_manual_roi_writes_file_and_params(synth_video, tmp_path, capsys):
    roi = "100,100,800,500"
    rc = main(["process", synth_video, "--out", str(tmp_path), "--roi", roi,
               "--device", "cpu", "--no-reel"])
    assert rc == 0
    out = tmp_path / "synth"
    assert (out / "roi.txt").read_text() == roi
    data = json.loads((out / "rallies.json").read_text())
    assert data["params"]["roi"] == [100.0, 100.0, 800.0, 500.0]
    capsys.readouterr()


def test_process_no_audio(synth_video, tmp_path):
    import subprocess
    silent = tmp_path / "noaudio.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", synth_video, "-an",
                    "-c", "copy", str(silent)], check=True)
    rc = main(["process", str(silent), "--out", str(tmp_path / "o"), "--device", "cpu"])
    assert rc == 0


def _write_court(out: Path) -> None:
    cal = {"corners": [[200, 200], [1000, 200], [1200, 600], [100, 600]],
           "net_mid": [650, 400]}
    (out / "court.json").write_text(json.dumps(cal))


def test_process_pose_requires_court(synth_video, tmp_path, capsys):
    rc = main(["process", synth_video, "--pose", "--out", str(tmp_path), "--no-reel"])
    assert rc == 1
    assert "shuttlecut calibrate" in capsys.readouterr().out


def test_process_pose_uses_cached_poses(synth_video, tmp_path, capsys):
    import os
    out = tmp_path / "synth"
    out.mkdir(parents=True)
    _write_court(out)
    (out / "poses.jsonl").write_text(
        "".join(json.dumps({"t": i * 0.2, "persons": []}) + "\n" for i in range(3)))
    key = {"video": synth_video, "mtime": os.path.getmtime(synth_video),
           "fps": 5.0, "width": 1280, "mode": "pose"}
    (out / "cache.json").write_text(json.dumps(key))
    rc = main(["process", synth_video, "--pose", "--out", str(tmp_path), "--no-reel"])
    assert rc == 0
    assert "复用姿态缓存" in capsys.readouterr().out
    data = json.loads((out / "rallies.json").read_text())
    assert data["params"]["mode"] == "pose"
    assert data["rallies"] == []


def test_process_pose_estimates_and_caches(synth_video, tmp_path, monkeypatch):
    import os
    from shuttlecut import cli
    from shuttlecut.poses import FramePose
    out = tmp_path / "synth"
    out.mkdir(parents=True)
    _write_court(out)

    def fake_estimate(frames, out_jsonl=None, mode="lightweight"):
        rows = [{"t": i * 0.2, "persons": []} for i in range(len(frames))]
        if out_jsonl is not None:
            Path(out_jsonl).write_text(
                "".join(json.dumps(r) + "\n" for r in rows))
        return [FramePose(r["t"], []) for r in rows]

    monkeypatch.setattr(cli, "estimate_poses", fake_estimate)
    rc = main(["process", synth_video, "--pose", "--out", str(tmp_path), "--no-reel"])
    assert rc == 0
    assert (out / "poses.jsonl").exists()
    cache = json.loads((out / "cache.json").read_text())
    assert cache["mode"] == "pose"
    assert cache["mtime"] == os.path.getmtime(synth_video)


def test_calibrate_frame_beyond_duration(synth_video, tmp_path, capsys):
    rc = main(["calibrate", synth_video, "--frame", "9999", "--out", str(tmp_path)])
    assert rc == 1
    assert "抽帧" in capsys.readouterr().out


def test_calibrate_help_exits_zero():
    import pytest
    with pytest.raises(SystemExit) as exc:
        main(["calibrate", "--help"])
    assert exc.value.code == 0