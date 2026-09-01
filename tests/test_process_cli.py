import json

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
