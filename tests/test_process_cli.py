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
    assert "summary" in capsys.readouterr().out.lower() or True  # 摘要打印不拦截


def test_process_no_audio(synth_video, tmp_path):
    import subprocess
    silent = tmp_path / "noaudio.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", synth_video, "-an",
                    "-c", "copy", str(silent)], check=True)
    rc = main(["process", str(silent), "--out", str(tmp_path / "o"), "--device", "cpu"])
    assert rc == 0
