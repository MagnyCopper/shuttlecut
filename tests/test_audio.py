import subprocess

from shuttlecut.audio import Transient, audio_transients


def test_audio_transients_runs(tmp_path):
    wav = tmp_path / "a.wav"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=2", "-ar", "16000", str(wav)],
        check=True,
    )
    ts = audio_transients(str(wav))
    assert isinstance(ts, list)
    assert all(isinstance(t, Transient) for t in ts)
    assert all(t.t >= 0.0 for t in ts)
