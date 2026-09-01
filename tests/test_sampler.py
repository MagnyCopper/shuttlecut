from shuttlecut.sampler import extract_audio, extract_frames, probe


def test_probe(synth_video):
    meta = probe(synth_video)
    assert 19.5 < meta.duration_s < 20.5
    assert (meta.width, meta.height) == (1280, 720)
    assert abs(meta.fps - 30.0) < 0.1


def test_extract_frames(synth_video, tmp_path):
    frames = extract_frames(synth_video, str(tmp_path), fps=5.0, width=1280)
    assert 98 <= len(frames) <= 101
    assert frames[0].endswith("frame_000001.jpg")


def test_extract_frames_window(synth_video, tmp_path):
    frames = extract_frames(synth_video, str(tmp_path / "w"), fps=5.0, width=1280,
                            t_start=2.0, t_end=6.0)
    assert 15 <= len(frames) <= 25


def test_extract_frames_clears_previous_frames(synth_video, tmp_path):
    outdir = str(tmp_path / "shared")
    extract_frames(synth_video, outdir, fps=5.0, width=1280, t_start=0.0, t_end=6.0)
    frames = extract_frames(synth_video, outdir, fps=5.0, width=1280,
                            t_start=10.0, t_end=12.0)
    assert 5 <= len(frames) <= 15


def test_extract_audio(synth_video, tmp_path):
    wav = extract_audio(synth_video, str(tmp_path / "a.wav"))
    import soundfile as sf
    info = sf.info(wav)
    assert info.samplerate == 16000 and info.channels == 1
