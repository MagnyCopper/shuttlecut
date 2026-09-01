from shuttlecut.exporter import export_clips, export_reel
from shuttlecut.sampler import probe
from shuttlecut.segmenter import Rally


def test_export_clips_duration(synth_video, tmp_path):
    rallies = [Rally(3.0, 9.0, 9.0, 0.9), Rally(12.0, 17.0, 9.0, 0.8)]
    clips = export_clips(synth_video, rallies, str(tmp_path / "clips"))
    assert len(clips) == 2
    m = probe(clips[0])
    # 期望 (9+2) - max(0, 3-1.5) = 8.5s,容差 0.5
    assert abs(m.duration_s - 8.5) < 0.5


def test_export_clip_head_clamped(synth_video, tmp_path):
    clips = export_clips(synth_video, [Rally(0.5, 4.0, 9.0, 0.9)], str(tmp_path / "c2"))
    assert abs(probe(clips[0]).duration_s - (4.0 + 2.0)) < 0.5  # start-pre 截到 0


def test_export_reel(synth_video, tmp_path):
    clips = export_clips(synth_video, [Rally(1.0, 5.0, 9.0, 0.9),
                                       Rally(8.0, 12.0, 9.0, 0.9)], str(tmp_path / "c3"))
    reel = export_reel(clips, str(tmp_path / "c3" / "highlights.mp4"))
    m = probe(reel)
    assert m.duration_s > 8.0
