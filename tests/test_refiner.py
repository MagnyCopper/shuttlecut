import numpy as np
import soundfile as sf

from shuttlecut.refiner import Transient, audio_transients, refine
from shuttlecut.segmenter import Rally

SR = 16000


def click_wav(path, times, duration=12.0):
    n = int(duration * SR)
    y = np.random.default_rng(0).normal(0, 0.003, n)
    for t in times:
        i = int(t * SR)
        w = min(200, n - i)
        env = np.exp(-np.linspace(0, 12, w))
        y[i : i + w] += 0.7 * env * np.sin(2 * np.pi * np.linspace(3e3, 1e3, w))
    sf.write(path, y, SR)


def test_audio_transients_finds_clicks(tmp_path):
    wav = str(tmp_path / "c.wav")
    click_wav(wav, [3.0, 4.0, 5.5])
    ts = audio_transients(wav)
    assert len(ts) == 3
    assert all(abs(t.t - e) < 0.15 for t, e in zip(ts, [3.0, 4.0, 5.5]))


def test_min_gap_dedup(tmp_path):
    wav = str(tmp_path / "c.wav")
    click_wav(wav, [3.0, 3.1])
    assert len(audio_transients(wav)) == 1


def test_refine_pulls_start_to_transient():
    rallies = [Rally(start=10.0, end=20.0, motion_peak=9.0, confidence=0.9)]
    trans = [Transient(8.4, 3.0), Transient(12.0, 2.5), Transient(14.0, 2.2)]
    out = refine(rallies, trans, search_back_s=3.0)
    assert abs(out[0].start - (8.4 - 0.2)) < 1e-6
    assert out[0].hits == 2


def test_refine_no_transient_keeps_start():
    rallies = [Rally(start=10.0, end=20.0, motion_peak=9.0, confidence=0.9)]
    out = refine(rallies, [Transient(4.0, 3.0)])
    assert out[0].start == 10.0 and out[0].hits == 0


def test_refine_does_not_cross_previous_rally():
    rallies = [Rally(start=2.0, end=8.0, motion_peak=9.0, confidence=0.9),
               Rally(start=10.0, end=20.0, motion_peak=9.0, confidence=0.9)]
    out = refine(rallies, [Transient(9.5, 3.0)])
    assert out[1].start >= 8.2
