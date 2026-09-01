import numpy as np
import soundfile as sf
import pytest

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
    assert sum(abs(tr.t - 3.04) < 0.15 for tr in audio_transients(wav)) == 1


@pytest.mark.parametrize(
    ("onsets", "env", "strongest_t"),
    [([0.2, 0.1], [1.0, 2.0, 4.0], 0.1),
     ([0.1, 0.2], [1.0, 2.0, 4.0], 0.2)],
)
def test_min_gap_dedup_sorts_and_keeps_stronger(monkeypatch, onsets, env, strongest_t):
    monkeypatch.setattr("shuttlecut.refiner.librosa.load", lambda *args, **kwargs: (np.zeros(3), 16000))
    monkeypatch.setattr("shuttlecut.refiner.librosa.onset.onset_strength", lambda **kwargs: np.array(env))
    monkeypatch.setattr("shuttlecut.refiner.librosa.onset.onset_detect", lambda **kwargs: np.array(onsets))
    monkeypatch.setattr("shuttlecut.refiner.librosa.time_to_frames", lambda times, **kwargs: np.array([1, 2]))

    result = audio_transients("unused", z_thresh=-10.0)

    assert len(result) == 1
    assert result[0].t == strongest_t
    assert result[0].z == pytest.approx((4.0 - np.mean([1.0, 4.0, 2.0])) /
                                         np.std([1.0, 4.0, 2.0]))


def test_audio_transients_keeps_early_transient(monkeypatch):
    monkeypatch.setattr("shuttlecut.refiner.librosa.load", lambda *args, **kwargs: (np.zeros(3), 16000))
    monkeypatch.setattr("shuttlecut.refiner.librosa.onset.onset_strength", lambda **kwargs: np.array([1.0, 4.0]))
    monkeypatch.setattr("shuttlecut.refiner.librosa.onset.onset_detect", lambda **kwargs: np.array([0.1]))
    monkeypatch.setattr("shuttlecut.refiner.librosa.time_to_frames", lambda times, **kwargs: np.array([1]))

    result = audio_transients("unused", z_thresh=0.0)

    assert [tr.t for tr in result] == [0.1]


def test_refine_pulls_start_to_transient():
    rallies = [Rally(start=10.0, end=20.0, motion_peak=9.0, confidence=0.9)]
    trans = [Transient(8.4, 3.0), Transient(12.0, 2.5), Transient(14.0, 2.2)]
    out = refine(rallies, trans, search_back_s=3.0)
    assert abs(out[0].start - (8.4 - 0.2)) < 1e-6
    assert out[0].hits == 3


def test_refine_counts_hits_from_corrected_start_and_does_not_mutate_inputs():
    rallies = [Rally(start=10.0, end=20.0, motion_peak=9.0, confidence=0.9)]
    original = [Rally(start=r.start, end=r.end, motion_peak=r.motion_peak,
                      confidence=r.confidence, hits=r.hits) for r in rallies]
    trans = [Transient(8.4, 3.0), Transient(9.0, 2.5)]

    out = refine(rallies, trans, search_back_s=3.0)

    assert out[0].hits == 1
    assert rallies == original


def test_refine_no_transient_keeps_start():
    rallies = [Rally(start=10.0, end=20.0, motion_peak=9.0, confidence=0.9)]
    out = refine(rallies, [Transient(4.0, 3.0)])
    assert out[0].start == 10.0 and out[0].hits == 0


def test_refine_does_not_cross_previous_rally():
    rallies = [Rally(start=2.0, end=8.0, motion_peak=9.0, confidence=0.9),
               Rally(start=10.0, end=20.0, motion_peak=9.0, confidence=0.9)]
    out = refine(rallies, [Transient(9.5, 3.0)])
    assert out[1].start >= 8.2
