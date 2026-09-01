from dataclasses import dataclass

import librosa
import numpy as np

from shuttlecut.segmenter import Rally


@dataclass
class Transient:
    t: float
    z: float


def audio_transients(wav: str, z_thresh: float = 2.0, min_gap_s: float = 0.3,
                     hop_length: int = 512) -> list[Transient]:
    y, sr = librosa.load(wav, sr=16000, mono=True)
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length,
                                        units="time", backtrack=False)
    frames = np.clip(librosa.time_to_frames(onsets, sr=sr, hop_length=hop_length),
                     0, len(env) - 1)
    strengths = env[frames]
    z = (strengths - env.mean()) / (env.std() + 1e-9)
    picked: list[Transient] = []
    for t, zi in zip(onsets, z):
        if t < min_gap_s:
            continue
        if zi < z_thresh:
            continue
        if picked and t - picked[-1].t < min_gap_s:
            if zi > picked[-1].z:
                picked[-1] = Transient(float(t), float(zi))
            continue
        picked.append(Transient(float(t), float(zi)))
    return picked


def refine(rallies: list[Rally], transients: list[Transient],
           search_back_s: float = 3.0) -> list[Rally]:
    times = [tr.t for tr in transients]
    out: list[Rally] = []
    for i, r in enumerate(rallies):
        floor = rallies[i - 1].end + 0.2 if i > 0 else 0.0
        lo, hi = max(floor, r.start - search_back_s), r.start
        cands = [tr for tr in transients if lo <= tr.t <= hi]
        new_start = (cands[-1].t - 0.2) if cands else r.start
        new_start = max(new_start, floor)
        hits = sum(1 for t in times if r.start <= t <= r.end)
        out.append(Rally(start=new_start, end=r.end, motion_peak=r.motion_peak,
                         confidence=r.confidence, hits=hits))
    return out
