"""场地声分析:击球瞬态检测(自主标注音频票 / 精彩度排序特征)。

注意历史结论(eval_history 2026-09-07/08):瞬态密度不可用于段级投票或切分,
仅限人工分诊辅助与排序特征。
"""
from dataclasses import dataclass

import librosa
import numpy as np


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
    for t, zi in sorted(zip(onsets, z), key=lambda item: item[0]):
        if zi < z_thresh:
            continue
        if picked and t - picked[-1].t < min_gap_s:
            if zi > picked[-1].z:
                picked[-1] = Transient(float(t), float(zi))
            continue
        picked.append(Transient(float(t), float(zi)))
    return picked
