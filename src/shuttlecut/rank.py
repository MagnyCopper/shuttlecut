"""精彩度排序 v1(无监督):回合段级评分与排序。

特征(全部来自概率曲线与段几何,可选音频击打密度):
- duration: 回合时长(长回合≈多拍相持)
- peak / mean / var: 段内 actionness 峰值/均值/方差(高且起伏=对抗强度高)
- hit_density: 击球瞬态密度(可选,仅场地声)

评分 = 各特征稳健 z 分加权和;仅在同视频的段间比较(排名),不跨视频比较分数。
历史结论约束(eval-history):音频只作辅助特征,不 veto、不参与切分。
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RankFeatures:
    duration_s: float
    peak: float
    mean: float
    var: float
    hit_density: float = 0.0


@dataclass
class RankedRally:
    start: float
    end: float
    score: float
    rank: int = 0
    features: RankFeatures = field(default_factory=lambda: RankFeatures(0, 0, 0, 0))


def _robust_z(xs: np.ndarray) -> np.ndarray:
    xs = np.asarray(xs, dtype=np.float64)
    if len(xs) < 2:
        return np.zeros_like(xs)
    med = np.median(xs)
    mad = np.median(np.abs(xs - med)) * 1.4826
    if mad < 1e-9:
        sd = xs.std()
        if sd < 1e-9:
            return np.zeros_like(xs)
        return (xs - xs.mean()) / sd
    return (xs - med) / mad


def extract_features(segs: list[tuple[float, float]], centers: np.ndarray,
                     probs: np.ndarray, hit_times: list[float] | None = None,
                     ) -> list[RankFeatures]:
    feats: list[RankFeatures] = []
    for s, e in segs:
        m = (centers >= s) & (centers <= e)
        seg_p = probs[m] if m.any() else np.array([0.0])
        hits = 0
        if hit_times:
            hits = sum(1 for t in hit_times if s <= t <= e)
        feats.append(RankFeatures(
            duration_s=float(e - s),
            peak=float(seg_p.max()),
            mean=float(seg_p.mean()),
            var=float(seg_p.var()),
            hit_density=hits / max(e - s, 1e-9),
        ))
    return feats


def score_rallies(segs: list[tuple[float, float]], centers: np.ndarray, probs: np.ndarray,
                  hit_times: list[float] | None = None,
                  weights: dict[str, float] | None = None,
                  ) -> list[RankedRally]:
    """加权稳健 z 分评分;返回按分数降序标记 rank(1=最精彩)的段列表(保持原时间顺序)。"""
    w = {"duration": 1.0, "peak": 1.0, "var": 0.5, "mean": 0.3, "hits": 0.5}
    w |= (weights or {})
    feats = extract_features(segs, centers, probs, hit_times)
    if not feats:
        return []
    z = {
        "duration": _robust_z(np.log(np.array([f.duration_s for f in feats]) + 1.0)),
        "peak": _robust_z(np.array([f.peak for f in feats])),
        "var": _robust_z(np.array([f.var for f in feats])),
        "mean": _robust_z(np.array([f.mean for f in feats])),
        "hits": _robust_z(np.array([f.hit_density for f in feats])) if hit_times
        else np.zeros(len(feats)),
    }
    scores = (w["duration"] * z["duration"] + w["peak"] * z["peak"] + w["var"] * z["var"]
              + w["mean"] * z["mean"] + w["hits"] * z["hits"])
    out = [RankedRally(start=s, end=e, score=float(sc), features=f)
           for (s, e), sc, f in zip(segs, scores, feats)]
    order = sorted(range(len(out)), key=lambda i: out[i].score, reverse=True)
    for rank, i in enumerate(order, start=1):
        out[i].rank = rank
    return out


def top_rallies(ranked: list[RankedRally], frac: float = 0.3, min_k: int = 3,
                max_k: int = 12) -> list[RankedRally]:
    """按分数取 top 段(时间顺序返回,便于直接拼接集锦)。"""
    k = max(min_k, min(max_k, round(len(ranked) * frac))) if ranked else 0
    k = min(k, len(ranked))
    sel = [r for r in ranked if r.rank <= k]
    return sorted(sel, key=lambda r: r.start)
