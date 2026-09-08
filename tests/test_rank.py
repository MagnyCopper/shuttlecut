import numpy as np

from shuttlecut.rank import extract_features, score_rallies, top_rallies


def _curve(segs, peak_inside=0.9, base=0.05):
    """构造覆盖 segs 的概率曲线:段内高、段外低。"""
    centers = np.arange(0.0, 300.0, 2 / 15)
    probs = np.full_like(centers, base)
    for s, e in segs:
        probs[(centers >= s) & (centers <= e)] = peak_inside
    return centers, probs


def test_score_prefers_long_intense_rally():
    segs = [(10, 15), (100, 140), (200, 203)]
    centers, probs = _curve(segs)
    ranked = score_rallies(segs, centers, probs)
    best = min(ranked, key=lambda r: r.rank)
    assert (best.start, best.end) == (100, 140)  # 最长+高actionness → rank 1
    worst = max(ranked, key=lambda r: r.rank)
    assert (worst.start, worst.end) == (200, 203)  # 最短垫底


def test_rank_ids_and_order_consistent():
    segs = [(5, 10), (50, 60), (120, 126)]
    centers, probs = _curve(segs)
    ranked = score_rallies(segs, centers, probs)
    assert sorted(r.rank for r in ranked) == [1, 2, 3]
    assert all(r.rank >= 1 for r in ranked)


def test_single_rally_zero_score():
    segs = [(0, 10)]
    centers, probs = _curve(segs)
    ranked = score_rallies(segs, centers, probs)
    assert len(ranked) == 1 and ranked[0].rank == 1


def test_top_rallies_time_ordered_and_bounded():
    segs = [(i * 40, i * 40 + 20) for i in range(10)]
    centers, probs = _curve(segs)
    ranked = score_rallies(segs, centers, probs)
    sel = top_rallies(ranked, frac=0.3, min_k=3, max_k=12)
    assert 3 <= len(sel) <= 12
    starts = [r.start for r in sel]
    assert starts == sorted(starts)  # 时间顺序
    assert max(r.rank for r in sel) <= len(sel)  # 只含 top-k


def test_extract_features_shapes():
    segs = [(0, 5), (10, 15)]
    centers, probs = _curve(segs)
    feats = extract_features(segs, centers, probs, hit_times=[1.0, 2.0, 12.0])
    assert abs(feats[0].hit_density - 2 / 5) < 1e-9
    assert feats[0].duration_s == 5.0
    assert feats[0].peak > feats[0].mean >= 0.0
