"""官方口径段级评测(与 src/shuttlecut/eval/evaluate.py 同语义的自包含副本)。

用法:
    python segment_eval.py --gt ground_truth/<stem>.json --tag b1r3d [--grid]
"""
import argparse
import json

import numpy as np
from scipy.ndimage import median_filter


def _inter(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def evaluate(detected, gt, tol_s=2.5, overlap=0.5, mae_report_s=1.5):
    used = set()
    matches = []
    for g in gt:
        best, bi = 0.0, -1
        for i, d in enumerate(detected):
            if i in used:
                continue
            ov = _inter(d, g)
            if ov <= 0:
                continue
            ratio = ov / max(d[1] - d[0], g[1] - g[0])
            ds, de = abs(d[0] - g[0]), abs(d[1] - g[1])
            if ratio >= overlap and ds <= tol_s and de <= tol_s and ov > best:
                best, bi = ov, i
        if bi >= 0:
            used.add(bi)
            matches.append(bi)
    p = len(matches) / len(detected) if detected else 0.0
    r = len(matches) / len(gt) if gt else 0.0
    return p, r


def segment(centers, probs, hi, lo, ml):
    pred, armed, start = [], False, 0.0
    for t, p in zip(centers, probs):
        if not armed and p > hi:
            armed, start = True, t
        elif armed and p < lo:
            armed = False
            if t - start >= ml:
                pred.append((start, t))
    if armed and centers[-1] - start >= ml:
        pred.append((start, centers[-1]))
    return pred


def merge(pred, gap):
    out = []
    for a, b in pred:
        if out and a - out[-1][1] <= gap:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--grid", action="store_true")
    a = ap.parse_args()
    gt = json.load(open(a.gt))
    segs = [(s["start_s"], s["end_s"]) for s in gt["rallies"]]
    centers = np.load(f"prob_centers_{a.tag}.npy")
    probs = np.load(f"prob_values_{a.tag}.npy")
    if not a.grid:
        pred = merge(segment(centers, median_filter(probs, size=5), 0.6, 0.4, 1.5), 1.2)
        p, r = evaluate(pred, segs)
        print(f"{a.tag}: P={p:.3f} R={r:.3f} 段={len(pred)}/{len(segs)}")
        return
    best = (0, None)
    for sm in (0, 3, 5, 7, 9):
        pp = median_filter(probs, size=sm) if sm > 1 else probs
        for hi in (0.4, 0.5, 0.6, 0.7, 0.75):
            for lo in (0.2, 0.3, 0.35, 0.4, 0.45):
                for ml in (1.5, 2.0, 3.0):
                    for mg in (0.0, 1.2, 1.8):
                        pred = merge(segment(centers, pp, hi, lo, ml), mg) if mg > 0 else segment(centers, pp, hi, lo, ml)
                        p, r = evaluate(pred, segs)
                        if min(p, r) > best[0]:
                            best = (min(p, r), (sm, hi, lo, ml, mg, p, r))
    m, (sm, hi, lo, ml, mg, p, r) = best
    print(f"{a.tag} 最优: sm={sm} HI={hi} LO={lo} ml={ml} mg={mg} → P={p:.3f} R={r:.3f} min={m:.3f}")


if __name__ == "__main__":
    main()
