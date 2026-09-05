"""Full-video inference: probability curve → hysteresis segmentation → segment-level P/R."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_temporal import H, Tiny3D, W, WIN, build_diffs, GRID, GH, GW  # noqa: E402

FRAMES = Path(sys.argv[1])
GT = Path(sys.argv[2])
MODEL = __import__("os").environ.get("SC_CKPT", str(ROOT / "models" / "temporal_rally.pt"))
STRIDE = 2  # dense inference


def main():
    gt = json.load(open(GT))
    segs = [(s["start_s"], s["end_s"]) for s in gt["rallies"]]
    files = sorted(FRAMES.glob("frame_*.jpg"))
    n = len(files)
    import os
    from train_temporal import GRID, GH, GW, WIN, build_diffs
    store = np.zeros((n, H, W), np.uint8)
    for i, f in enumerate(files):
        store[i] = cv2.resize(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE), (W, H))
    diffs = build_diffs(store)
    if __import__("os").environ.get("SC_RAW", "0") != "1":
        diffs = np.clip(diffs * 32.0, 0, 255).astype(np.uint8).astype(np.float32)  # 与训练 u8 量化一致
    if GRID:
        T = diffs.shape[0]
        g = diffs.reshape(T, GH, H // GH, GW, W // GW).mean(axis=(2, 4))
        diffs = np.log1p(g * 4.0).astype(np.float32)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = Tiny3D().to(dev)
    model.load_state_dict(torch.load(MODEL, map_location=dev))
    model.eval()

    centers, probs = [], []
    B = 32
    with torch.no_grad():
        for k in range(0, B):  # placeholder
            break
        starts = list(range(0, n - WIN + 1, STRIDE))
        for k in range(0, len(starts), B):
            batch = [diffs[s:s + WIN] for s in starts[k:k + B]]
            x = torch.from_numpy(np.stack(batch)).unsqueeze(1).to(dev)
            p = torch.sigmoid(model(x)).cpu().numpy()
            probs.extend(p.tolist())
            centers.extend([(s + WIN / 2) / 15.0 for s in starts[k:k + B]])
    centers = np.array(centers)
    probs = np.array(probs, dtype=np.float64)
    tag = __import__("os").environ.get("SC_TAG", "x")
    np.save(str(ROOT / "temp" / f"prob_centers_{tag}.npy"), centers)
    np.save(str(ROOT / "temp" / f"prob_values_{tag}.npy"), probs)

    # 滞回切分
    HI, LO, MIN_GAP, MIN_LEN = 0.6, 0.3, 1.5, 1.5
    pred = []
    armed = False
    start = 0.0
    dur = centers[-1]
    for t, p in zip(centers, probs):
        if not armed and p > HI:
            armed = True
            start = t - WIN / 2 / 15.0
        elif armed and p < LO:
            armed = False
            if t - start >= MIN_LEN:
                pred.append((max(0.0, start), t))
    if armed and dur - start >= MIN_LEN:
        pred.append((max(0.0, start), dur))
    # 合并近邻
    merged = []
    for a, b in pred:
        if merged and a - merged[-1][1] <= MIN_GAP:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))

    # 段级匹配(容差 1.5s)
    TOL = 1.5
    matched = set()
    for a, b in merged:
        if any(max(a - ga, 0) <= TOL and (gb - b) <= TOL and abs(a - ga) <= TOL + 1 and abs(b - gb) <= TOL + 1
               for ga, gb in segs):
            pass
    tp = 0
    used = set()
    for ga, gb in segs:
        best, bi = None, -1
        for i, (a, b) in enumerate(merged):
            if i in used:
                continue
            ov = max(0, min(b, gb) - max(a, ga))
            if best is None or ov > best:
                best, bi = ov, i
        if bi >= 0 and best >= 0.5 * min(gb - ga, merged[bi][1] - merged[bi][0]):
            tp += 1
            used.add(bi)
    prec = tp / len(merged) if merged else 0
    rec = tp / len(segs)
    print(f"预测段数={len(merged)} GT段数={len(segs)} TP={tp} P={prec:.3f} R={rec:.3f}")
    print("预测段(前12):", [(round(a,1), round(b,1)) for a, b in merged[:12]])
    print("GT段(前12):  ", [(round(a,1), round(b,1)) for a, b in segs[:12]])
    json.dump(merged, open(ROOT / "temp" / "pred_segments.json", "w"))


if __name__ == "__main__":
    main()
