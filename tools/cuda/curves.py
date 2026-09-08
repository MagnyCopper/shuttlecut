"""推理曲线:对整视频滑窗输出概率,保存 npy(带回本机评测切分)。

用法:
    python curves.py --frames data/<stem>/frames15 --ckpt ckpt/r3d_b1.pt --win 64 --tag b1r3d
产出: prob_centers_<tag>.npy / prob_values_<tag>.npy
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

H, W = 112, 112


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--win", type=int, default=64)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tsub", type=int, default=2)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    import train_heavy as th
    files = sorted(Path(a.frames).glob("frame_*.jpg"))
    n = len(files)
    store = np.zeros((n, 90, 160), np.uint8)
    for i, f in enumerate(files):
        store[i] = cv2.resize(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE), (160, 90))
    diffs = th.build_diffs(store)

    dev = {"auto": "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"), "cuda": "cuda", "mps": "mps", "cpu": "cpu"}[a.device]
    import torchvision
    import torch.nn as nn
    model = torchvision.models.video.r3d_18()
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev))
    model = model.to(dev).eval()

    starts = list(range(0, n - a.win + 1, 2))
    probs, centers = [], []
    with torch.no_grad():
        for k in range(0, len(starts), a.batch):
            xs = []
            for s in starts[k:k + a.batch]:
                d = diffs[s:s + a.win].astype(np.float32)
                if a.win // a.tsub > 8:
                    d = d[:: a.tsub]
                frames = np.stack([cv2.resize(f, (W, H)) for f in d])
                xs.append(np.repeat(frames[None], 3, axis=0))
            x = torch.from_numpy(np.ascontiguousarray(np.stack(xs))).to(dev)
            p = torch.sigmoid(model(x)).squeeze(-1).float().cpu().numpy()
            probs.extend(p.tolist())
            centers.extend([(s + a.win / 2) / 15.0 for s in starts[k:k + a.batch]])
    np.save(f"prob_centers_{a.tag}.npy", np.array(centers))
    np.save(f"prob_values_{a.tag}.npy", np.array(probs, dtype=np.float64))
    print(f"OK {a.tag}: {len(probs)} points")


if __name__ == "__main__":
    main()
