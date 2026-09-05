"""CUDA 重骨干微调:R3D-18(Kinetics-400 预训练)→ 回合窗口二分类。

用法(每个视频一次):
    python train_heavy.py --frames data/DJI_20260830153830_0015_D/frames15 \
        --gt ground_truth/DJI_20260830153830_0015_D.json --out ckpt/r3d_b1.pt
    # 可选: --win 24 / 96,--epochs 8,--batch 8
输出: --out 指定的最优 state_dict。
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

H, W = 112, 112


def build_diffs(store):
    n, h, w = store.shape
    diffs = np.zeros_like(store, np.float32)
    for i in range(1, n):
        a, b = store[i - 1], store[i]
        pa = cv2.goodFeaturesToTrack(a, 200, 0.01, 8)
        raw = np.abs(b.astype(np.float32) - a.astype(np.float32)) / 32.0
        if pa is None or len(pa) < 10:
            diffs[i] = raw
            continue
        pb, st, _ = cv2.calcOpticalFlowPyrLK(a, b, pa, None)
        if pb is None or st is None:
            diffs[i] = raw
            continue
        good = st.ravel().astype(bool)
        aff, _ = cv2.estimateAffinePartial2D(pa[good], pb[good], method=cv2.RANSAC)
        if aff is None:
            diffs[i] = raw
            continue
        bw = cv2.warpAffine(b, np.float32([[1, 0, -aff[0, 2]], [0, 1, -aff[1, 2]]]), (w, h))
        diffs[i] = np.abs(bw.astype(np.float32) - a.astype(np.float32)) / 32.0
    return diffs


class WinDS(Dataset):
    def __init__(self, diffs, items, win):
        self.diffs, self.items, self.win = diffs, items, win

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        s, y = self.items[i]
        d = self.diffs[s:s + self.win].astype(np.float32)
        if self.win > 32:
            d = d[::2]
        frames = np.stack([cv2.resize(f, (W, H)) for f in d])
        x = np.repeat(frames[None], 3, axis=0)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(y, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--win", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    a = ap.parse_args()
    torch.manual_seed(13)
    random.seed(13)
    np.random.seed(13)

    gt = json.load(open(a.gt))
    segs = [(s["start_s"], s["end_s"]) for s in gt["rallies"]]
    files = sorted(Path(a.frames).glob("frame_*.jpg"))
    n = len(files)
    ts = np.arange(n) / 15.0
    cov = np.zeros(n)
    for x0, x1 in segs:
        cov[(ts >= x0) & (ts <= x1)] = 1.0
    store = np.zeros((n, 90, 160), np.uint8)
    for i, f in enumerate(files):
        store[i] = cv2.resize(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE), (160, 90))
    diffs = build_diffs(store)
    items = []
    for s in range(0, n - a.win, 4):
        c = cov[s + a.win // 4: s + 3 * a.win // 4].mean()
        if c >= 0.6:
            items.append((s, 1.0))
        elif c <= 0.2:
            items.append((s, 0.0))
    print(f"windows={len(items)} pos={int(sum(y for _, y in items))}", flush=True)
    perm = np.random.permutation(len(items))
    n_val = max(200, len(items) // 8)
    val = [items[i] for i in perm[:n_val]]
    trn = [items[i] for i in perm[n_val:]]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import torchvision
    model = torchvision.models.video.r3d_18(weights=torchvision.models.video.R3D_18_Weights.K400_400_V1)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")
    dl_t = DataLoader(WinDS(diffs, trn, a.win), batch_size=a.batch, shuffle=True, num_workers=4)
    dl_v = DataLoader(WinDS(diffs, val, a.win), batch_size=a.batch, num_workers=2)
    best = 0.0
    for ep in range(a.epochs):
        model.train()
        tot = 0.0
        for xb, yb in dl_t:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=dev == "cuda"):
                out = model(xb).squeeze(-1)
                loss = lossf(out, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach()) * len(yb)
        model.eval()
        sv, yv = [], []
        with torch.no_grad():
            for xb, yb in dl_v:
                with torch.cuda.amp.autocast(enabled=dev == "cuda"):
                    o = torch.sigmoid(model(xb.to(dev))).float()
                sv.append(o.cpu().numpy())
                yv.append(yb.numpy())
        sv, yv = np.concatenate(sv), np.concatenate(yv)
        order = np.argsort(sv)
        ranks = np.empty_like(order, float)
        ranks[order] = np.arange(1, len(sv) + 1)
        n1 = yv.sum()
        auc = (ranks[yv == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * (len(yv) - n1))
        print(f"epoch {ep+1}: loss={tot/len(trn):.4f} val_AUC={auc:.4f}", flush=True)
        if auc > best:
            best = auc
            torch.save(model.state_dict(), a.out)
    print("best:", best)


if __name__ == "__main__":
    main()
