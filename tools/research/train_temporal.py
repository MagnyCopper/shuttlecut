"""Temporal rally classifier: compensated frame-diff windows + tiny Conv3D.

Env:
  SC_VIDEO  comma-separated video stems (joint training)
  SC_CKPT   checkpoint path
  SC_GRID   1 → 6x10 motion-energy grid input
  SC_EPOCHS epochs (default 8)
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

WIN = int(os.environ.get("SC_WIN", "64"))
STRIDE, H, W = 4, 90, 160
GRID = os.environ.get("SC_GRID", "0") == "1"
GH, GW = 6, 10
POS_TH, NEG_TH = 0.6, 0.2
ROOT = Path(__file__).resolve().parents[2]


def build_diffs(store: np.ndarray) -> np.ndarray:
    """Global-motion-compensated frame differences (LK+RANSAC translation only)."""
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


def to_grid(diffs: np.ndarray) -> np.ndarray:
    t = diffs.shape[0]
    g = diffs.reshape(t, GH, H // GH, GW, W // GW).mean(axis=(2, 4))
    return np.log1p(g * 4.0).astype(np.float32)


def load_one(video: str):
    """返回 (diffs_u8[n,H,W], starts, labels) —— 窗口按批即时构建,避免 OOM。"""
    gt = json.load(open(ROOT / "ground_truth" / f"{video}.json"))
    segs = [(s["start_s"], s["end_s"]) for s in gt["rallies"]]
    files = sorted((ROOT / "temp/work" / video / "frames15").glob("frame_*.jpg"))
    n = len(files)
    ts = np.arange(n) / 15.0
    cov = np.zeros(n)
    for a, b in segs:
        cov[(ts >= a) & (ts <= b)] = 1.0
    store = np.zeros((n, H, W), np.uint8)
    for i, f in enumerate(files):
        store[i] = cv2.resize(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE), (W, H))
    diffs = build_diffs(store)
    starts, ys = [], []
    for s in range(0, n - WIN, STRIDE):
        c = cov[s + WIN // 4: s + 3 * WIN // 4].mean()
        if c >= POS_TH:
            yv = 1
        elif c <= NEG_TH:
            yv = 0
        else:
            continue
        starts.append(s)
        ys.append(yv)
    diffs_u8 = np.clip(diffs * 32.0, 0, 255).astype(np.uint8)  # diffs≈[0,8]
    return diffs_u8, np.array(starts, np.int64), np.array(ys, np.float32)


def load_windows():
    """返回 (diffs_list, starts, ys);窗口样本 = (视频索引, 起始帧)。"""
    vids = os.environ.get("SC_VIDEO", "DJI_20260830153830_0015_D").split(",")
    diffs_list, starts, ys, vidx = [], [], [], []
    for k, v in enumerate(vids):
        d, st, yy = load_one(v.strip())
        diffs_list.append(d)
        starts.extend(st.tolist())
        ys.extend(yy.tolist())
        vidx.extend([k] * len(st))
    return diffs_list, np.array(starts, np.int64), np.array(ys, np.float32), np.array(vidx, np.int64)


class Tiny3D(nn.Module):
    def __init__(self):
        super().__init__()

        def block(ci, co):
            return nn.Sequential(nn.Conv3d(ci, co, 3, stride=(2, 2, 2), padding=1),
                                 nn.BatchNorm3d(co), nn.ReLU())

        self.f = nn.Sequential(
            block(1, 16), nn.MaxPool3d((1, 2, 2)),
            block(16, 32), nn.MaxPool3d((1, 2, 2)),
            block(32, 48),
            nn.AdaptiveAvgPool3d((2, 2, 2)), nn.Flatten(),
            nn.Linear(48 * 8, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1))

    def forward(self, x):
        return self.f(x).squeeze(1)


def main():
    epochs = int(os.environ.get("SC_EPOCHS", "8"))
    ckpt = os.environ.get("SC_CKPT", str(ROOT / "models/temporal_rally.pt"))
    D, starts, y, vidx = load_windows()
    print(f"windows={len(y)} pos={int(y.sum())} neg={int((1 - y).sum())} grid={GRID}", flush=True)
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(y))
    n_val = max(200, len(y) // 8)
    vi, ti = perm[:n_val], perm[n_val:]

    def batch(idx):
        xs = np.stack([D[vidx[i]][starts[i]:starts[i] + WIN].astype(np.float32) for i in idx])
        if GRID:
            xs = np.stack([to_grid(x) for x in xs])
        return xs

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = Tiny3D().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    pos_w = torch.tensor([float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)], device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    B = 48
    best_auc = 0.0
    for ep in range(epochs):
        model.train()
        p_ = rng.permutation(len(ti))
        tot = 0.0
        for k in range(0, len(p_), B):
            sel = p_[k:k + B]
            b = ti[sel]
            xb = batch(b)
            if rng.random() < 0.5:
                xb = xb[..., ::-1]
            xb = xb * rng.uniform(0.7, 1.3)
            xb = torch.from_numpy(np.ascontiguousarray(xb)).unsqueeze(1).to(dev)
            yb = torch.from_numpy(y[b]).to(dev)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        model.eval()
        sv = []
        with torch.no_grad():
            for k in range(0, len(vi), B):
                xb = torch.from_numpy(batch(vi[k:k + B])).unsqueeze(1).to(dev)
                sv.append(torch.sigmoid(model(xb)).cpu().numpy())
        sv = np.concatenate(sv)
        order = np.argsort(sv)
        ranks = np.empty_like(order, float)
        ranks[order] = np.arange(1, len(sv) + 1)
        yv_np = y[vi]
        n1 = yv_np.sum()
        n0 = len(yv_np) - n1
        auc = (ranks[yv_np == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)
        print(f"epoch {ep+1}: train_loss={tot/len(ti):.4f} val_AUC={auc:.4f}", flush=True)
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), ckpt)
    print("best val AUC:", best_auc)


if __name__ == "__main__":
    main()
