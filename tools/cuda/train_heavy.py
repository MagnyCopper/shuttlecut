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
    def __init__(self, diffs, items, win, tsub=2):
        self.diffs, self.items, self.win, self.tsub = diffs, items, win, tsub

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        if len(it) == 3:
            v, s, y = it
            d = self.diffs[v][s:s + self.win].astype(np.float32)
        else:
            s, y = it
            d = self.diffs[s:s + self.win].astype(np.float32)
        if self.win // self.tsub > 8:
            d = d[:: self.tsub]
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
    ap.add_argument("--tsub", type=int, default=2, help="temporal subsample factor")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", action="store_true", help="warm-start from <out>.last if present")
    a = ap.parse_args()
    torch.manual_seed(13)
    random.seed(13)
    np.random.seed(13)

    # 多场次联合:--frames/--gt 支持逗号分隔
    frames_list = [x for x in a.frames.split(",") if x]
    gt_list = [x for x in a.gt.split(",") if x]
    diffs_list, items = [], []
    for fdir, gtf in zip(frames_list, gt_list):
        gt = json.load(open(gtf))
        segs = [(s["start_s"], s["end_s"]) for s in gt["rallies"]]
        files = sorted(Path(fdir).glob("frame_*.jpg"))
        n = len(files)
        ts = np.arange(n) / 15.0
        cov = np.zeros(n)
        for x0, x1 in segs:
            cov[(ts >= x0) & (ts <= x1)] = 1.0
        store = np.zeros((n, 90, 160), np.uint8)
        for i, f in enumerate(files):
            store[i] = cv2.resize(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE), (160, 90))
        d = build_diffs(store)
        vid = len(diffs_list)
        diffs_list.append(d)
        for s in range(0, n - a.win, 4):
            c = cov[s + a.win // 4: s + 3 * a.win // 4].mean()
            if c >= 0.6:
                items.append((vid, s, 1.0))
            elif c <= 0.2:
                items.append((vid, s, 0.0))
    print(f"windows={len(items)} pos={int(sum(y for *_, y in items))}", flush=True)
    perm = np.random.permutation(len(items))
    n_val = max(200, len(items) // 8)
    val = [items[i] for i in perm[:n_val]]
    trn = [items[i] for i in perm[n_val:]]

    dev = {"auto": "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"), "cuda": "cuda", "mps": "mps", "cpu": "cpu"}[a.device]
    import torchvision
    model = torchvision.models.video.r3d_18(weights=torchvision.models.video.R3D_18_Weights.KINETICS400_V1)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")
    import sys
    nworker = 0 if sys.platform == "win32" else 4
    dl_t = DataLoader(WinDS(diffs_list, trn, a.win, a.tsub), batch_size=a.batch, shuffle=True, num_workers=nworker)
    dl_v = DataLoader(WinDS(diffs_list, val, a.win, a.tsub), batch_size=a.batch, num_workers=nworker)
    start_ep, best, _it = 0, 0.0, 0
    last_path = str(a.out) + ".last"
    if a.resume and Path(last_path).exists():  # 断点续训:MPS 楔死后只损失部分 epoch
        st = torch.load(last_path, map_location="cpu")
        if isinstance(st, dict) and "model" in st:
            model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
            start_ep, best = st["ep"] + 1, st.get("best", 0.0)
            print(f"resume from epoch {start_ep} (best={best:.4f})", flush=True)
    for ep in range(start_ep, a.epochs):
        model.train()
        tot = 0.0
        for xb, yb in dl_t:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=dev == "cuda"):
                out = model(xb).squeeze(-1)
                if out.dim() > 1:
                    out = out.squeeze(-1)
                loss = lossf(out, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if dev == "mps":  # MPS 楔死保险:周期清缓存
                _it += 1
                if _it % 300 == 0:
                    torch.mps.empty_cache()
            tot += float(loss.detach()) * len(yb)
        model.eval()
        sv, yv = [], []
        with torch.no_grad():
            for xb, yb in dl_v:
                with torch.cuda.amp.autocast(enabled=dev == "cuda"):
                    o = torch.sigmoid(model(xb.to(dev))).squeeze(-1).float()
                sv.append(o.cpu().numpy())
                yv.append(yb.numpy())
        sv, yv = np.concatenate(sv).reshape(-1), np.concatenate(yv).reshape(-1)
        order = np.argsort(sv)
        ranks = np.empty_like(order, float)
        ranks[order] = np.arange(1, len(sv) + 1)
        n1 = yv.sum()
        auc = (ranks[yv == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * (len(yv) - n1))
        print(f"epoch {ep+1}: loss={tot/len(trn):.4f} val_AUC={auc:.4f}", flush=True)
        if auc > best:
            best = auc
            torch.save(model.state_dict(), a.out)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "ep": ep, "best": best}, last_path)  # 每轮末态+续训状态
    print("best:", best)


if __name__ == "__main__":
    main()
