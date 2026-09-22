"""BDR 边界精修实验:双头探针(分类 + 符号距离回归)。

标签:窗心到最近回合边界的带符号距离(内正外负,截断 ±3s)。
推理:滞回粗分段后,边界向 ±1.5s 内最近零交叉点精修。
对比:同 LOEO 折、同种子,cls-only vs cls+BDR。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from loeo_probe import (VENUES, curve, grid_on, load_stem, seg_apply,  # noqa: E402
                        train_probe)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from scipy.ndimage import median_filter  # noqa: E402
from shuttlecut.temporal import segment_curve  # noqa: E402


def dist_labels(cent: np.ndarray, segs) -> np.ndarray:
    """窗心到最近边界的带符号距离(回合内为正,外为负),截断 ±3s。"""
    bounds = sorted({b for a, b in segs} | {a for a, b in segs})
    d = np.zeros(len(cent), np.float32)
    inside = np.zeros(len(cent), np.float32)
    for i, t in enumerate(cent):
        if any(a <= t <= b for a, b in segs):
            inside[i] = 1.0
            d[i] = min(min(abs(t - a), abs(t - b)) for a, b in segs if a <= t <= b)
        else:
            d[i] = -min((abs(t - b) for b in bounds), default=0.0)
    return np.clip(d, -3.0, 3.0), inside


def build_dual(seed: int, dim_in: int = 1024, d: int = 256, layers: int = 3):
    torch.manual_seed(seed)
    enc = nn.TransformerEncoderLayer(d, 4, dim_feedforward=512, dropout=0.1,
                                     batch_first=True, norm_first=True)
    return nn.Sequential(
        nn.Linear(dim_in, d), nn.GELU(),
        nn.TransformerEncoder(enc, layers),
        nn.Linear(d, 2),  # [logit, dist]
    )


def train_dual(m, vids, seed, epochs=20, chunk=128, bs=8, dev="cuda", lam=0.2):
    rng = np.random.default_rng(seed)
    crops = []
    for v in vids:
        f, lab, dist = v["feat"], v["lab"], v["dist"]
        for _ in range(max(8, len(f) // 32)):
            s = rng.integers(0, max(1, len(f) - chunk))
            g, y, dd = f[s:s + chunk], lab[s:s + chunk], dist[s:s + chunk]
            if len(g) < chunk:
                pad = chunk - len(g)
                g = np.concatenate([g, np.repeat(g[-1:], pad, axis=0)])
                y = np.concatenate([y, np.repeat(y[-1:], pad)])
                dd = np.concatenate([dd, np.repeat(dd[-1:], pad)])
            crops.append((g, y, dd))
    pos = np.concatenate([c[1] for c in crops])
    pw = float((pos == 0).sum() / max((pos == 1).sum(), 1))
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=dev))
    huber = nn.SmoothL1Loss(beta=0.5)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    m.train()
    for _ in range(epochs):
        rng.shuffle(crops)
        for k in range(0, len(crops), bs):
            xb = torch.from_numpy(np.stack([c[0] for c in crops[k:k + bs]])).to(dev)
            yb = torch.from_numpy(np.stack([c[1] for c in crops[k:k + bs]])).to(dev)
            db = torch.from_numpy(np.stack([c[2] for c in crops[k:k + bs]])).to(dev)
            opt.zero_grad()
            out = m(xb)
            loss = lossf(out[..., 0], yb) + lam * huber(out[..., 1], db)
            loss.backward()
            opt.step()
    m.eval()


def curves_dual(m, feat, dev="cuda", chunk=1024, ov=128):
    n = len(feat)
    pc = np.zeros(n)
    pd = np.zeros(n)
    cnt = np.zeros(n)
    for s in range(0, n, chunk - ov):
        e = min(s + chunk, n)
        with torch.no_grad():
            out = m(torch.from_numpy(feat[s:e][None]).to(dev))[0].cpu().numpy()
        pc[s:e] += out[:, 0]
        pd[s:e] += out[:, 1]
        cnt[s:e] += 1
        if e == n:
            break
    return 1 / (1 + np.exp(-pc / cnt)), pd / cnt


def _nearest_zero(cent, dsm, target, win_s, default):
    """target ±win_s 内第一个符号变化点(线性插值),无则原值。"""
    m = np.where((cent >= target - win_s) & (cent <= target + win_s))[0]
    for i in range(len(m) - 1):
        x, y = dsm[m[i]], dsm[m[i + 1]]
        if x == 0:
            return float(cent[m[i]])
        if x * y < 0:
            t = x / (x - y)
            return float(cent[m[i]] + t * (cent[m[i + 1]] - cent[m[i]]))
    return default


def refine(pred, cent, dists, win_s=1.5):
    """每个边界向 ±win_s 内最近的零交叉点精修。"""
    dsm = median_filter(dists, size=5)
    out = []
    for a, b in pred:
        na = _nearest_zero(cent, dsm, a, win_s, a)
        nb = _nearest_zero(cent, dsm, b, win_s, b)
        if nb - na >= 1.0:
            out.append((na, nb))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[13, 21, 42])
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--hold", default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def load_all(stem):
        v = load_stem(stem)
        v["dist"], _ = dist_labels(v["cent"], v["segs"])
        return v

    from shuttlecut.eval.evaluate import evaluate
    print("== BDR vs cls-only(同折同种子,LOEO)==")
    holds = [a.hold] if a.hold else list(VENUES)
    for hold in holds:
        tr_stems = [s for v2, ss in VENUES.items() if v2 != hold for s in ss]
        tr = [v for v in (load_all(s) for s in tr_stems) if v]
        te = [v for v in (load_all(s) for s in VENUES[hold]) if v]
        # 集成曲线(3 种子;训练侧与测试侧同源,避免尺度不匹配)
        pcs, pds = {v["stem"]: [] for v in te}, {v["stem"]: [] for v in te}
        tcs = {v["stem"]: [] for v in tr}
        for seed in a.seeds:
            m = build_dual(seed).to(dev)
            train_dual(m, tr, seed, dev=dev, lam=a.lam)
            for v in te:
                c, d = curves_dual(m, v["feat"], dev)
                pcs[v["stem"]].append(c)
                pds[v["stem"]].append(d)
            for v in tr:
                c, _ = curves_dual(m, v["feat"], dev)
                tcs[v["stem"]].append(c)
        _, par = grid_on([(v["cent"], np.mean(tcs[v["stem"]], axis=0), v["segs"]) for v in tr])
        for v in te:
            c = np.mean(pcs[v["stem"]], axis=0)
            d = np.mean(pds[v["stem"]], axis=0)
            base = seg_apply(par, v["cent"], c)
            ref = refine(base, v["cent"], d)
            r0 = evaluate(base, v["segs"])
            r1 = evaluate(ref, v["segs"])
            print(f"[{hold}] {v['stem']}: cls {r0.precision:.3f}/{r0.recall:.3f}"
                  f" → +BDR {r1.precision:.3f}/{r1.recall:.3f}", flush=True)


if __name__ == "__main__":
    main()
