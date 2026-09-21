"""LOEO(leave-one-venue-out)探针实验:V-JEPA 2 冻结特征 → 微型 Transformer 探针 → 回合分割评测。

用法:
  python tools/frontier/loeo_probe.py --mode invideo --stems DJI_..._0013_D   # 单视频金丝雀
  python tools/frontier/loeo_probe.py --mode sanity                            # 场馆内随机划分
  python tools/frontier/loeo_probe.py --mode loeo [--seeds 13 21 42]           # 5 折 LOEO
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from shuttlecut.eval.evaluate import evaluate  # noqa: E402
from shuttlecut.temporal import segment_curve  # noqa: E402

VENUES = {
    "dji": [
        "DJI_20260912150541_0007_D", "DJI_20260912151553_0008_D",
        "DJI_20260912152303_0009_D", "DJI_20260912160351_0010_D", "DJI_20260912164500_0011_D",
        "DJI_20260912174203_0013_D", "DJI_20260912180439_0015_D",
        "DJI_20260830153830_0015_D", "DJI_20260905151437_0030_D",
        "DJI_20260905154319_0033_D", "DJI_20260905161114_0034_D",
        "DJI_20260905170125_0036_D", "DJI_20260830173600_0025_D",
        "DJI_20260912174644_0014_D",
    ],
    "baoganghui": ["baoganghui_p1", "baoganghui_p2"],
    "jiguang": ["jiguang"],
    "linzhou": ["linzhou"],
    "yygq": ["yygq"],
}
GT_PATH = {
    **{s: f"data/ground_truth/{s}.json" for vs in VENUES.values() for s in vs},
}
FEAT = "vjepa_feat_w16.npy"
CENT = "vjepa_centers_w16.npy"
CHUNK, OVERLAP = 1024, 128


def load_stem(stem: str) -> dict | None:
    d = Path(f"temp/work/{stem}")
    if not (d / FEAT).exists():
        return None
    feat = np.load(d / FEAT).astype(np.float32)
    cent = np.load(d / CENT)
    gt = json.loads(Path(GT_PATH[stem]).read_text(encoding="utf-8"))["rallies"]
    segs = [(r["start_s"], r["end_s"]) for r in gt]
    lab = np.zeros(len(cent), np.float32)
    for i, t in enumerate(cent):
        if any(a <= t <= b for a, b in segs):
            lab[i] = 1.0
    feat = (feat - feat.mean(0, keepdims=True)) / (feat.std(0, keepdims=True) + 1e-6)  # 逐视频 z-score(无标签域适配)
    return {"stem": stem, "feat": feat, "cent": cent, "lab": lab, "segs": segs}


def build_probe(dim_in: int = 1024, d: int = 256, layers: int = 3, seed: int = 0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    enc = nn.TransformerEncoderLayer(d, 4, dim_feedforward=512, dropout=0.1,
                                     batch_first=True, norm_first=True)
    return nn.Sequential(
        nn.Linear(dim_in, d), nn.GELU(),
        nn.TransformerEncoder(enc, layers),
        nn.Linear(d, 1),
    )


def train_probe(model, train_stems, seed, epochs=20, chunk=128, bs=8, dev="cuda"):
    import torch
    import torch.nn as nn
    rng = np.random.default_rng(seed)
    crops = []
    for v in train_stems:
        f, lab = v["feat"], v["lab"]
        for _ in range(max(8, len(f) // 32)):
            s = rng.integers(0, max(1, len(f) - chunk))
            g, y = f[s:s + chunk], lab[s:s + chunk]
            if len(g) < chunk:  # 短视频边缘补齐
                pad = chunk - len(g)
                g = np.concatenate([g, np.repeat(g[-1:], pad, axis=0)])
                y = np.concatenate([y, np.repeat(y[-1:], pad)])
            crops.append((g, y))
    pos = np.concatenate([c[1] for c in crops])
    pw = float((pos == 0).sum() / max((pos == 1).sum(), 1))
    pos_w = torch.tensor([pw], device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    model.train()
    for ep in range(epochs):
        rng.shuffle(crops)
        tot = 0.0
        for k in range(0, len(crops), bs):
            xb = torch.from_numpy(np.stack([c[0] for c in crops[k:k + bs]])).to(dev)
            yb = torch.from_numpy(np.stack([c[1] for c in crops[k:k + bs]])).to(dev)
            opt.zero_grad()
            loss = lossf(model(xb).squeeze(-1), yb)
            loss.backward()
            opt.step()
            tot += float(loss) * xb.shape[0]
    model.eval()
    return tot / len(crops)


def curve(model, feat, dev="cuda"):
    """滑动 chunk 推理,重叠区取平均 → 概率曲线。"""
    import torch
    n = len(feat)
    acc = np.zeros(n, np.float64)
    cnt = np.zeros(n, np.float64)
    step = CHUNK - OVERLAP
    for s in range(0, n, step):
        e = min(s + CHUNK, n)
        x = torch.from_numpy(feat[s:e][None]).to(dev)
        with torch.no_grad():
            p = torch.sigmoid(model(x)).squeeze().cpu().numpy()
        acc[s:e] += p
        cnt[s:e] += 1
        if e == n:
            break
    return (acc / cnt).astype(np.float64)


def seg_apply(par, cent, probs):
    from scipy.ndimage import median_filter
    sm, hi, lo, ml, mg = par
    pp = median_filter(probs, size=sm) if sm > 1 else probs
    pred = segment_curve(cent, pp, hi, lo, ml)
    out = []
    for a, b in pred:
        if out and a - out[-1][1] <= mg:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def grid_on(items):
    """items: [(cent, probs, segs)];返回训练侧平均 min(P,R) 最优分割参数。"""
    best = (-1.0, (5, 0.6, 0.4, 1.5, 1.2))
    for sm in (3, 5, 7):
        for hi in (0.45, 0.55, 0.65, 0.75):
            for lo in (0.2, 0.3, 0.4):
                for ml in (1.5, 2.0):
                    for mg in (0.0, 1.2):
                        par = (sm, hi, lo, ml, mg)
                        ss = []
                        for cent, probs, segs in items:
                            pred = seg_apply(par, cent, probs)
                            rep = evaluate(pred, segs)
                            ss.append(min(rep.precision, rep.recall))
                        m = float(np.mean(ss))
                        if m > best[0]:
                            best = (m, par)
    return best


def seg_eval(par, cent, probs, gt):
    pred = seg_apply(par, cent, probs)
    rep = evaluate(pred, gt)
    return rep.precision, rep.recall, len(pred), rep.n_gt

def ensemble_curves(videos, seeds, dev):
    """每种子训一探针,曲线取平均(集成降噪)。返回 {stem: probs}。"""
    curves = {v["stem"]: [] for v in videos}
    for seed in seeds:
        m = build_probe(seed=seed).to(dev)
        train_probe(m, videos, seed, dev=dev)
        for v in videos:
            curves[v["stem"]].append(curve(m, v["feat"], dev))
    return {s: np.mean(cs, axis=0) for s, cs in curves.items()}


import torch as _torch

def torch_no_grad_marker(fn):
    def wrapper(*a, **k):
        import torch
        with torch.no_grad():
            return fn(*a, **k)
    return wrapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["invideo", "sanity", "loeo"], required=True)
    ap.add_argument("--stems", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[13, 21, 42])
    a = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if a.mode == "invideo":
        for stem in a.stems:
            v = load_stem(stem)
            if v is None:
                print(f"[miss] {stem}")
                continue
            n = len(v["feat"])
            cut = int(n * 0.6)
            tr = [{"feat": v["feat"][:cut], "lab": v["lab"][:cut]}]
            for seed in a.seeds:
                m = build_probe(seed=seed).to(dev)
                train_probe(m, tr, seed, dev=dev)
                pr = curve(m, v["feat"][cut:], dev)
                p, r, nd, ng = seg_eval((5, 0.6, 0.4, 1.5, 1.2), v["cent"][cut:], pr, v["segs"])
                lo = v["lab"][cut:] == 0
                print(f"[invideo:{stem} seed={seed}] P={p:.3f} R={r:.3f} 段={nd}/{ng} "
                      f"p_rally={pr[~lo].mean():.2f} p_pause={pr[lo].mean():.2f}")
        return

    if a.mode == "sanity":
        dji = [s for s in VENUES["dji"]]
        tr_stems, te_stems = dji[:10], dji[10:]
        tr = [v for v in (load_stem(s) for s in tr_stems) if v]
        te = [v for v in (load_stem(s) for s in te_stems) if v]
        probs = ensemble_curves(tr + te, a.seeds, dev)
        score, par = grid_on([(v["cent"], probs[v["stem"]], v["segs"]) for v in tr])
        print(f"[sanity] 分割参数={par} 训练侧={score:.3f}")
        for v in te:
            p, r, nd, ng = seg_eval(par, v["cent"], probs[v["stem"]], v["segs"])
            print(f"[sanity] {v['stem']}: P={p:.3f} R={r:.3f} 段={nd}/{ng}")
        return

    results = {}
    for hold in VENUES:
        tr_stems = [s for v2, ss in VENUES.items() if v2 != hold for s in ss]
        te_stems = VENUES[hold]
        tr = [v for v in (load_stem(s) for s in tr_stems) if v]
        if len(tr) < len(tr_stems) - 0:
            missing = len(tr_stems) - len(tr)
            print(f"[wait] fold={hold}: 缺 {missing} 条特征,跳过(提取未完)")
            continue
        fold = []
        te = [v for v in (load_stem(s) for s in te_stems) if v]
        probs = ensemble_curves(tr + te, a.seeds, dev)
        tr_items = [(v["cent"], probs[v["stem"]], v["segs"]) for v in tr]
        score, par = grid_on(tr_items)
        print(f"[loeo hold={hold}] 分割参数={par} 训练侧={score:.3f}")
        for v in te:
            p, r, nd, ng = seg_eval(par, v["cent"], probs[v["stem"]], v["segs"])
            fold.append({"stem": v["stem"], "P": p, "R": r, "n_det": nd, "n_gt": ng})
            print(f"[loeo hold={hold}] {v['stem']}: "
                  f"P={p:.3f} R={r:.3f} 段={nd}/{ng}", flush=True)
        results[hold] = fold
    Path("temp/loeo_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("== 汇总(各 fold 平均) ==")
    for hold, rows in results.items():
        ps = np.mean([r["P"] for r in rows])
        rs = np.mean([r["R"] for r in rows])
        print(f"{hold}: P={ps:.3f} R={rs:.3f}")


if __name__ == "__main__":
    main()
