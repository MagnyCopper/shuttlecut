"""V-JEPA 2 冻结特征 + Transformer 探针:跨场馆零校准回合检测(路线 20 产品化)。

LOEO 实测 5 场馆均值 P/R 0.83/0.80(纯 DJI 训→B站 0.85/0.84),对比 R3D 零训练
跨场馆 0.1-0.5。产物:models/shuttlecut-probe.pt(3 种子集成+固化分割参数),
编码器 facebook/vjepa2-vitl-fpc16-256-ssv2(HF 自动下载至 models/hf,1.2GB,一次性)。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

PROBE_PATH = "models/shuttlecut-probe.pt"
ENCODER_ID = "facebook/vjepa2-vitl-fpc16-256-ssv2"
WIN, STRIDE, FPS = 16, 8, 15
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def ensure_probe(progress=None) -> str | None:
    """探针权重缺失时按 manifest probe 条目自动下载;不可用返回 None。"""
    if Path(PROBE_PATH).exists():
        return PROBE_PATH
    try:
        from shuttlecut.modelhub import download_model, load_manifest
        m = load_manifest()
        p = m.get("probe") if isinstance(m.get("probe"), dict) else None
        if not p or not p.get("url"):
            return None
        if progress:
            progress(f"[model] 首次使用:下载探针 {p['version']}({p['bytes'] >> 20} MB)…")
        download_model(PROBE_PATH, p, progress=progress)
        return PROBE_PATH
    except Exception as e:
        Path(PROBE_PATH + ".part").unlink(missing_ok=True)
        if progress:
            progress(f"[warn] 探针自动下载失败({e})")
        return None


def _load_encoder(device: str, progress=None):
    os.environ.setdefault("HF_HOME", str(Path("models/hf").resolve()))
    import torch
    from transformers import VJEPA2Model
    dtype = torch.float16 if device == "cuda" else torch.float32
    if progress:
        progress(f"[2/5] V-JEPA 编码器加载(首次需下载 1.2GB 至 models/hf)…")
    m = VJEPA2Model.from_pretrained(ENCODER_ID, dtype=dtype)
    return m.to(device).eval()


def extract_features(frames: list[str], cache_path: str, device: str = "auto",
                     progress=None) -> tuple[np.ndarray, np.ndarray]:
    """16 帧滑窗(步长 8)→ 编码器空间均值池化 → (feat f32 [N,1024], centers [N])。幂等缓存。"""
    import cv2
    import torch
    if Path(cache_path).exists():
        return (np.load(cache_path).astype(np.float32),
                np.load(str(cache_path).replace("vjepa_feat", "vjepa_centers")))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_encoder(device, progress)
    starts = list(range(0, len(frames) - WIN + 1, STRIDE))
    feats: list[np.ndarray] = []
    t_last = [0.0]
    import time
    t0 = time.time()
    batch = 8
    with torch.no_grad():
        for k in range(0, len(starts), batch):
            xs = np.zeros((min(batch, len(starts) - k), WIN, 256, 256, 3), np.float32)
            for bi, s in enumerate(starts[k:k + batch]):
                for w in range(WIN):
                    img = cv2.imread(frames[s + w], cv2.IMREAD_COLOR)
                    xs[bi, w] = cv2.resize(img, (256, 256))
            xs = xs[..., ::-1]  # BGR→RGB
            xs = (xs / 255.0 - _MEAN) / _STD
            x = torch.from_numpy(xs.transpose(0, 1, 4, 2, 3).copy()).to(device)  # (B,T,C,H,W)
            if device == "cuda":
                with torch.autocast("cuda", torch.float16):
                    h = model(x, skip_predictor=True).last_hidden_state
            else:
                h = model(x, skip_predictor=True).last_hidden_state
            feats.append(h.mean(dim=1).float().cpu().numpy())
            if progress and time.time() - t_last[0] > 10:
                t_last[0] = time.time()
                done = min(k + batch, len(starts))
                rate = done / max(time.time() - t0, 1e-6)
                progress(f"[2/5] V-JEPA 特征 {done}/{len(starts)}"
                         f"(ETA {(len(starts) - done) / max(rate, .01) / 60:.0f}m)")
    feat = np.concatenate(feats).astype(np.float16)
    cent = np.array([(s + WIN / 2) / FPS for s in starts], np.float32)
    np.save(cache_path, feat)
    np.save(str(cache_path).replace("vjepa_feat", "vjepa_centers"), cent)
    return feat.astype(np.float32), cent


def _snap(pred: list[tuple[float, float]], cent: np.ndarray, probs: np.ndarray,
          win_s: float = 1.5) -> list[tuple[float, float]]:
    """边界向 ±win_s 内最大 |dp/dt| 点吸附(边界精修,实测跨场馆 +0.02)。"""
    from scipy.ndimage import median_filter
    pp = median_filter(probs, size=5)
    slope = np.gradient(pp, cent)
    out = []
    for a, b in pred:
        na, nb = a, b
        for cand, sign, is_start in ((a, 1.0, True), (b, -1.0, False)):
            m = (cent >= cand - win_s) & (cent <= cand + win_s)
            if m.sum() < 3:
                continue
            idx = np.where(m)[0]
            k = idx[int(np.argmax(slope[idx] * sign))]
            if is_start:
                na = float(cent[k])
            else:
                nb = float(cent[k])
        if nb - na >= 1.0:
            out.append((na, nb))
    return out


def detect(video: str, work_dir: str, device: str = "auto", progress=None):
    """完整探针路径:视频 → (centers, probs, rallies)。需 frames15 已提取。"""
    import torch
    from shuttlecut.temporal import extract_frames15, segment_curve
    from scipy.ndimage import median_filter

    ck = ensure_probe(progress)
    if not ck:
        raise RuntimeError("探针权重不可用(models/shuttlecut-probe.pt)")
    blob = torch.load(ck, map_location="cpu")
    meta = blob["meta"]
    sm, hi, lo, ml, mg = blob["seg"]
    frames = extract_frames15(video, str(Path(work_dir) / "frames15"))
    if len(frames) < WIN:
        raise RuntimeError(f"视频过短({len(frames)} 帧 < {WIN})")
    feat, cent = extract_features(frames, str(Path(work_dir) / "vjepa_feat_w16.npy"),
                                  device=device, progress=progress)
    feat = (feat - feat.mean(0, keepdims=True)) / (feat.std(0, keepdims=True) + 1e-6)

    import torch.nn as nn
    probes = []
    for sd in blob["probes"]:
        d, L = meta["d"], meta["layers"]
        enc = nn.TransformerEncoderLayer(d, 4, dim_feedforward=512, dropout=0.1,
                                         batch_first=True, norm_first=True)
        p = nn.Sequential(nn.Linear(feat.shape[1], d), nn.GELU(),
                          nn.TransformerEncoder(enc, L), nn.Linear(d, 1))
        p.load_state_dict(sd)
        probes.append(p.eval())
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    probs_sum = np.zeros(len(cent), np.float64)
    chunk, ov = 1024, 128
    with torch.no_grad():
        for p in probes:
            p = p.to(dev)
            acc = np.zeros(len(cent), np.float64)
            cnt = np.zeros(len(cent))
            for s in range(0, len(cent), chunk - ov):
                e = min(s + chunk, len(cent))
                x = torch.from_numpy(feat[s:e][None]).to(dev)
                out = torch.sigmoid(p(x)).squeeze(-1)[0].cpu().numpy()
                acc[s:e] += out
                cnt[s:e] += 1
                if e == len(cent):
                    break
            probs_sum += acc / cnt
    probs = probs_sum / len(probes)
    pred = segment_curve(cent, median_filter(probs, size=sm) if sm > 1 else probs,
                         hi, lo, ml)
    merged = []
    for a, b in pred:
        if merged and a - merged[-1][1] <= mg:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    rallies = _snap(merged, cent, probs, meta.get("snap_win_s", 1.5))
    return cent, probs, rallies
