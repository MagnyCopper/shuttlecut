"""Temporal rally segmentation: R3D window classifier over compensated frame diffs.

Pipeline pieces (pure/testable except model inference):
- build_diffs(store): global-motion-compensated frame differences.
- segment_curve / split_merged / two_scale_segments: hysteresis + fine-curve splitting.
- infer_curve(frames_dir, ckpt): sliding-window R3D probabilities (torch, MPS/CUDA/CPU).

Model naming convention: models/r3d_{stem}_w64.pt (+ optional models/r3d_{stem}_w24.pt).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

H, W = 90, 160
TWO_SCALE_DEFAULTS = {"hi": 0.4, "lo": 0.1, "min_len_s": 1.5, "split_th": 0.2, "min_sub_s": 1.5,
               "sm": 0, "mg": 0.0}  # sm=median 平滑窗(曲线点数), mg=邻接段合并间隙(s);由训练侧选参定默认


def build_diffs(store: np.ndarray) -> np.ndarray:
    """Global-motion(LK+RANSAC translation)-compensated |frame diffs|, /32 normalized."""
    n, h, w = store.shape
    diffs = np.zeros_like(store, np.float32)
    for i in range(1, n):
        a, b = store[i - 1], store[i]
        raw = np.abs(b.astype(np.float32) - a.astype(np.float32)) / 32.0
        pa = cv2.goodFeaturesToTrack(a, 200, 0.01, 8)
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


def segment_curve(centers: np.ndarray, probs: np.ndarray, hi: float, lo: float,
                  min_len_s: float) -> list[tuple[float, float]]:
    """Hysteresis segmentation over a probability curve. centers are window-centre times."""
    pred: list[tuple[float, float]] = []
    armed, start = False, 0.0
    for t, p in zip(centers, probs):
        if not armed and p > hi:
            armed, start = True, float(t)
        elif armed and p < lo:
            armed = False
            if t - start >= min_len_s:
                pred.append((start, float(t)))
    if armed and centers[-1] - start >= min_len_s:
        pred.append((start, float(centers[-1])))
    return pred


def _median(xs: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return xs
    pad = k // 2
    out = np.copy(xs)
    for i in range(len(xs)):
        out[i] = np.median(xs[max(0, i - pad):i + pad + 1])
    return out


def split_merged(fine_centers: np.ndarray, fine_probs: np.ndarray, s: float, e: float,
                 th: float, min_sub_s: float) -> list[tuple[float, float]]:
    """Split [s,e] at deep local minima of the fine (short-window) curve."""
    m = (fine_centers >= s) & (fine_centers <= e)
    if int(m.sum()) < 9:
        return [(s, e)]
    tf = fine_centers[m]
    q = _median(fine_probs[m].astype(np.float64), 7)
    out: list[tuple[float, float]] = []
    st = s
    for k in range(2, len(tf) - 2):
        if q[k] < th and q[k] <= q[k - 1] and q[k] <= q[k + 1] \
                and tf[k] - st >= min_sub_s and e - tf[k] >= min_sub_s:
            out.append((st, float(tf[k])))
            st = float(tf[k])
    out.append((st, e))
    return out


def two_scale_segments(coarse_centers: np.ndarray, coarse_probs: np.ndarray,
                       fine_centers: np.ndarray | None, fine_probs: np.ndarray | None,
                       **params) -> list[tuple[float, float]]:
    """W64 hysteresis + optional W24 deep-valley splitting (the validated recipe)."""
    cfg = {**TWO_SCALE_DEFAULTS, **params}
    probs = coarse_probs
    if cfg["sm"] > 1:
        from scipy.ndimage import median_filter
        probs = median_filter(coarse_probs, size=cfg["sm"])
    base = segment_curve(coarse_centers, probs, cfg["hi"], cfg["lo"], cfg["min_len_s"])
    if cfg["mg"] > 0.0:
        merged: list[tuple[float, float]] = []
        for a, b in base:
            if merged and a - merged[-1][1] <= cfg["mg"]:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        base = merged
    if fine_centers is None or fine_probs is None:
        return base
    out: list[tuple[float, float]] = []
    for a, b in base:
        out.extend(split_merged(fine_centers, fine_probs, a, b, cfg["split_th"], cfg["min_sub_s"]))
    return out


def boundary_vote(seg_lists: list[list[tuple[float, float]]],
                 min_gap_s: float = 1.0) -> list[tuple[float, float]]:
    """多模型边界投票:以首模型段为锚,各模型重叠段边界取中位数(实验验证的集成配方)。"""
    if not seg_lists or not seg_lists[0]:
        return []
    voted = []
    for a, b in seg_lists[0]:
        starts, ends = [a], [b]
        for segs in seg_lists[1:]:
            ov = [(x, y) for x, y in segs if min(y, b) - max(x, a) > 0.25 * (b - a)]
            if ov:
                x, y = max(ov, key=lambda q: q[1] - q[0])
                starts.append(x)
                ends.append(y)
        na, nb = float(np.median(starts)), float(np.median(ends))
        if nb - na >= 2.5:
            voted.append((na, nb))
    merged: list[list[float]] = []
    for s, e in sorted(voted):
        if merged and s <= merged[-1][1] + min_gap_s:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(float(s), float(e)) for s, e in merged]



def extract_frames15(video: str, out_dir: str) -> list[str]:
    """Cached 15fps/960-wide frame extraction for temporal models."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    frames = sorted(str(p) for p in d.glob("frame_*.jpg"))
    if frames:
        return frames
    from shuttlecut.ffmpeg import media_run
    media_run(lambda dec, _: [
        "ffmpeg", "-loglevel", "error", "-y", *dec, "-i", video,
        "-vf", "fps=15,scale=960:-2", "-q:v", "3", str(d / "frame_%06d.jpg")])
    return sorted(str(p) for p in d.glob("frame_*.jpg"))


def load_diffs(frames: list[str], cache_path: str | None = None,
                 progress=None) -> np.ndarray:
    """帧→差分;cache_path 命中则直接加载(f16),否则计算并可保存。多模型共享。"""
    if cache_path and Path(cache_path).exists():
        return np.load(cache_path).astype(np.float32)
    store = np.zeros((len(frames), H, W), np.uint8)
    for i, f in enumerate(frames):
        store[i] = cv2.resize(cv2.imread(f, cv2.IMREAD_GRAYSCALE), (W, H))
        if progress and (i + 1) % 500 == 0:
            progress(f"帧读取 {i + 1}/{len(frames)}")
    diffs = build_diffs(store)
    if cache_path:
        np.save(cache_path, diffs.astype(np.float16))
    return diffs


def infer_curve(frames: list[str], ckpt: str, win: int = 64, tsub: int = 4,
                batch: int = 8, device: str = "auto", diffs: np.ndarray | None = None,
                progress=None) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window R3D inference → (centres_s, probs). Heavy: torch import deferred."""
    import torch

    if diffs is None:
        diffs = load_diffs(frames)

    if device == "auto":
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
    import torch.nn as nn
    import torchvision
    model = torchvision.models.video.r3d_18()
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model = model.to(device).eval()

    starts = list(range(0, len(frames) - win + 1, 2))
    probs, centers = [], []
    nbatch = (len(starts) + batch - 1) // batch
    done = 0
    with torch.no_grad():
        for k in range(0, len(starts), batch):
            xs = []
            for s in starts[k:k + batch]:
                d = diffs[s:s + win].astype(np.float32)
                if win // tsub > 8:
                    d = d[::tsub]
                f112 = np.stack([cv2.resize(x, (112, 112)) for x in d])
                xs.append(np.repeat(f112[None], 3, axis=0))
            x = torch.from_numpy(np.ascontiguousarray(np.stack(xs))).to(device)
            p = torch.sigmoid(model(x)).squeeze(-1).float().cpu().numpy().reshape(-1)
            probs.extend(p.tolist())
            centers.extend([(s + win / 2) / 15.0 for s in starts[k:k + batch]])
            done += 1
            if progress and (done % 25 == 0 or done == nbatch):
                progress(f"{done * 100 // nbatch}%")
    return np.array(centers), np.array(probs, dtype=np.float64)
