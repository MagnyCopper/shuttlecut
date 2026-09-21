"""V-JEPA 2 冻结特征提取(LOEO 实验第一步)。

对每条视频的 frames15 缓存做 64 帧滑窗(步长 8)→ VJEPA2-ViT-L 编码 →
空间 token 均值池化 → temp/work/<stem>/vjepa_feat.npy (N×1024, f16)。
幂等:已有缓存即跳过。GPU 长任务,走分离进程 + 日志轮询。
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import VJEPA2Model

MODEL_ID = "facebook/vjepa2-vitl-fpc16-256-ssv2"
WIN = 16
STRIDE = 8
FPS = 15
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# 提取顺序:先小后大,让主 fold(训练=DJI)最早可用
ORDER = [
    "DJI_20260912150541_0007_D", "DJI_20260912151553_0008_D",
    "DJI_20260912152303_0009_D", "DJI_20260912160351_0010_D",
    "DJI_20260912164500_0011_D", "DJI_20260912174203_0013_D",
    "DJI_20260912180439_0015_D", "DJI_20260830153830_0015_D",
    "DJI_20260905151437_0030_D", "DJI_20260905154319_0033_D",
    "DJI_20260905161114_0034_D", "DJI_20260905170125_0036_D",
    "DJI_20260830173600_0025_D", "DJI_20260912174644_0014_D",
    "linzhou", "baoganghui_p1", "baoganghui_p2", "jiguang", "yygq",
]


def load_window(frames: list[str], s: int, win: int = WIN) -> np.ndarray:
    xs = np.zeros((win, 256, 256, 3), np.float32)
    for k in range(win):
        img = cv2.imread(frames[s + k], cv2.IMREAD_COLOR)
        xs[k] = cv2.resize(img, (256, 256))
    xs = xs[..., ::-1]  # BGR→RGB
    return (xs / 255.0 - MEAN) / STD


class WinDS(Dataset):
    """模块级:Windows spawn DataLoader 需要 pickle。"""

    def __init__(self, frames, starts, win=WIN):
        self.frames, self.starts, self.win = frames, starts, win

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        return torch.from_numpy(
            load_window(self.frames, s, self.win).transpose(0, 3, 1, 2).copy())  # (T,C,H,W);HF 取 (B,T,C,H,W)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--win", type=int, default=WIN)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--stems", nargs="*", default=None, help="只处理这些 stem(默认全部)")
    a = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = VJEPA2Model.from_pretrained(a.model, dtype=torch.float16).to(dev).eval()
    model = VJEPA2Model.from_pretrained(a.model, dtype=torch.float16).to(dev).eval()

    stems = a.stems or ORDER
    for stem in stems:
        out = Path(f"temp/work/{stem}/vjepa_feat_w{a.win}.npy")
        if out.exists():
            print(f"[skip] {stem}", flush=True)
            continue
        fdir = Path(f"temp/work/{stem}/frames15")
        frames = sorted(str(p) for p in fdir.glob("frame_*.jpg"))
        if len(frames) < a.win:
            print(f"[warn] {stem}: 仅 {len(frames)} 帧 < {WIN},跳过", flush=True)
            continue
        starts = list(range(0, len(frames) - a.win + 1, a.stride))
        dl = DataLoader(WinDS(frames, starts, a.win), batch_size=a.batch,
                        num_workers=6, pin_memory=True)
        feats, centers, t0 = [], [], time.time()
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            for bi, xb in enumerate(dl):
                xb = xb.to(dev, non_blocking=True)
                h = model(xb, skip_predictor=True).last_hidden_state  # (B,tok,1024)
                feats.append(h.mean(dim=1).float().cpu().numpy())
                if bi % 50 == 0:
                    done = bi * a.batch
                    rate = done / max(time.time() - t0, 1e-6)
                    print(f"[{stem}] {done}/{len(starts)} "
                          f"({rate:.1f} win/s, ETA {(len(starts) - done) / max(rate, .01) / 60:.0f}m)",
                          flush=True)
        np.save(out, np.concatenate(feats).astype(np.float16))
        np.save(out.with_name(f"vjepa_centers_w{a.win}.npy"),
                np.array([(s + a.win / 2) / FPS for s in starts], np.float32))
        print(f"[done] {stem}: {len(starts)} windows → {out.name} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
