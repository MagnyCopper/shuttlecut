import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shuttlecut.sampler import extract_frames, probe


def make_contact_sheets(video: str, outdir: str, step_s: float = 2.0,
                        cols: int = 5, rows: int = 6) -> list[str]:
    """每张 sheet 含 cols*rows 格(默认 60s),左上到右下按时间排列。"""
    meta = probe(video)
    work = Path("temp") / "label" / Path(video).stem / "_sheet_frames"
    frames = extract_frames(video, str(work), fps=1.0 / step_s, width=480)
    per = cols * rows
    sheets: list[str] = []
    for si in range(math.ceil(len(frames) / per)):
        chunk = frames[si * per : (si + 1) * per]
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2.7))
        for ax, f in zip(axes.flat, chunk):
            t = (si * per + list(frames).index(f) + 1) * step_s
            ax.imshow(plt.imread(f))
            ax.set_title(f"{t:.0f}s", fontsize=9)
            ax.axis("off")
        for ax in axes.flat[len(chunk):]:
            ax.axis("off")
        fig.suptitle(f"{Path(video).name} sheet {si + 1} "
                     f"({si * per * step_s:.0f}s+)", fontsize=12)
        plt.tight_layout()
        p = Path(outdir) / f"sheet_{si + 1:03d}.jpg"
        fig.savefig(p, dpi=90)
        plt.close(fig)
        sheets.append(str(p))
    return sheets


def make_dense_strip(video: str, t0: float, t1: float, outpath: str,
                     step_s: float = 0.2) -> str:
    """[t0,t1] 每 step_s 一帧,2 行网格,时间戳到 0.1s。"""
    work = Path("temp") / "label" / Path(video).stem / f"_strip_{t0:.0f}_{t1:.0f}"
    frames = extract_frames(video, str(work), fps=1.0 / step_s, width=480,
                            t_start=t0, t_end=t1)
    cols = math.ceil(len(frames) / 2) or 1
    fig, axes = plt.subplots(2, cols, figsize=(cols * 2.6, 4.2), squeeze=False)
    for ax, f in zip(axes.flat, frames):
        idx = int(Path(f).stem.split("_")[-1]) - 1
        ax.imshow(plt.imread(f))
        ax.set_title(f"{t0 + idx * step_s:.1f}s", fontsize=8)
        ax.axis("off")
    for ax in axes.flat[len(frames):]:
        ax.axis("off")
    plt.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=100)
    plt.close(fig)
    return outpath
