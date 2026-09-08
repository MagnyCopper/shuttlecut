"""Autonomous labeling: model candidates + 9-frame vision triage + audio tie-break.

Usage:
    python tools/autolabel/label.py --video temp/X.MP4 --ckpt models/r3d_joint_w64.pt
Produces:
    temp/autolabel_<stem>/segments.json   (final segments, confidence flagged)
    temp/autolabel_<stem>/vision/seg_*.jpg (9-frame grids for spot checks)

Pipeline per candidate segment:
  1. model prob (joint model curve mean in span)
  2. vision vote: 9-frame 3x3 grid judged by multimodal vision (~85% acc)
  3. audio vote (only on disagreement): transient density vs adaptive threshold
  4. agreement of 2/3 -> accept; all-disagree -> flag "review"
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def build_grids(frames_dir: Path, segs: list[tuple[float, float]], out: Path) -> None:
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    out.mkdir(parents=True, exist_ok=True)
    for i, (a, b) in enumerate(segs, start=1):
        tiles = []
        for k in range(9):
            t = a + (b - a) * k / 8
            f = frames[min(int(t * 15), len(frames) - 1)]
            img = cv2.resize(cv2.imread(str(f)), (426, 240))
            if k == 0:
                cv2.putText(img, f"SEG {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            tiles.append(img)
        grid = np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6]), np.hstack(tiles[6:9])])
        cv2.imwrite(str(out / f"seg_{i:02d}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])


def audio_rates(wav: Path, segs: list[tuple[float, float]]) -> list[float]:
    from shuttlecut.audio import audio_transients
    tr = audio_transients(str(wav))
    ts = np.array([x.t for x in tr]) if tr else np.array([0.0])
    return [float(((ts >= a) & (ts <= b)).sum() / max(b - a, 0.1)) for a, b in segs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--win", type=int, default=64)
    args = ap.parse_args()
    stem = Path(args.video).stem
    work = ROOT / "temp/work" / stem
    outdir = ROOT / "temp" / f"autolabel_{stem}"
    outdir.mkdir(parents=True, exist_ok=True)

    from shuttlecut.temporal import extract_frames15, infer_curve, two_scale_segments
    frames = extract_frames15(args.video, str(work / "frames15"))
    centers, probs = infer_curve(frames, args.ckpt, win=args.win)
    segs = two_scale_segments(centers, probs, None, None)
    seg_mean = lambda c, p, a, b: float(p[(c >= a) & (c <= b)].mean()) if ((c >= a) & (c <= b)).any() else 0.0
    model_votes = [1 if seg_mean(centers, probs, a, b) > 0.5 else 0 for a, b in segs]

    build_grids(work / "frames15", segs, outdir / "vision")

    wav = work / "audio.wav"
    if not wav.exists():
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", args.video,
                        "-vn", "-ac", "1", "-ar", "16000", str(wav)], check=True)
    rates = audio_rates(wav, segs)
    th_lo = float(np.percentile([r for r, v in zip(rates, model_votes) if v == 1], 25)) if any(model_votes) else 0.7
    th_hi = float(np.percentile([r for r, v in zip(rates, model_votes) if v == 0], 75)) if (~np.array(model_votes, bool)).any() else 0.7
    audio_votes = [1 if r > (th_lo + th_hi) / 2 else 0 for r in rates]

    result = []
    for i, ((a, b), mv, av, r) in enumerate(zip(segs, model_votes, audio_votes, rates), start=1):
        votes = {"model": mv, "audio": av}
        # vision vote 由调用方(视觉智能)对 outdir/vision/*.jpg 逐张判定后回填;
        # 此处先输出待判清单与 model/audio 结论。
        result.append({"id": i, "start_s": round(a, 1), "end_s": round(b, 1),
                       "model": mv, "audio": av, "audio_rate": round(r, 2),
                       "vision": None, "final": None})
    json.dump(result, open(outdir / "segments.json", "w"), ensure_ascii=False, indent=1)
    print(f"{len(segs)} 段;视觉判定图 → {outdir}/vision/(待视觉智能回填 vision 字段)")


if __name__ == "__main__":
    main()
