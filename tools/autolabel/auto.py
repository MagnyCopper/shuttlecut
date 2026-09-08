"""自主标注 v2:模型候选 → 9 帧网格渲染 → Agent look_at 视觉裁定 → GT 合成。

三段式架构(人工视觉由 Agent look_at 替代,响应用户目标):
1. prepare: 对模型候选段渲染 3×3 网格图 + manifest.json(段元数据)
2. judge:   Agent 对每张网格 look_at 裁定 playing/rest/uncertain,写 verdicts.json
3. merge:   manifest + verdicts + 曲线体检 → data/ground_truth/<stem>.json

用法:
    python tools/autolabel/auto.py prepare --frames temp/work/<stem>/frames15 \
        --curves artifacts/curves/prob_<tag> --stem <stem> --out temp/autolabel/<stem>
    # (Agent look_at 裁定后)
    python tools/autolabel/auto.py merge --stem <stem> --in temp/autolabel/<stem>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

GRID, CELL = 3, 320


def pick_indices(n: int, k: int = 9) -> list[int]:
    """段内均匀取 k 帧(含首尾),返回全局帧索引偏移规则:相对段内。"""
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    pos = np.linspace(0, n - 1, k).round().astype(int)
    return sorted(set(int(i) for i in pos))


def render_grid(frames: list[str], idxs: list[int], out_path: str) -> bool:
    """3×3 网格图;frames 为全视频帧路径列表,idxs 为段内帧下标。"""
    cells = []
    for i in idxs:
        img = cv2.imread(frames[i])
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = CELL / max(h, w)
        cells.append(cv2.resize(img, (int(w * scale), int(h * scale))))
    if not cells:
        return False
    canvas = np.full((CELL * GRID, CELL * GRID, 3), 24, np.uint8)
    for j, c in enumerate(cells[:GRID * GRID]):
        r, col = divmod(j, GRID)
        y, x = r * CELL + (CELL - c.shape[0]) // 2, col * CELL + (CELL - c.shape[1]) // 2
        canvas[y:y + c.shape[0], x:x + c.shape[1]] = c
    cv2.imwrite(out_path, canvas)
    return True


def segment_frames(segs: list[tuple[float, float]]) -> list[tuple[int, int]]:
    """(start_s,end_s) → (start_idx,end_idx) @15fps,clamp 到 ≥0。"""
    return [(max(0, int(round(s * 15))), max(0, int(round(e * 15)))) for s, e in segs]


def prepare(frames_dir: str, curves_tag: str, stem: str, out_dir: str) -> None:
    from shuttlecut.temporal import two_scale_segments

    frames = sorted(str(p) for p in Path(frames_dir).glob("frame_*.jpg"))
    centers = np.load(f"artifacts/curves/prob_centers_{curves_tag}.npy")
    probs = np.load(f"artifacts/curves/prob_values_{curves_tag}.npy")
    segs = two_scale_segments(centers, probs, None, None)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, (s, e) in enumerate(segs):
        si, ei = segment_frames([(s, e)])[0]
        idxs = [min(si + d, len(frames) - 1) for d in pick_indices(max(ei - si, 1))]
        img = str(out / f"grid_{i:03d}_{s:.0f}-{e:.0f}s.jpg")
        ok = render_grid(frames, idxs, img)
        m = (centers >= s) & (centers <= e)
        manifest.append({
            "i": i, "start_s": round(s, 2), "end_s": round(e, 2),
            "dur": round(e - s, 1), "mean_p": float(probs[m].mean()) if m.any() else 0.0,
            "grid": img if ok else None,
        })
    (out / "manifest.json").write_text(json.dumps({"stem": stem, "segments": manifest},
                                                  ensure_ascii=False, indent=1))
    print(f"prepare: {len(manifest)} 段候选,网格图 → {out}")


def merge(stem: str, in_dir: str, gt_out: str | None = None) -> None:
    d = Path(in_dir)
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    verdicts = json.loads((d / "verdicts.json").read_text(encoding="utf-8"))
    segs = []
    for m in manifest["segments"]:
        v = verdicts.get(str(m["i"]), "uncertain")
        if v == "playing":
            segs.append({"start_s": m["start_s"], "end_s": m["end_s"]})
    gt_out = gt_out or f"data/ground_truth/{stem}.json"
    Path(gt_out).parent.mkdir(parents=True, exist_ok=True)
    Path(gt_out).write_text(json.dumps({
        "video": stem, "source": "autolabel-v2(look_at)",
        "rallies": segs,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    kept = len(segs)
    print(f"merge: {kept}/{len(manifest['segments'])} 段判 playing → {gt_out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    p1 = sub.add_parser("prepare")
    p1.add_argument("--frames", required=True)
    p1.add_argument("--curves", required=True, help="曲线 tag(artifacts/curves/prob_*_<tag>.npy)")
    p1.add_argument("--stem", required=True)
    p1.add_argument("--out", required=True)
    p2 = sub.add_parser("merge")
    p2.add_argument("--stem", required=True)
    p2.add_argument("--in", dest="indir", required=True)
    p2.add_argument("--gt-out", default=None)
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.frames, a.curves, a.stem, a.out)
    elif a.cmd == "merge":
        merge(a.stem, a.indir, a.gt_out)


if __name__ == "__main__":
    main()
