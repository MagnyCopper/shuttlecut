"""CUDA 机数据准备:从原始 MP4 + GT 抽 15fps 帧目录(与本项目 temp/work 布局一致)。

用法:
    python prep_data.py --video DJI_20260830153830_0015_D.MP4 --gt ground_truth/DJI_20260830153830_0015_D.json --out data/
产出:
    data/<stem>/frames15/frame_%06d.jpg (960 宽)
"""
import argparse
import json
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    stem = Path(a.video).stem
    gt = json.load(open(a.gt))
    assert gt["video"] == stem or len(gt["rallies"]) > 0
    d = Path(a.out) / stem / "frames15"
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", a.video,
                    "-vf", "fps=15,scale=960:-2", "-q:v", "3", str(d / "frame_%06d.jpg")], check=True)
    print(f"OK {d} ({len(list(d.glob('*.jpg')))} frames)")


if __name__ == "__main__":
    main()
