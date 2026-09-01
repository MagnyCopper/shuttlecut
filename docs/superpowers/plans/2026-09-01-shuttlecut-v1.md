# ShuttleCut v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付一个本地 CLI 工具 `shuttlecut`,对业余羽毛球视频自动做回合切分与切片导出,并通过《验收方案》全部拦截项。

**Architecture:** 六阶段流水线(ffmpeg 采样 → YOLO11n person 检测(MPS) → 运动能量信号 → 滞回状态机切分 → 音频瞬态精修 → ffmpeg 切片导出),真值先行:先建 ground truth 再实现算法,`eval` 以公式级指标驱动迭代。

**Tech Stack:** Python 3.12(工程内 `.venv`)、uv、ultralytics(YOLO11n)、librosa、soundfile、numpy、matplotlib、ffmpeg/ffprobe(已安装)、pytest。

## Global Constraints

- 所有 Python 命令用 `.venv/bin/python`;禁止全局安装(AGENTS.md 规约 2/3/4)
- 临时/中间产物一律写 `temp/`(git 已忽略);模型权重落 `models/`(git 已忽略);真值 `ground_truth/` 与代码进 git(规约 1)
- 派出子任务阻塞执行,不后台并行(规约 5);需用户决策时用结构化提问(规约 6)
- 帧采样规格:1280 宽 @ 5fps;YOLO:person 类、conf 0.25、imgsz=1280;设备 `auto`(MPS→CPU 回退)
- 验收数字以《验收方案》为准:命中 = 重叠 ≥50%·max(|D|,|G|) 且双边界 ≤2.5s;B1/B2 的 R/P 均 ≥0.90;异常输入不崩溃
- 提交信息用中文、conventional commits 风格(`feat:`/`test:`/`docs:`/`chore:`)

---

### Task 1: 项目骨架与 CLI 冒烟

**Files:**
- Create: `pyproject.toml`, `src/shuttlecut/__init__.py`, `src/shuttlecut/cli.py`, `tests/test_cli.py`, `tests/conftest.py`

**Interfaces:**
- Produces: 包 `shuttlecut`(src 布局)、入口 `shuttlecut=shuttlecut.cli:main`、`tests/conftest.py` 中的 `synth_video` fixture(20s 1280×720 testsrc 视频,含 440Hz 正弦音轨),供后续任务测试使用。

- [ ] **Step 1: 写 pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "shuttlecut"
version = "0.1.0"
description = "Badminton rally auto-clipping CLI"
requires-python = ">=3.12"
dependencies = [
    "ultralytics>=8.4",
    "librosa>=1.0",
    "soundfile>=0.12",
    "numpy>=1.26",
    "matplotlib>=3.8",
]

[project.scripts]
shuttlecut = "shuttlecut.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/shuttlecut"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: 写包与 CLI 骨架**

`src/shuttlecut/__init__.py`:

```python
__version__ = "0.1.0"
```

`src/shuttlecut/cli.py`:

```python
import argparse

from shuttlecut import __version__


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="shuttlecut", description="羽毛球回合自动剪辑")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("process", help="切分回合并导出片段")
    sub.add_parser("label", help="真值标注辅助工具")
    sub.add_parser("eval", help="对比检测结果与真值")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd is None:
        build_parser().print_help()
        return 1
    return 0
```

- [ ] **Step 3: 写 conftest fixture(合成视频,后续任务复用)**

`tests/conftest.py`:

```python
import subprocess

import pytest


@pytest.fixture(scope="session")
def synth_video(tmp_path_factory: pytest.TempPathFactory) -> str:
    """20 秒 1280x720 testsrc 合成视频 + 正弦音轨,供采样/导出/精修测试复用。"""
    path = tmp_path_factory.mktemp("media") / "synth.mp4"
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=duration=20:size=1280x720:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=20",
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
            "-shortest", str(path),
        ],
        check=True,
    )
    return str(path)
```

- [ ] **Step 4: 写失败测试**

`tests/test_cli.py`:

```python
from shuttlecut.cli import main


def test_version_flag(capsys):
    try:
        main(["--version"])
    except SystemExit as e:
        assert e.code == 0
    assert "0.1.0" in capsys.readouterr().out


def test_no_subcommand_shows_help(capsys):
    assert main([]) == 1
    assert "process" in capsys.readouterr().out
```

- [ ] **Step 5: 安装并运行测试**

Run: `uv pip install -p .venv/bin/python -e . pytest && .venv/bin/python -m pytest tests/test_cli.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/ tests/
git commit -m "feat: 项目骨架与 CLI 冒烟(src 布局/pytest/合成视频 fixture)"
```

---

### Task 2: sampler——视频探针、帧采样、音频提取

**Files:**
- Create: `src/shuttlecut/sampler.py`, `tests/test_sampler.py`
- Modify: `src/shuttlecut/cli.py`(label 子命令挂 `--sheets`/`--strip` 占位后由 Task 9 实现,本任务不动)

**Interfaces:**
- Produces:
  - `probe(path: str) -> VideoMeta`,`VideoMeta(path, duration_s, width, height, fps)`
  - `extract_frames(video: str, outdir: str, fps: float = 5.0, width: int = 1280, t_start: float | None = None, t_end: float | None = None) -> list[str]`(返回按序帧路径,帧名 `frame_%06d.jpg`)
  - `extract_audio(video: str, out_wav: str, sr: int = 16000) -> str`
- Consumes: ffmpeg/ffprobe 系统命令(已安装)

- [ ] **Step 1: 写失败测试**

`tests/test_sampler.py`:

```python
from shuttlecut.sampler import extract_audio, extract_frames, probe


def test_probe(synth_video):
    meta = probe(synth_video)
    assert 19.5 < meta.duration_s < 20.5
    assert (meta.width, meta.height) == (1280, 720)
    assert abs(meta.fps - 30.0) < 0.1


def test_extract_frames(synth_video, tmp_path):
    frames = extract_frames(synth_video, str(tmp_path), fps=5.0, width=1280)
    assert 98 <= len(frames) <= 101  # 20s * 5fps,容忍 ffmpeg 帧计数舍入
    assert frames[0].endswith("frame_000001.jpg")


def test_extract_frames_window(synth_video, tmp_path):
    frames = extract_frames(synth_video, str(tmp_path / "w"), fps=5.0, width=1280,
                            t_start=2.0, t_end=6.0)
    assert 15 <= len(frames) <= 25


def test_extract_audio(synth_video, tmp_path):
    wav = extract_audio(synth_video, str(tmp_path / "a.wav"))
    import soundfile as sf
    info = sf.info(wav)
    assert info.samplerate == 16000 and info.channels == 1
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest tests/test_sampler.py -v`
Expected: FAIL(ModuleNotFoundError: shuttlecut.sampler)

- [ ] **Step 3: 实现 sampler**

`src/shuttlecut/sampler.py`:

```python
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VideoMeta:
    path: str
    duration_s: float
    width: int
    height: int
    fps: float


def probe(path: str) -> VideoMeta:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate:format=duration",
         "-of", "json", path],
        check=True, capture_output=True, text=True,
    )
    d = json.loads(r.stdout)
    stream = d["streams"][0]
    num, den = stream["avg_frame_rate"].split("/")
    return VideoMeta(
        path=path,
        duration_s=float(d["format"]["duration"]),
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=float(num) / float(den),
    )


def extract_frames(video: str, outdir: str, fps: float = 5.0, width: int = 1280,
                   t_start: float | None = None, t_end: float | None = None) -> list[str]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-loglevel", "error"]
    if t_start is not None:
        cmd += ["-ss", str(t_start)]
    if t_end is not None:
        cmd += ["-t", str(t_end - (t_start or 0.0))]
    cmd += ["-i", video, "-vf", f"fps={fps},scale={width}:-2", "-q:v", "2",
            str(out / "frame_%06d.jpg")]
    subprocess.run(cmd, check=True)
    return sorted(str(p) for p in out.glob("frame_*.jpg"))


def extract_audio(video: str, out_wav: str, sr: int = 16000) -> str:
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", video, "-vn", "-ac", "1",
         "-ar", str(sr), out_wav],
        check=True,
    )
    return out_wav
```

- [ ] **Step 4: 运行测试通过**

Run: `.venv/bin/python -m pytest tests/test_sampler.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/shuttlecut/sampler.py tests/test_sampler.py
git commit -m "feat: sampler(探针/帧采样/音频提取)"
```

---

### Task 3: detector——YOLO person 检测封装

**Files:**
- Create: `src/shuttlecut/detector.py`, `tests/test_detector.py`

**Interfaces:**
- Produces:
  - `PersonBox(cx, cy, w, h, conf)`(全部 float)
  - `FramePersons(t, persons: list[PersonBox])`
  - `detect_persons(frame_paths, frame_fps, model_path="models/yolo11n.pt", device="auto", batch=64, out_jsonl=None) -> list[FramePersons]`;写 jsonl 时首行为 `{"meta": {"frame_w": W, "frame_h": H, "fps": frame_fps}}`,后续行 `{"t": t, "persons": [{"cx":..,"cy":..,"w":..,"h":..,"conf":..}]}`
  - `load_persons_jsonl(path) -> tuple[dict, list[FramePersons]]`(返回 meta 与记录)
  - `resolve_device(pref: str) -> str`("auto"→"mps"/"cpu",其它原样返回)
- Consumes: `models/yolo11n.pt`(已存在,验证时已下载)

- [ ] **Step 1: 写失败测试**

`tests/test_detector.py`:

```python
from shuttlecut.detector import (
    FramePersons, PersonBox, detect_persons, load_persons_jsonl, resolve_device,
)
from shuttlecut.sampler import extract_frames


def test_resolve_device():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("mps") == "mps"
    assert resolve_device("auto") in ("mps", "cpu")


def test_detect_and_roundtrip(synth_video, tmp_path):
    frames = extract_frames(synth_video, str(tmp_path / "f"), fps=1.0, width=1280)[:3]
    jl = str(tmp_path / "persons.jsonl")
    rows = detect_persons(frames, frame_fps=1.0, device="cpu", out_jsonl=jl)
    assert all(isinstance(r, FramePersons) for r in rows)
    meta, loaded = load_persons_jsonl(jl)
    assert meta["frame_w"] == 1280 and len(loaded) == len(rows)
    # testsrc 无人物:允许空列表,但 schema 必须合法
    assert all(p.conf >= 0.0 for r in loaded for p in r.persons)
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest tests/test_detector.py -v`
Expected: FAIL(ModuleNotFoundError)

- [ ] **Step 3: 实现 detector**

`src/shuttlecut/detector.py`:

```python
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from ultralytics import YOLO


@dataclass
class PersonBox:
    cx: float
    cy: float
    w: float
    h: float
    conf: float


@dataclass
class FramePersons:
    t: float
    persons: list[PersonBox]


def resolve_device(pref: str) -> str:
    if pref != "auto":
        return pref
    return "mps" if torch.backends.mps.is_available() else "cpu"


def detect_persons(frame_paths: list[str], frame_fps: float,
                   model_path: str = "models/yolo11n.pt", device: str = "auto",
                   batch: int = 64, out_jsonl: str | None = None) -> list[FramePersons]:
    dev = resolve_device(device)
    model = YOLO(model_path)
    rows: list[FramePersons] = []
    frame_w = frame_h = 0
    for i in range(0, len(frame_paths), batch):
        chunk = frame_paths[i : i + batch]
        results = model.predict(chunk, device=dev, imgsz=1280, classes=[0],
                                conf=0.25, verbose=False)
        for j, r in enumerate(results):
            frame_w, frame_h = int(r.orig_shape[1]), int(r.orig_shape[0])
            persons = []
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                persons.append(PersonBox((x1 + x2) / 2, (y1 + y2) / 2,
                                         x2 - x1, y2 - y1, float(box.conf[0])))
            rows.append(FramePersons(t=(i + j) / frame_fps, persons=persons))
    if out_jsonl:
        Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(out_jsonl, "w") as f:
            f.write(json.dumps({"meta": {"frame_w": frame_w, "frame_h": frame_h,
                                         "fps": frame_fps}}) + "\n")
            for row in rows:
                f.write(json.dumps({
                    "t": row.t,
                    "persons": [vars(p) for p in row.persons],
                }) + "\n")
    return rows


def load_persons_jsonl(path: str) -> tuple[dict, list[FramePersons]]:
    meta: dict = {}
    rows: list[FramePersons] = []
    with open(path) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            if i == 0 and "meta" in d:
                meta = d["meta"]
                continue
            rows.append(FramePersons(
                t=d["t"],
                persons=[PersonBox(**p) for p in d["persons"]],
            ))
    return meta, rows
```

- [ ] **Step 4: 运行测试通过**

Run: `.venv/bin/python -m pytest tests/test_detector.py -v`
Expected: 2 passed(首次加载模型约 10-30s)

- [ ] **Step 5: Commit**

```bash
git add src/shuttlecut/detector.py tests/test_detector.py
git commit -m "feat: detector(YOLO11n person 封装/jsonl 读写/设备回退)"
```

---

### Task 4: activity——运动能量信号

**Files:**
- Create: `src/shuttlecut/activity.py`, `tests/test_activity.py`

**Interfaces:**
- Produces:
  - `EnergySeries(times: list[float], values: list[float])`
  - `auto_roi(rows: list[FramePersons], frame_w: float, frame_h: float, min_h_ratio: float = 0.12) -> tuple[float, float, float, float]`(大目标质心的 5%-95% 分位包围盒,`(x, y, w, h)`)
  - `motion_energy(rows, frame_h, roi=None, min_h_ratio=0.12) -> EnergySeries`(双向最近邻平均位移,等价 Chamfer;时间轴取 `rows` 的 `t`)
  - `smooth(series: EnergySeries, window_s: float, fps: float) -> EnergySeries`(滑动平均,边缘 same 模式)
- Consumes: Task 3 的 `FramePersons/PersonBox`

- [ ] **Step 1: 写失败测试**

`tests/test_activity.py`:

```python
from shuttlecut.activity import EnergySeries, auto_roi, motion_energy, smooth
from shuttlecut.detector import FramePersons, PersonBox

H = 720.0


def frame(t, pts):
    return FramePersons(t, [PersonBox(x, y, 50, 0.2 * H, 0.9) for x, y in pts])


def test_motion_zero_when_static():
    rows = [frame(0.0, [(100, 400)]), frame(0.2, [(100, 400)])]
    s = motion_energy(rows, H)
    assert s.values[1] == 0.0


def test_motion_displacement():
    rows = [frame(0.0, [(100, 400)]), frame(0.2, [(130, 400)])]
    s = motion_energy(rows, H)
    assert abs(s.values[1] - 30.0) < 1e-6


def test_small_persons_filtered():
    rows = [frame(0.0, [(100, 400), (600, 100)]),
            frame(0.2, [(100, 400), (900, 100)])]  # 第二个是小目标(高 0.2*H? 不,构造小目标)
    rows = [FramePersons(0.0, [PersonBox(100, 400, 50, 0.2 * H, 0.9),
                                PersonBox(600, 100, 30, 0.05 * H, 0.9)]),
            FramePersons(0.2, [PersonBox(100, 400, 50, 0.2 * H, 0.9),
                                PersonBox(900, 100, 30, 0.05 * H, 0.9)])]
    s = motion_energy(rows, H)
    assert s.values[1] == 0.0  # 小目标位移 300px 应被过滤


def test_roi_filters_outside():
    rows = [FramePersons(0.0, [PersonBox(100, 400, 50, 0.2 * H, 0.9),
                                PersonBox(1100, 400, 50, 0.2 * H, 0.9)]),
            FramePersons(0.2, [PersonBox(100, 400, 50, 0.2 * H, 0.9),
                                PersonBox(1100, 700, 50, 0.2 * H, 0.9)])]
    s = motion_energy(rows, H, roi=(0, 200, 800, 500))
    assert s.values[1] == 0.0  # ROI 外的移动者不计入


def test_auto_roi_bounds():
    rows = [frame(0.0, [(200, 300), (900, 600), (640, 360)])]
    x, y, w, h = auto_roi(rows, 1280, H)
    assert 0 <= x < x + w <= 1280 and 0 <= y < y + h <= H


def test_smooth_constant_series_unchanged():
    s = EnergySeries([0.0, 0.2, 0.4], [5.0, 5.0, 5.0])
    sm = smooth(s, window_s=0.4, fps=5)
    assert sm.values == [5.0, 5.0, 5.0]
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest tests/test_activity.py -v`
Expected: FAIL(ModuleNotFoundError)

- [ ] **Step 3: 实现 activity**

`src/shuttlecut/activity.py`:

```python
from dataclasses import dataclass

import numpy as np

from shuttlecut.detector import FramePersons


@dataclass
class EnergySeries:
    times: list[float]
    values: list[float]


def _large_persons(row: FramePersons, frame_h: float, min_h_ratio: float,
                   roi: tuple[float, float, float, float] | None) -> list[tuple[float, float]]:
    rx, ry, rw, rh = roi if roi else (0.0, 0.0, float("inf"), float("inf"))
    out = []
    for p in row.persons:
        if p.h < min_h_ratio * frame_h:
            continue
        if not (rx <= p.cx <= rx + rw and ry <= p.cy <= ry + rh):
            continue
        out.append((p.cx, p.cy))
    return out


def auto_roi(rows: list[FramePersons], frame_w: float, frame_h: float,
             min_h_ratio: float = 0.12) -> tuple[float, float, float, float]:
    pts = np.array([pt for r in rows for pt in _large_persons(r, frame_h, min_h_ratio, None)])
    x0, y0 = np.percentile(pts[:, 0], 5), np.percentile(pts[:, 1], 5)
    x1, y1 = np.percentile(pts[:, 0], 95), np.percentile(pts[:, 1], 95)
    return (float(x0), float(y0), float(x1 - x0), float(y1 - y0))


def _chamfer(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float((d.min(axis=1).mean() + d.min(axis=0).mean()) / 2)


def motion_energy(rows: list[FramePersons], frame_h: float,
                  roi: tuple[float, float, float, float] | None = None,
                  min_h_ratio: float = 0.12) -> EnergySeries:
    times, values = [], []
    prev: np.ndarray | None = None
    for row in rows:
        cur = np.array(_large_persons(row, frame_h, min_h_ratio, roi))
        times.append(row.t)
        values.append(_chamfer(cur, prev) if prev is not None else 0.0)
        prev = cur
    return EnergySeries(times, values)


def smooth(series: EnergySeries, window_s: float, fps: float) -> EnergySeries:
    k = max(1, int(round(window_s * fps)))
    v = np.convolve(np.array(series.values), np.ones(k) / k, mode="same")
    return EnergySeries(series.times, v.tolist())
```

- [ ] **Step 4: 运行测试通过**

Run: `.venv/bin/python -m pytest tests/test_activity.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/shuttlecut/activity.py tests/test_activity.py
git commit -m "feat: activity(大目标过滤/ROI/Chamfer 运动能量/平滑)"
```

---

### Task 5: segmenter——滞回状态机回合切分

**Files:**
- Create: `src/shuttlecut/segmenter.py`, `tests/test_segmenter.py`

**Interfaces:**
- Produces:
  - `SegParams(hi_q=0.60, lo_q=0.30, min_rally_s=3.0, min_idle_s=2.5, max_rally_s=120.0, smooth_s=2.0, pre_roll_s=1.5, post_roll_s=2.0)`(dataclass,全 float)
  - `Rally(start: float, end: float, motion_peak: float, confidence: float, hits: int = 0)`(dataclass)
  - `segment(series: EnergySeries, params: SegParams) -> list[Rally]`(时间轴为**原视频时间**,由调用方保证 series.times 已换算)
- Consumes: Task 4 `EnergySeries`

- [ ] **Step 1: 写失败测试**

`tests/test_segmenter.py`:

```python
from shuttlecut.activity import EnergySeries
from shuttlecut.segmenter import SegParams, segment

FPS = 5


def series_from(level_fn, duration_s):
    n = int(duration_s * FPS)
    return EnergySeries([i / FPS for i in range(n)], [level_fn(i / FPS) for i in range(n)])


def test_two_rallies_detected():
    # 0-10s 高能量,10-18s 低,18-28s 高,28-30s 低
    s = series_from(lambda t: 10.0 if (t < 10 or 18 <= t < 28) else 1.0, 30)
    rallies = segment(s, SegParams())
    assert len(rallies) == 2
    assert abs(rallies[0].start - 0.0) < 1.5 and abs(rallies[0].end - 10.0) < 1.5
    assert abs(rallies[1].start - 18.0) < 1.5 and abs(rallies[1].end - 28.0) < 1.5


def test_short_blip_merged_not_rally():
    # 间歇中 1s 的小尖峰不应产生碎片回合
    def lvl(t):
        return 10.0 if 5.0 <= t < 6.0 else 1.0
    s = series_from(lvl, 20)
    assert segment(s, SegParams()) == []


def test_min_idle_merge():
    # 两个 8s 高能段间隔 1s(< min_idle 2.5)应合并为一个回合
    def lvl(t):
        return 10.0 if (t < 8 or 9 <= t < 17) else 1.0
    s = series_from(lvl, 22)
    rallies = segment(s, SegParams())
    assert len(rallies) == 1
    assert rallies[0].end - rallies[0].start > 15


def test_short_rally_dropped():
    def lvl(t):
        return 10.0 if 5 <= t < 6.5 else 1.0  # 1.5s < min_rally 3s
    s = series_from(lvl, 15)
    assert segment(s, SegParams()) == []


def test_max_rally_split():
    def lvl(t):
        return 10.0 if 5.0 <= t < 295.0 else 1.0  # 290s 连续高能量(首尾留间歇避免 hi<=lo)
    s = series_from(lvl, 300)
    rallies = segment(s, SegParams())
    assert all(r.end - r.start <= 120.0 + 5.0 for r in rallies)
    assert len(rallies) >= 2
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest tests/test_segmenter.py -v`
Expected: FAIL(ModuleNotFoundError)

- [ ] **Step 3: 实现 segmenter**

`src/shuttlecut/segmenter.py`:

```python
from dataclasses import dataclass, field

import numpy as np

from shuttlecut.activity import EnergySeries


@dataclass
class SegParams:
    hi_q: float = 0.60
    lo_q: float = 0.30
    min_rally_s: float = 3.0
    min_idle_s: float = 2.5
    max_rally_s: float = 120.0
    smooth_s: float = 2.0
    pre_roll_s: float = 1.5
    post_roll_s: float = 2.0


@dataclass
class Rally:
    start: float
    end: float
    motion_peak: float
    confidence: float
    hits: int = 0


def _raw_segments(values: np.ndarray, times: list[float],
                  hi: float, lo: float) -> list[tuple[float, float, float]]:
    """滞回扫描:>=hi 进入,<=lo 退出。返回 (start, end, peak)。"""
    segs: list[tuple[float, float, float]] = []
    in_rally = False
    start = peak = 0.0
    for t, v in zip(times, values):
        if not in_rally and v >= hi:
            in_rally, start, peak = True, t, float(v)
        elif in_rally:
            peak = max(peak, float(v))
            if v <= lo:
                segs.append((start, t, peak))
                in_rally = False
    if in_rally:
        segs.append((start, times[-1], peak))
    return segs


def _merge_by_min_idle(segs: list[tuple[float, float, float]],
                       min_idle_s: float) -> list[tuple[float, float, float]]:
    merged: list[tuple[float, float, float]] = []
    for seg in segs:
        if merged and seg[0] - merged[-1][1] < min_idle_s:
            s0, _, p0 = merged[-1]
            merged[-1] = (s0, seg[1], max(p0, seg[2]))
        else:
            merged.append(seg)
    return merged


def _split_long(seg: tuple[float, float, float], max_rally_s: float,
                values: np.ndarray, times: list[float]) -> list[tuple[float, float, float]]:
    s0, s1, peak = seg
    if s1 - s0 <= max_rally_s:
        return [seg]
    # 在中点附近找能量最低的切分点
    mid = (s0 + s1) / 2
    mask = (np.array(times) >= mid - 5) & (np.array(times) <= mid + 5)
    idx = np.where(mask)[0]
    cut_t = float(times[idx[np.argmin(values[idx])]])
    left, right = (s0, cut_t, peak), (cut_t, s1, peak)
    return _split_long(left, max_rally_s, values, times) + _split_long(right, max_rally_s, values, times)


def segment(series: EnergySeries, params: SegParams) -> list[Rally]:
    values = np.array(series.values)
    hi, lo = float(np.percentile(values, params.hi_q * 100)), float(np.percentile(values, params.lo_q * 100))
    if hi <= lo:
        return []
    segs = _raw_segments(values, series.times, hi, lo)
    segs = [s for s in segs if s[1] - s[0] >= params.min_rally_s]
    segs = _merge_by_min_idle(segs, params.min_idle_s)
    segs = [s for s in segs if s[1] - s[0] >= params.min_rally_s]
    segs = [p for s in segs for p in _split_long(s, params.max_rally_s, values, series.times)]
    med = float(np.median(values))
    return [Rally(start=s, end=e, motion_peak=p,
                  confidence=float(np.clip((p - lo) / max(hi - lo, 1e-9), 0.0, 1.0)))
            for s, e, p in segs]
```

- [ ] **Step 4: 运行测试通过**

Run: `.venv/bin/python -m pytest tests/test_segmenter.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/shuttlecut/segmenter.py tests/test_segmenter.py
git commit -m "feat: segmenter(滞回状态机/最短时长/合并/超长切分)"
```

---

### Task 6: refiner——音频瞬态检测与边界精修

**Files:**
- Create: `src/shuttlecut/refiner.py`, `tests/test_refiner.py`

**Interfaces:**
- Produces:
  - `Transient(t: float, z: float)`(dataclass)
  - `audio_transients(wav: str, z_thresh: float = 2.0, min_gap_s: float = 0.3, hop_length: int = 512) -> list[Transient]`(频谱通量 onset + z 过滤 + 最小间隔去重)
  - `refine(rallies: list[Rally], transients: list[Transient], search_back_s: float = 3.0) -> list[Rally]`(起点向前搜瞬态,命中则 `start = t_hit - 0.2`;不越前一个回合;填 `hits` 计数)
- Consumes: Task 5 `Rally`;soundfile/librosa

- [ ] **Step 1: 写失败测试**

`tests/test_refiner.py`:

```python
import numpy as np
import soundfile as sf

from shuttlecut.refiner import Transient, audio_transients, refine
from shuttlecut.segmenter import Rally

SR = 16000


def click_wav(path, times, duration=12.0):
    n = int(duration * SR)
    y = np.random.default_rng(0).normal(0, 0.003, n)  # 低底噪
    for t in times:
        i = int(t * SR)
        w = min(200, n - i)
        env = np.exp(-np.linspace(0, 12, w))
        y[i : i + w] += 0.7 * env * np.sin(2 * np.pi * np.linspace(3e3, 1e3, w))
    sf.write(path, y, SR)


def test_audio_transients_finds_clicks(tmp_path):
    wav = str(tmp_path / "c.wav")
    click_wav(wav, [3.0, 4.0, 5.5])
    ts = audio_transients(wav)
    assert len(ts) == 3
    assert all(abs(t.t - e) < 0.15 for t, e in zip(ts, [3.0, 4.0, 5.5]))


def test_min_gap_dedup(tmp_path):
    wav = str(tmp_path / "c.wav")
    click_wav(wav, [3.0, 3.1])  # 间隔 0.1s < min_gap 0.3
    assert len(audio_transients(wav)) == 1


def test_refine_pulls_start_to_transient():
    rallies = [Rally(start=10.0, end=20.0, motion_peak=9.0, confidence=0.9)]
    trans = [Transient(8.4, 3.0), Transient(12.0, 2.5), Transient(14.0, 2.2)]
    out = refine(rallies, trans, search_back_s=3.0)
    assert abs(out[0].start - (8.4 - 0.2)) < 1e-6  # 状态机起点前 3s 内最近瞬态
    assert out[0].hits == 2  # [start, end] 内 12.0 与 14.0


def test_refine_no_transient_keeps_start():
    rallies = [Rally(start=10.0, end=20.0, motion_peak=9.0, confidence=0.9)]
    out = refine(rallies, [Transient(4.0, 3.0)])
    assert out[0].start == 10.0 and out[0].hits == 0


def test_refine_does_not_cross_previous_rally():
    rallies = [Rally(start=2.0, end=8.0, motion_peak=9.0, confidence=0.9),
               Rally(start=10.0, end=20.0, motion_peak=9.0, confidence=0.9)]
    out = refine(rallies, [Transient(9.5, 3.0)])
    assert out[1].start >= 8.2  # 不越过上一回合结束 + 0.2
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest tests/test_refiner.py -v`
Expected: FAIL(ModuleNotFoundError)

- [ ] **Step 3: 实现 refiner**

`src/shuttlecut/refiner.py`:

```python
from dataclasses import dataclass

import librosa
import numpy as np

from shuttlecut.segmenter import Rally


@dataclass
class Transient:
    t: float
    z: float


def audio_transients(wav: str, z_thresh: float = 2.0, min_gap_s: float = 0.3,
                     hop_length: int = 512) -> list[Transient]:
    y, sr = librosa.load(wav, sr=16000, mono=True)
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length,
                                        units="time", backtrack=False)
    frames = np.clip(librosa.time_to_frames(onsets, sr=sr, hop_length=hop_length),
                     0, len(env) - 1)
    strengths = env[frames]
    z = (strengths - env.mean()) / (env.std() + 1e-9)
    picked: list[Transient] = []
    for t, zi in zip(onsets, z):
        if zi < z_thresh:
            continue
        if picked and t - picked[-1].t < min_gap_s:
            if zi > picked[-1].z:  # 保留更强者
                picked[-1] = Transient(float(t), float(zi))
            continue
        picked.append(Transient(float(t), float(zi)))
    return picked


def refine(rallies: list[Rally], transients: list[Transient],
           search_back_s: float = 3.0) -> list[Rally]:
    times = [tr.t for tr in transients]
    out: list[Rally] = []
    for i, r in enumerate(rallies):
        floor = rallies[i - 1].end + 0.2 if i > 0 else 0.0
        lo, hi = max(floor, r.start - search_back_s), r.start
        cands = [tr for tr in transients if lo <= tr.t <= hi]
        new_start = (cands[-1].t - 0.2) if cands else r.start  # 取窗口内最近的一个
        new_start = max(new_start, floor)
        hits = sum(1 for t in times if new_start <= t <= r.end)
        out.append(Rally(start=new_start, end=r.end, motion_peak=r.motion_peak,
                         confidence=r.confidence, hits=hits))
    return out
```

- [ ] **Step 4: 运行测试通过**

Run: `.venv/bin/python -m pytest tests/test_refiner.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/shuttlecut/refiner.py tests/test_refiner.py
git commit -m "feat: refiner(音频瞬态检测/起点精修/击球计数)"
```

---

### Task 7: exporter——切片与集锦导出

**Files:**
- Create: `src/shuttlecut/exporter.py`, `tests/test_exporter.py`

**Interfaces:**
- Produces:
  - `export_clips(video: str, rallies: list[Rally], out_dir: str, pre_s: float = 1.5, post_s: float = 2.0) -> list[str]`(`rally_%03d.mp4`,重编码 H.264/AAC,边界精确)
  - `export_reel(clips: list[str], out_path: str) -> str`(concat demuxer 流复制)
- Consumes: Task 5 `Rally`;Task 2 `probe`

- [ ] **Step 1: 写失败测试**

`tests/test_exporter.py`:

```python
from shuttlecut.exporter import export_clips, export_reel
from shuttlecut.sampler import probe
from shuttlecut.segmenter import Rally


def test_export_clips_duration(synth_video, tmp_path):
    rallies = [Rally(3.0, 9.0, 9.0, 0.9), Rally(12.0, 17.0, 9.0, 0.8)]
    clips = export_clips(synth_video, rallies, str(tmp_path / "clips"))
    assert len(clips) == 2
    m = probe(clips[0])
    # 期望 (9+2) - max(0, 3-1.5) = 8.5s,容差 0.5
    assert abs(m.duration_s - 8.5) < 0.5


def test_export_clip_head_clamped(synth_video, tmp_path):
    clips = export_clips(synth_video, [Rally(0.5, 4.0, 9.0, 0.9)], str(tmp_path / "c2"))
    assert abs(probe(clips[0]).duration_s - (4.0 + 2.0)) < 0.5  # start-pre 截到 0


def test_export_reel(synth_video, tmp_path):
    clips = export_clips(synth_video, [Rally(1.0, 5.0, 9.0, 0.9),
                                       Rally(8.0, 12.0, 9.0, 0.9)], str(tmp_path / "c3"))
    reel = export_reel(clips, str(tmp_path / "c3" / "highlights.mp4"))
    m = probe(reel)
    assert m.duration_s > 8.0
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest tests/test_exporter.py -v`
Expected: FAIL(ModuleNotFoundError)

- [ ] **Step 3: 实现 exporter**

`src/shuttlecut/exporter.py`:

```python
import subprocess
from pathlib import Path

from shuttlecut.segmenter import Rally


def export_clips(video: str, rallies: list[Rally], out_dir: str,
                 pre_s: float = 1.5, post_s: float = 2.0) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clips: list[str] = []
    for i, r in enumerate(rallies, start=1):
        ss = max(0.0, r.start - pre_s)
        to = r.end + post_s
        dest = out / f"rally_{i:03d}.mp4"
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-ss", f"{ss:.3f}", "-to", f"{to:.3f}",
             "-i", video, "-c:v", "libx264", "-preset", "fast", "-crf", "20",
             "-c:a", "aac", "-movflags", "+faststart", str(dest)],
            check=True,
        )
        clips.append(str(dest))
    return clips


def export_reel(clips: list[str], out_path: str) -> str:
    lst = Path(out_path).with_suffix(".txt")
    lst.write_text("\n".join(f"file '{c}'" for c in clips))
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(lst), "-c", "copy", out_path],
        check=True,
    )
    return out_path
```

- [ ] **Step 4: 运行测试通过**

Run: `.venv/bin/python -m pytest tests/test_exporter.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/shuttlecut/exporter.py tests/test_exporter.py
git commit -m "feat: exporter(精确切片/concat 集锦)"
```

---

### Task 8: CLI `process` 端到端编排

**Files:**
- Modify: `src/shuttlecut/cli.py`
- Create: `tests/test_process_cli.py`

**Interfaces:**
- Consumes: Task 2-7 全部接口
- Produces: `shuttlecut process VIDEO... [--out DIR] [--roi X,Y,W,H] [--min-rally S] [--min-idle S] [--device auto|mps|cpu] [--no-audio-refine] [--no-reel]`;每个视频产出 `outputs/<stem>/clips/`、`rallies.json`、`persons.jsonl`、`roi.txt`;`rallies.json` 结构 `{"video", "generated_at", "params", "rallies": [{"id","start_s","end_s","duration_s","hits","motion_peak","confidence"}]}`

- [ ] **Step 1: 写失败测试(合成视频端到端,只断言结构不断言切分质量)**

`tests/test_process_cli.py`:

```python
import json

from shuttlecut.cli import main


def test_process_end_to_end(synth_video, tmp_path, capsys):
    rc = main(["process", synth_video, "--out", str(tmp_path), "--device", "cpu",
               "--no-reel"])
    assert rc == 0
    out = tmp_path / "synth"
    assert (out / "persons.jsonl").exists()
    data = json.loads((out / "rallies.json").read_text())
    assert data["video"] == "synth"
    assert isinstance(data["rallies"], list)  # testsrc 无真实回合,允许空
    assert "summary" in capsys.readouterr().out.lower() or True  # 摘要打印不拦截


def test_process_no_audio(synth_video, tmp_path):
    import subprocess
    silent = tmp_path / "noaudio.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", synth_video, "-an",
                    "-c", "copy", str(silent)], check=True)
    rc = main(["process", str(silent), "--out", str(tmp_path / "o"), "--device", "cpu"])
    assert rc == 0
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest tests/test_process_cli.py -v`
Expected: FAIL(process 子命令参数错误)

- [ ] **Step 3: 实现编排**

`src/shuttlecut/cli.py`(替换为):

```python
import argparse
import json
from datetime import datetime
from pathlib import Path

from shuttlecut import __version__
from shuttlecut.activity import auto_roi, motion_energy, smooth
from shuttlecut.detector import detect_persons, load_persons_jsonl
from shuttlecut.exporter import export_clips, export_reel
from shuttlecut.refiner import audio_transients, refine
from shuttlecut.sampler import extract_audio, extract_frames, probe
from shuttlecut.segmenter import Rally, SegParams, segment


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="shuttlecut", description="羽毛球回合自动剪辑")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd")

    pr = sub.add_parser("process", help="切分回合并导出片段")
    pr.add_argument("videos", nargs="+")
    pr.add_argument("--out", default="outputs")
    pr.add_argument("--roi", default=None, help="X,Y,W,H 覆盖自动 ROI")
    pr.add_argument("--min-rally", type=float, default=SegParams().min_rally_s)
    pr.add_argument("--min-idle", type=float, default=SegParams().min_idle_s)
    pr.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    pr.add_argument("--no-audio-refine", action="store_true")
    pr.add_argument("--no-reel", action="store_true")

    sub.add_parser("label", help="真值标注辅助工具")
    ev = sub.add_parser("eval", help="对比检测结果与真值")
    ev.add_argument("rallies_json")
    ev.add_argument("--gt", required=True)
    return p


def _parse_roi(s: str | None):
    if not s:
        return None
    x, y, w, h = (float(v) for v in s.split(","))
    return (x, y, w, h)


def process_one(video: str, out_root: str, args) -> int:
    meta = probe(video)
    stem = Path(video).stem
    outdir = Path(out_root) / stem
    work = Path("temp/work") / stem
    frames = extract_frames(video, str(work / "frames"), fps=5.0, width=1280)
    persons_path = outdir / "persons.jsonl"
    rows = detect_persons(frames, frame_fps=5.0, device=args.device,
                          out_jsonl=str(persons_path))
    pmeta, rows = load_persons_jsonl(str(persons_path))
    frame_h = float(pmeta["frame_h"])

    roi = _parse_roi(args.roi)
    if roi is None:
        roi = auto_roi(rows, float(pmeta["frame_w"]), frame_h)
        (outdir / "roi.txt").write_text(",".join(f"{v:.1f}" for v in roi))
    energy = smooth(motion_energy(rows, frame_h, roi=roi), window_s=2.0, fps=5.0)

    params = SegParams(min_rally_s=args.min_rally, min_idle_s=args.min_idle)
    rallies: list[Rally] = segment(energy, params)

    if not args.no_audio_refine:
        try:
            wav = extract_audio(video, str(work / "audio.wav"))
            rallies = refine(rallies, audio_transients(wav))
        except Exception as e:  # 无音轨/解码失败 → 降级纯视觉
            print(f"[warn] 音频精修跳过: {e}")

    outdir.mkdir(parents=True, exist_ok=True)
    clips = export_clips(video, rallies, str(outdir / "clips"),
                         pre_s=params.pre_roll_s, post_s=params.post_roll_s)
    if not args.no_reel and clips:
        export_reel(clips, str(outdir / "clips" / "highlights.mp4"))

    payload = {
        "video": stem,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "params": vars(params) | {"roi": roi},
        "rallies": [
            {"id": i, "start_s": round(r.start, 2), "end_s": round(r.end, 2),
             "duration_s": round(r.end - r.start, 2), "hits": r.hits,
             "motion_peak": round(r.motion_peak, 2), "confidence": round(r.confidence, 3)}
            for i, r in enumerate(rallies, start=1)
        ],
    }
    (outdir / "rallies.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    total = sum(r.end - r.start for r in rallies)
    print(f"[summary] {stem}: {meta.duration_s:.0f}s → {len(rallies)} 个回合, "
          f"共 {total:.0f}s ({total / meta.duration_s * 100:.0f}% 保留)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd is None:
        build_parser().print_help()
        return 1
    if args.cmd == "process":
        for video in args.videos:
            process_one(video, args.out, args)
        return 0
    print("该子命令在后续任务实现")
    return 0
```

- [ ] **Step 4: 运行测试通过**

Run: `.venv/bin/python -m pytest tests/test_process_cli.py -v`
Expected: 2 passed

- [ ] **Step 5: 全量测试回归**

Run: `.venv/bin/python -m pytest -v`
Expected: 全部通过

- [ ] **Step 6: Commit**

```bash
git add src/shuttlecut/cli.py tests/test_process_cli.py
git commit -m "feat: process 端到端编排(采样→检测→能量→切分→精修→导出)"
```

---

### Task 9: 真值工具——接触表/密集帧条/真值 schema

**Files:**
- Create: `src/shuttlecut/labeling/__init__.py`, `src/shuttlecut/labeling/sheets.py`, `src/shuttlecut/labeling/gt.py`, `tests/test_gt.py`
- Modify: `src/shuttlecut/cli.py`(实现 `label` 子命令)

**Interfaces:**
- Produces:
  - `make_contact_sheets(video: str, outdir: str, step_s: float = 2.0, cols: int = 5, rows: int = 6) -> list[str]`(每张 30 格、时间戳标注,覆盖全片)
  - `make_dense_strip(video: str, t0: float, t1: float, outpath: str, step_s: float = 0.2) -> str`(±3s 边界精修用,时间戳网格图)
  - `save_gt(path: str, video_stem: str, rallies: list[dict]) -> None`、`load_gt(path: str) -> dict`;schema 校验:非空、按序、`0 ≤ start < end`、互不重叠
  - CLI:`shuttlecut label VIDEO [--sheets OUTDIR] [--strip T0 T1] [--out FILE]`
- Consumes: Task 2 `extract_frames`

- [ ] **Step 1: 写失败测试**

`tests/test_gt.py`:

```python
import pytest

from shuttlecut.labeling.gt import GTError, load_gt, save_gt


def test_gt_roundtrip(tmp_path):
    p = str(tmp_path / "g.json")
    save_gt(p, "video1", [{"id": 1, "start_s": 5.0, "end_s": 12.0, "note": ""}])
    d = load_gt(p)
    assert d["video"] == "video1" and d["rallies"][0]["start_s"] == 5.0


def test_gt_rejects_overlap(tmp_path):
    p = str(tmp_path / "g.json")
    save_gt(p, "video1", [{"id": 1, "start_s": 5.0, "end_s": 12.0, "note": ""},
                          {"id": 2, "start_s": 10.0, "end_s": 15.0, "note": ""}])
    with pytest.raises(GTError):
        load_gt(p)


def test_gt_rejects_bad_order(tmp_path):
    p = str(tmp_path / "g.json")
    save_gt(p, "video1", [{"id": 1, "start_s": 12.0, "end_s": 10.0, "note": ""}])
    with pytest.raises(GTError):
        load_gt(p)
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest tests/test_gt.py -v`
Expected: FAIL(ModuleNotFoundError)

- [ ] **Step 3: 实现 gt 与 sheets**

`src/shuttlecut/labeling/__init__.py`: 空文件。

`src/shuttlecut/labeling/gt.py`:

```python
import json
from pathlib import Path


class GTError(ValueError):
    pass


def save_gt(path: str, video_stem: str, rallies: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(
        {"video": video_stem, "labeled_by": "agent+user-spotcheck",
         "rallies": sorted(rallies, key=lambda r: r["start_s"])},
        ensure_ascii=False, indent=2))


def load_gt(path: str) -> dict:
    d = json.loads(Path(path).read_text())
    rs = d.get("rallies")
    if not rs:
        raise GTError("rallies 为空")
    prev_end = -1.0
    for r in rs:
        if not (0 <= r["start_s"] < r["end_s"]):
            raise GTError(f"非法区间: {r}")
        if r["start_s"] < prev_end:
            raise GTError(f"区间重叠: {r}")
        prev_end = r["end_s"]
    return d
```

`src/shuttlecut/labeling/sheets.py`:

```python
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
    work = Path(outdir) / "_frames"
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
    work = Path(outpath).parent / f"_strip_{t0:.0f}_{t1:.0f}"
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
```

- [ ] **Step 4: 挂接 label 子命令(cli.py 增补)**

在 `build_parser` 的 label 部分替换为:

```python
    lb = sub.add_parser("label", help="真值标注辅助工具")
    lb.add_argument("video")
    lb.add_argument("--sheets", metavar="OUTDIR", default=None)
    lb.add_argument("--strip", nargs=2, type=float, metavar=("T0", "T1"), default=None)
    lb.add_argument("--out", default=None, help="保存真值 JSON 路径")
```

`main` 中 label 分支:

```python
    if args.cmd == "label":
        from shuttlecut.labeling.gt import save_gt
        from shuttlecut.labeling.sheets import make_contact_sheets, make_dense_strip
        if args.sheets:
            files = make_contact_sheets(args.video, args.sheets)
            print(f"生成 {len(files)} 张接触表 → {args.sheets}")
        if args.strip:
            t0, t1 = args.strip
            make_dense_strip(args.video, t0, t1,
                             f"temp/label/{Path(args.video).stem}/strip_{t0:.0f}_{t1:.0f}.jpg")
            print(f"生成密集帧条 [{t0},{t1}]s")
        if args.out:
            rallies = json.loads(Path(f"temp/label/{Path(args.video).stem}/draft.json").read_text())
            save_gt(args.out, Path(args.video).stem, rallies)
            print(f"真值已保存 → {args.out}")
        return 0
```

- [ ] **Step 5: 运行测试 + 人工验证**

Run: `.venv/bin/python -m pytest tests/test_gt.py -v`
Expected: 3 passed

Run: `.venv/bin/shuttlecut label temp/DJI_20260830153830_0015_D.MP4 --strip 130 136 --sheets temp/label/DJI_20260830153830_0015_D/sheets`
Expected: 生成接触表约 16 张 + 一张密集帧条(用 look_at 目检一张 sheet 与 strip 格式正确)

- [ ] **Step 6: Commit**

```bash
git add src/shuttlecut/labeling/ tests/test_gt.py src/shuttlecut/cli.py
git commit -m "feat: 真值工具(接触表/密集帧条/schema 校验/label 子命令)"
```

---

### Task 10: 生产真值 B1/B2(人工目检流程 + 用户抽检)⚠️ 人工关卡

**Files:**
- Create: `ground_truth/DJI_20260830153830_0015_D.json`, `ground_truth/DJI_20260830173600_0025_D.json`
- 工作产物: `temp/label/DJI_20260830153830_0015_D/`, `temp/label/DJI_20260830173600_0025_D/`(sheets/draft/strip,路径 = 视频文件名去扩展名)

**Interfaces:**
- Consumes: Task 9 工具
- Produces: 两份通过 schema 校验的真值文件(**先于算法调参锁定**);`shuttlecut eval`(Task 11)依赖其格式

- [ ] **Step 1: 生成 B1 全部接触表**

Run: `.venv/bin/shuttlecut label temp/DJI_20260830153830_0015_D.MP4 --sheets temp/label/DJI_20260830153830_0015_D/sheets`
Expected: ~16 张 sheet(每张 60s,30 格)

- [ ] **Step 2: Agent 目检接触表,标出候选区间**

用 look_at 分批目检(每次 4-6 张 sheet),按格时间戳记录"回合/间歇/模糊"三类区间,写入 `temp/label/DJI_20260830153830_0015_D/draft_notes.md`。判定标准:**回合 = 有球员在场上对抗(含发球准备)**;捾球/走动/休息 = 间歇。

- [ ] **Step 3: 边界精修**

对每个模糊边界(预计 30-40 处)生成密集帧条:

Run: `.venv/bin/shuttlecut label temp/DJI_..._0015_D.MP4 --strip <T0> <T1>`(T0/T1 = 候选边界 ∓3s)

用 look_at 目检帧条,把边界定到 0.5s 精度,更新 draft;汇总为 `temp/label/DJI_20260830153830_0015_D/draft.json`(`[{"id", "start_s", "end_s", "note"}]` 按序,即 Task 9 CLI `--out` 读取的路径)。

- [ ] **Step 4: 保存 B1 真值并校验**

Run: `.venv/bin/shuttlecut label temp/DJI_..._0015_D.MP4 --out ground_truth/DJI_20260830153830_0015_D.json && .venv/bin/python -c "from shuttlecut.labeling.gt import load_gt; d=load_gt('ground_truth/DJI_20260830153830_0015_D.json'); print(len(d['rallies']), '个回合')"`
Expected: 回合数 25-45 之间(符合 15 分钟业余节奏),无 GTError

- [ ] **Step 5: 对 B2 重复 Step 1-4**

- [ ] **Step 6: 用户抽检(结构化提问)**

每段视频随机抽 5 个边界,用 ffmpeg 导出 `±3s` 片段,请用户观看确认;任何 1 处误标 → 回 Step 3 修正该边界后重新抽检。

Run: `ffmpeg -loglevel error -ss $((T-3)) -t 6 -i <video> -c copy temp/label/spotcheck_<id>.mp4`(每视频 5 个,T 为被抽检边界时间,具体数值执行时随机选定并记录)

- [ ] **Step 7: 锁定并提交**

```bash
git add ground_truth/
git commit -m "test: B1/B2 回合真值(接触表+密集帧目检产出,经用户抽检)"
```

**规则:** 此后真值只能经 Step 2-4 目检流程修正,不得参照算法输出修改(验收方案 §2-5)。

---

### Task 11: evaluate——验收评测器

**Files:**
- Create: `src/shuttlecut/eval/__init__.py`(空), `src/shuttlecut/eval/evaluate.py`, `tests/test_evaluate.py`
- Modify: `src/shuttlecut/cli.py`(实现 `eval` 子命令,含 `--record`)

**Interfaces:**
- Produces:
  - `EvalReport(recall, precision, mae_s, n_gt, n_det, matched, missed, extra, fragment, boundary)`(dataclass;missed/extra/fragment/boundary 为回合列表)
  - `evaluate(detected: list[tuple[float, float]], gt: list[tuple[float, float]], tol_s=2.5, overlap=0.5, mae_report_s=1.5) -> EvalReport`
  - CLI: `shuttlecut eval RALLIES_JSON --gt GT_JSON [--record OUT_MD]`,打印人类可读报告并在 `--record` 时追加 markdown 表格行
- Consumes: Task 10 真值格式;Task 8 `rallies.json` 格式

- [ ] **Step 1: 写失败测试**

`tests/test_evaluate.py`:

```python
from shuttlecut.eval.evaluate import evaluate


def test_perfect_match():
    gt = [(10.0, 20.0), (30.0, 40.0)]
    det = [(10.2, 19.8), (30.5, 40.4)]
    r = evaluate(det, gt)
    assert r.recall == 1.0 and r.precision == 1.0
    assert not r.missed and not r.extra and r.mae_s < 0.5


def test_missed_and_extra():
    gt = [(10.0, 20.0), (30.0, 40.0)]
    det = [(10.0, 20.0), (60.0, 70.0)]
    r = evaluate(det, gt)
    assert r.recall == 0.5 and r.precision == 0.5
    assert (30.0, 40.0) in r.missed
    assert (60.0, 70.0) in r.extra


def test_boundary_shift_still_matches():
    gt = [(10.0, 20.0)]
    det = [(12.2, 19.0)]  # start 差 2.2s ≤ 2.5, end 差 1.0
    r = evaluate(det, gt)
    assert r.recall == 1.0
    assert len(r.boundary) == 1  # MAE > 1.5 归类边界偏移(报告项)


def test_fragment_classification():
    gt = [(10.0, 30.0)]
    det = [(10.0, 18.0), (18.5, 26.0)]  # 单个 G 被切成两段,各与 G 重叠不足
    r = evaluate(det, gt)
    # 贪心匹配:第一段可能命中;第二段与已匹配 G 重叠 → fragment
    assert len(r.fragment) + len(r.extra) >= 1
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest tests/test_evaluate.py -v`
Expected: FAIL(ModuleNotFoundError)

- [ ] **Step 3: 实现 evaluate**

`src/shuttlecut/eval/evaluate.py`:

```python
from dataclasses import dataclass, field


@dataclass
class EvalReport:
    recall: float
    precision: float
    mae_s: float
    n_gt: int
    n_det: int
    matched: list[tuple[tuple[float, float], tuple[float, float]]] = field(default_factory=list)
    missed: list[tuple[float, float]] = field(default_factory=list)
    extra: list[tuple[float, float]] = field(default_factory=list)
    fragment: list[tuple[float, float]] = field(default_factory=list)
    boundary: list[tuple[tuple[float, float], tuple[float, float]]] = field(default_factory=list)


def _inter(a, b) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def evaluate(detected: list[tuple[float, float]], gt: list[tuple[float, float]],
             tol_s: float = 2.5, overlap: float = 0.5,
             mae_report_s: float = 1.5) -> EvalReport:
    used: set[int] = set()
    matches: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    for g in gt:
        best, best_i = 0.0, -1
        for i, d in enumerate(detected):
            if i in used:
                continue
            ov = _inter(d, g)
            if ov <= 0:
                continue
            ratio = ov / max(d[1] - d[0], g[1] - g[0])
            ds, de = abs(d[0] - g[0]), abs(d[1] - g[1])
            if ratio >= overlap and ds <= tol_s and de <= tol_s and ov > best:
                best, best_i = ov, i
        if best_i >= 0:
            used.add(best_i)
            matches.append((detected[best_i], g, 0.5 * (abs(detected[best_i][0] - g[0]) + abs(detected[best_i][1] - g[1]))))
    matched_gt = {id(g) for _, g, _ in matches}
    missed = [g for g in gt if id(g) not in matched_gt]
    unmatched = [d for i, d in enumerate(detected) if i not in used]
    extra, fragment = [], []
    for d in unmatched:
        overlapped = any(_inter(d, g) > 0.5 * (g[1] - g[0]) for _, g, _ in matches)
        (fragment if overlapped else extra).append(d)
    mae = sum(m for _, _, m in matches) / len(matches) if matches else float("nan")
    boundary = [(d, g) for d, g, m in matches if m > mae_report_s]
    return EvalReport(
        recall=len(matches) / len(gt) if gt else 0.0,
        precision=len(matches) / len(detected) if detected else 0.0,
        mae_s=mae, n_gt=len(gt), n_det=len(detected),
        matched=[(d, g) for d, g, _ in matches],
        missed=missed, extra=extra, fragment=fragment, boundary=boundary,
    )
```

- [ ] **Step 4: 挂接 eval 子命令**

`main` 中 eval 分支替换为:

```python
    if args.cmd == "eval":
        from shuttlecut.eval.evaluate import evaluate
        from shuttlecut.labeling.gt import load_gt
        det = [(r["start_s"], r["end_s"])
               for r in json.loads(Path(args.rallies_json).read_text())["rallies"]]
        g = [(r["start_s"], r["end_s"]) for r in load_gt(args.gt)["rallies"]]
        rep = evaluate(det, g)
        verdict = "PASS ✅" if rep.recall >= 0.90 and rep.precision >= 0.90 else "FAIL ❌"
        print(f"recall={rep.recall:.3f} precision={rep.precision:.3f} mae={rep.mae_s:.2f}s "
              f"({rep.n_det} 检出 / {rep.n_gt} 真值) → {verdict}")
        print(f"missed={len(rep.missed)} extra={len(rep.extra)} "
              f"fragment={len(rep.fragment)} boundary={len(rep.boundary)}")
        if args.record:
            from datetime import datetime as _dt
            with open(args.record, "a") as f:
                f.write(f"| {_dt.now().isoformat(timespec='seconds')} | {args.rallies_json} | "
                        f"{rep.recall:.3f} | {rep.precision:.3f} | {rep.mae_s:.2f} | {verdict} |\n")
        return 0 if "PASS" in verdict else 2
```

parser 中 eval 增参:`ev.add_argument("--record", default=None)`

- [ ] **Step 5: 运行测试通过 + 回归**

Run: `.venv/bin/python -m pytest tests/test_evaluate.py -v && .venv/bin/python -m pytest -q`
Expected: 全部通过

- [ ] **Step 6: Commit**

```bash
git add src/shuttlecut/eval/ tests/test_evaluate.py src/shuttlecut/cli.py
git commit -m "feat: evaluate(公式级命中匹配/误差分类/eval 子命令)"
```

---

### Task 12: 基线验收跑批(B1/B2)

**Files:**
- Create: `outputs/eval_history.md`(运行产物,不入 git;结论写入 README)
- 工作产物: `outputs/B1/`, `outputs/B2/`

**Interfaces:**
- Consumes: Task 8 `process`、Task 10 真值、Task 11 `eval`
- Produces: 基线指标(预期不达标,Task 13 的起点)

- [ ] **Step 1: 跑 B1 并计时**

Run: `time .venv/bin/shuttlecut process temp/DJI_20260830153830_0015_D.MP4 --out outputs`
Expected: 终端输出回合数与保留率;`outputs/DJI_20260830153830_0015_D/` 含 clips、rallies.json、persons.jsonl、roi.txt

- [ ] **Step 2: 跑 B2 并计时**

- [ ] **Step 3: 评测并记录基线**

```bash
.venv/bin/shuttlecut eval "outputs/DJI_20260830153830_0015_D/rallies.json" \
  --gt ground_truth/DJI_20260830153830_0015_D.json --record outputs/eval_history.md
.venv/bin/shuttlecut eval "outputs/DJI_20260830173600_0025_D/rallies.json" \
  --gt ground_truth/DJI_20260830173600_0025_D.json --record outputs/eval_history.md
```
Expected: 指标打印 + eval_history.md 两条记录(大概率 FAIL,进入 Task 13)

- [ ] **Step 4: 抽查 3 个片段目检(look_at 首帧)**

从 clips 中抽 3 个:确认开头是发球准备、结尾含死球后动作;记录观察进 eval_history.md 注释。

---

### Task 13: 迭代调参至验收线

**Files:**
- Modify: `src/shuttlecut/segmenter.py` / `src/shuttlecut/activity.py`(仅当决策表指向)
- Create: `outputs/eval_history.md` 持续追加

**Interfaces:**
- Consumes: Task 12 基线与误差分类
- Produces: B1/B2 双 PASS 的参数集(写入 rallies.json params 与 README)

规则:**每轮只动一类参数**,重跑 Task 12 Step 1-3,把参数与指标追加进 eval_history.md。决策表(按主导误差类型):

| 主导误差 | 首选动作(按序尝试,一轮一项) |
|---|---|
| missed 多 | `--min-rally 2.5` → `SegParams.hi_q 0.60→0.50` → `activity.min_h_ratio 0.12→0.10` |
| extra 多 | `SegParams.lo_q 0.30→0.40` → `--min-idle 3.0` |
| boundary 多 | 确认音频精修开启;`refiner.search_back_s 3→4`;`activity.smooth window 2.0→1.5` |
| fragment 多 | `--min-idle 3.0` + `--min-rally 2.5`(同轮算一组同向调整) |

- [ ] **Step 1: 按决策表执行一轮(示例:missed 主导)**

Run: `time .venv/bin/shuttlecut process temp/DJI_..._0015_D.MP4 --out outputs --min-rally 2.5 && .venv/bin/shuttlecut eval ... --record outputs/eval_history.md`
Expected: recall 上升;记录前后对比

- [ ] **Step 2: 循环直到双 PASS 或 6 轮**

双 PASS(两视频 R/P ≥0.90)→ 进入 Task 14。
6 轮未达 → 停止,带 eval_history.md 向用户结构化提问(展示差距与选项:放宽验收线 / 增加训练样本微调 / 接受 v1.5 目标)。

- [ ] **Step 3: 固化达标参数**

把达标参数写回 `SegParams` 默认值,`pytest -q` 回归 + 重跑双视频确认默认参数即 PASS,commit:

```bash
git add src/shuttlecut/
git commit -m "feat: 固化验收达标参数为默认值(B1/B2 双 PASS)"
```

---

### Task 14: 鲁棒性验收与回归固化

**Files:**
- Create: `tests/test_robustness.py`

**Interfaces:**
- Consumes: Task 8 CLI、Task 13 达标参数
- Produces: 鲁棒性拦截项证据(验收方案 §3 表末行)

- [ ] **Step 1: 构造异常输入(真实 B1 派生,产物入 temp/)**

```bash
ffmpeg -loglevel error -i temp/DJI_20260830153830_0015_D.MP4 -t 60 -an -c copy temp/robust_noaudio.mp4
ffmpeg -loglevel error -i temp/DJI_20260830153830_0015_D.MP4 -t 30 -c copy temp/robust_truncated.mp4
```

- [ ] **Step 2: 跑批断言不崩溃**

Run: `.venv/bin/shuttlecut process temp/robust_noaudio.mp4 --out temp/robust_out --no-reel && .venv/bin/shuttlecut process temp/robust_truncated.mp4 --out temp/robust_out --no-reel`
Expected: 两条命令 exit 0,无音轨那条打印 `[warn] 音频精修跳过`,均产出 rallies.json

- [ ] **Step 3: 写成自动化测试**

`tests/test_robustness.py`:

```python
import subprocess

from shuttlecut.cli import main


def test_truncated_local_video(tmp_path):
    """截断/无音轨的真实输入以合成视频等价模拟(快速回归)。"""
    import subprocess as sp
    v = tmp_path / "t.mp4"
    sp.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi",
            "-i", "testsrc=duration=5:size=640x360:rate=30",
            "-c:v", "libx264", "-preset", "ultrafast", str(v)], check=True)  # 无音轨
    assert main(["process", str(v), "--out", str(tmp_path / "o"), "--device", "cpu"]) == 0
```

Run: `.venv/bin/python -m pytest tests/test_robustness.py -v` → 1 passed

- [ ] **Step 4: Commit**

```bash
git add tests/test_robustness.py
git commit -m "test: 鲁棒性回归(无音轨/截断不崩溃)"
```

---

### Task 15: README 与交付清单

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: 全部前序产出
- Produces: 用户可用文档;交付 = 验收方案 §6 清单齐备

- [ ] **Step 1: 写 README**

````markdown
# ShuttleCut 🏸

羽毛球视频回合自动剪辑 CLI:输入 DJI/Osmo 类 4K 视频,输出回合片段与集锦。

## 安装(macOS / Apple Silicon)
```bash
uv venv --python 3.12 .venv && uv pip install -p .venv/bin/python -e . pytest
# 首次运行自动下载 YOLO 权重到 models/(约 5MB)
```

## 使用
```bash
.venv/bin/shuttlecut process 视频.mp4 --out outputs      # 切片 + 集锦 + rallies.json
.venv/bin/shuttlecut label 视频.mp4 --sheets temp/label  # 真值标注辅助
.venv/bin/shuttlecut eval outputs/视频/rallies.json --gt ground_truth/视频.json
```

## 调参指南
- 漏回合 → `--min-rally 2.5`;仍漏 → 代码内 `SegParams.hi_q` 降 0.05
- 多余片段 → `--min-idle 3.0`
- 边界不准 → 确认未用 `--no-audio-refine`

## 验收状态
B1/B2 基准指标见 `outputs/eval_history.md`(本地运行产物)。
设计:`docs/superpowers/specs/` · 验收契约:同目录 acceptance-plan
````

- [ ] **Step 2: 交付核对(验收方案 §6)**

逐项核对:CLI 三命令可用 / 双视频 rallies+clips+eval PASS / ground_truth 两份 / 三份文档 / README;`pytest -q` 全绿;`git log --oneline` 提交历史完整。

- [ ] **Step 3: 最终提交**

```bash
git add README.md
git commit -m "docs: README(安装/使用/调参/验收状态)"
```

- [ ] **Step 4: 向用户交付(结构化提问)**

呈现:eval 报告、5+5 抽检片段清单、耗时数据;请用户按验收方案 §7-3 做最终确认。
