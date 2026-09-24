# ShuttleCut

English | [简体中文](README.md)

[![CI](https://github.com/MagnyCopper/shuttlecut/actions/workflows/ci.yml/badge.svg)](https://github.com/MagnyCopper/shuttlecut/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

Automatic badminton rally clipping — full-match video in, rally reels and
highlight clips out. **Zero-training inference on brand-new videos.**

## Pipeline

```
MP4 → 15fps frame extraction (cached, HW decode)
    → frozen V-JEPA 2 features + 3-seed probe ensemble (default path, cross-venue)
    → [--model explicit] LK-RANSAC compensated diffs → R3D-18 sliding window
    → boundary slope snapping → ffmpeg HW encode (auto fallback chain)
```

Measured results (official metric in `src/shuttlecut/eval/evaluate.py`,
tol 2.5 s / overlap ≥ 0.5×longer segment):

| Scenario | P / R |
|---|---|
| Cross-venue zero-training (**V-JEPA probe, default path**) | **0.85 / 0.84** (train DJI-14 → test 5 Bilibili venues; LOEO mean 0.83/0.80) |
| Calibration protocol (5-min labeling + TTA) | 1.000/1.000 and 0.898/0.800 (held-out) |
| R3D route · in-training-set | 0.83–1.00 |
| R3D route · same-venue family (zero-training) | 0.15–0.54 (boundary variance wall) |

> Historical finding (19-route experiment archive): under R3D + in-domain features,
> cross-video P/R of 0.1–0.5 is a configuration invariant. Route 20 (frozen
> foundation-model features) broke through that wall. Full experiment history,
> including every failure, lives in `docs/eval-history.md` (Chinese).

## Install

```bash
git clone https://github.com/MagnyCopper/shuttlecut.git && cd shuttlecut
uv venv .venv --python 3.12 && source .venv/bin/activate    # Windows: .venv\Scripts\activate
uv pip install -e .            # CPU torch, works out of the box

# NVIDIA GPU (driver ≥ CUDA 13.x):
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

brew install ffmpeg            # macOS; VideoToolbox HW encode/decode is auto-enabled on Apple Silicon
# Windows: winget install Gyan.FFmpeg
# macOS needs no CUDA: --device auto-selects mps (Metal)
```

Videos and weights stay local (never committed); the probe, official model, and
HF encoder cache (`models/hf`) are **auto-downloaded on first use** with SHA-256
verification.

## Usage

```bash
# Core: one command → exactly 2 videos (all-rallies reel + highlights)
shuttlecut process match.mp4
# outputs: shuttlecut-output/<stem>-all-rallies.mp4 + <stem>-highlights.mp4

shuttlecut process a.mp4 b.mp4 -o exports --overwrite   # batch / custom dir / overwrite
shuttlecut process a.mp4 --model m1.pt,m2.pt            # explicit R3D weights (calibrated / voting)
shuttlecut process a.mp4 --post 5                       # longer tail padding (shuttle mid-air cuts)

shuttlecut --help          # the CLI is fully self-documenting
```

Model resolution (uniform for every video, no per-video special cases):
`shuttlecut-probe.pt` (V-JEPA probe, auto-downloaded) → `shuttlecut.pt`
(official R3D fallback); calibrated/experimental weights are used explicitly
via `--model`. Exit codes: `0` ok / `1` failure / `2` usage error. Progress on
stderr, summary on stdout. Cold run ≈ 1.7× video duration on an RTX 2070;
re-runs only re-encode clips.

## Calibration protocol (single-video 0.9–1.0, optional)

```bash
shuttlecut calibrate new.mp4                       # 1. renders 40 strips + template
# 2. review strips, fill 6-value Y/N verdicts → calib.json   (see calibrate --help)
shuttlecut calibrate new.mp4 --phase run --calib calib.json  # 3. TTA adaptation (~10 min GPU)
shuttlecut process new.mp4 --model models/shuttlecut-<stem>.pt  # 4. explicit use
```

## Model naming (train → promote → serve)

```
models/
  shuttlecut.pt          # official R3D (2nd priority, auto fallback)
  shuttlecut-probe.pt    # V-JEPA probe (1st priority, cross-venue zero-calibration)
  shuttlecut-<stem>.pt   # per-video calibrated model (calibrate output, --model explicit)
  exp/                   # experiment sandbox (never auto-discovered)
```

## Repository layout

```
src/shuttlecut/          pipeline code (cli/temporal/exporter/ffmpeg/audio/rank/labeling/eval/vjepa/modelhub)
tools/cuda/             training suite (see its README)
tools/frontier/         probe experiments (vjepa_extract/loeo_probe/train_probe_prod)
tools/autolabel/        GT construction helpers
data/ground_truth/      segment ground truth (19 files: 14 DJI + 5 Bilibili venues)
tests/                  pytest (63 tests, 3-OS CI matrix)
docs/                   eval-history.md (full experiment ledger, zh) / specs / history
```

## Further reference

- `shuttlecut --help` / `<command> --help` — the single source of operational truth
  (decision guide, labeling protocol, stream contract, error recovery)
- `CHANGELOG.md` / `CONTRIBUTING.md` — history & contributing guide
- `docs/eval-history.md` — complete experiment ledger including all negative results (Chinese)

## License

MIT
