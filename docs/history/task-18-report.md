# Task 18 Report

## Status

Implemented `shuttlecut.posefeat` with court-player selection, wrist/elbow speed, ready-stance detection, and per-frame feature aggregation.

## Changed files

- `src/shuttlecut/posefeat.py`
- `tests/test_posefeat.py`

## Verification

- RED: initial test collection failed because `shuttlecut.posefeat` did not exist.
- GREEN: `.venv/bin/python -m pytest tests/test_posefeat.py` — 4 passed.
- Full suite: `.venv/bin/python -m pytest` — 62 passed.
- LSP diagnostics were attempted for both changed Python files; basedpyright is not installed and was previously declined.

## Concerns

No functional concerns. LSP diagnostics remain unavailable due to the missing basedpyright installation.

## Task 18 审查修复（2026-09-03）

- 跨帧腕速改为空间最近邻立足点匹配，阈值 150px；移除 `frame_wh` 参数并同步计划接口。
- 补充换序、五人深度筛选、缺失关键点与场内 X 边界测试。
- `.venv/bin/python -m pytest tests/test_posefeat.py -v` — 8 passed。
- `.venv/bin/python -m pytest -q` — 66 passed。
