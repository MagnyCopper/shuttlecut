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
