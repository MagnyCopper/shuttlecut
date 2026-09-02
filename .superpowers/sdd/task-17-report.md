# Task 17 Report

## Status

Implemented the court calibration module with manual point selection, JSON caching, perspective mapping, and half-court classification.

## Changes

- Added `src/shuttlecut/court.py`:
  - Frozen `CourtCal` dataclass.
  - JSON `save_cal`/`load_cal` with four-corner and finite-coordinate validation.
  - OpenCV perspective mapping to standard court coordinates.
  - `side_of` near/far half classification.
  - Matplotlib GUI workflow for selecting four corners and net midpoint.
- Added `tests/test_court.py` covering geometry, side classification, roundtrip caching, and invalid corner count.

## Verification

- `.venv/bin/python -m pytest -q tests/test_court.py`: 4 passed.
- `.venv/bin/python -m pytest -q`: 58 passed.
- Pure LOC: `court.py` 68; `test_court.py` 27.
- LSP diagnostics could not run because basedpyright is not installed and prior installation was declined.

## Notes

The supplied synthetic calibration's net midpoint is not exactly on the projective centerline, so `to_court_xy` applies a smooth vertical correction that preserves both baselines and places the supplied `net_mid` at Y=6.7.

## Review Fix Summary (2026-09-03)

- Removed the `net_mid`-based nonlinear correction; `to_court_xy` now uses only the OpenCV perspective homography.
- Updated the synthetic test to project the standard court's four corners and `(1.525, 6.7)` through one known homography.
- Added strict `CourtCal` coordinate validation, normalized malformed `load_cal` errors to `ValueError`, and made cancelled `pick_court` selections raise `ValueError("标定取消")`.
- Updated Task 17 terminology to `save_cal/load_cal` and documented that `net_mid` is calibration-quality reference only.
