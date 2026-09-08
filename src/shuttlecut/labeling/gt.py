import json
from pathlib import Path


class GTError(ValueError):
    pass


def _validate_rallies(rallies: list[dict]) -> None:
    if not rallies:
        raise GTError("rallies 为空")
    prev_end = -1.0
    for r in rallies:
        if not (0 <= r["start_s"] < r["end_s"]):
            raise GTError(f"非法区间: {r}")
        if r["start_s"] < prev_end:
            raise GTError(f"区间重叠或乱序: {r}")
        prev_end = r["end_s"]


def save_gt(path: str, video_stem: str, rallies: list[dict]) -> None:
    _validate_rallies(rallies)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(
        {"video": video_stem, "labeled_by": "agent+user-spotcheck",
         "rallies": rallies},
        ensure_ascii=False, indent=2), encoding="utf-8")


def load_gt(path: str) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_rallies(d.get("rallies"))
    return d
