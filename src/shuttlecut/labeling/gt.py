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
