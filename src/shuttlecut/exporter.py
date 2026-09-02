import subprocess
from pathlib import Path

from shuttlecut.segmenter import Rally


def _video_encoder() -> list[str]:
    """优先 VideoToolbox 硬编(4K 切片快 ~10 倍),不可用回退 libx264。"""
    probe = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                           capture_output=True, text=True, check=True)
    if "h264_videotoolbox" in probe.stdout:
        return ["-c:v", "h264_videotoolbox", "-b:v", "10M"]
    return ["-c:v", "libx264", "-preset", "fast", "-crf", "20"]


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
             "-i", video, *_video_encoder(),
             "-c:a", "aac", "-movflags", "+faststart", str(dest)],
            check=True,
        )
        clips.append(str(dest))
    return clips


def _concat_line(path: str) -> str:
    """concat 行;路径含单引号时转义(闭引号+转义引号+重开引号)。"""
    escaped = path.replace("'", "'\\''")
    return f"file '{escaped}'"


def export_reel(clips: list[str], out_path: str) -> str:
    lst = Path(out_path).with_suffix(".txt")
    lst.write_text("\n".join(_concat_line(c) for c in clips))
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(lst), "-c", "copy", out_path],
        check=True,
    )
    return out_path
