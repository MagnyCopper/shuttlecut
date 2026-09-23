import subprocess
from dataclasses import dataclass
from pathlib import Path

from shuttlecut.ffmpeg import hwaccel_decode


@dataclass
class Rally:
    start: float
    end: float
    motion_peak: float
    confidence: float
    hits: int = 0


def _video_encoder() -> list[str]:
    """优先硬编(NVENC/VideoToolbox)输出 1080p;回退 libx264 也必须缩到 1080p
    (4K 软编码病理慢:实测 10s 片段 52s)。"""
    probe = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                           capture_output=True, text=True, check=True)
    if "h264_nvenc" in probe.stdout:
        return ["-c:v", "h264_nvenc", "-b:v", "8M",
                "-vf", "scale=-2:1080,format=yuv420p"]
    if "h264_videotoolbox" in probe.stdout:
        # 10bit HEVC 源→8bit 1080p:避免硬编不收 10bit 导致的软转换慢路径
        return ["-c:v", "h264_videotoolbox", "-b:v", "8M",
                "-vf", "scale=-2:1080,format=yuv420p"]
    return ["-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-vf", "scale=-2:1080,format=yuv420p"]


def export_clips(video: str, rallies: list[Rally], out_dir: str,
                 pre_s: float = 1.5, post_s: float = 2.0,
                 progress=None) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clips: list[str] = []
    for i, r in enumerate(rallies, start=1):
        ss = max(0.0, r.start - pre_s)
        to = r.end + post_s
        dest = out / f"rally_{i:03d}.mp4"
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", *hwaccel_decode(),
             "-ss", f"{ss:.3f}", "-to", f"{to:.3f}",
             "-i", video, *_video_encoder(),
             "-c:a", "aac", "-movflags", "+faststart", str(dest)],
            check=True,
        )
        clips.append(str(dest))
        if progress:
            progress(i, len(rallies))
    return clips
    return clips


def _concat_line(path: str) -> str:
    """concat 行;路径含单引号时转义(闭引号+转义引号+重开引号)。"""
    escaped = path.replace("'", "'\\''")
    return f"file '{escaped}'"


def export_reel(clips: list[str], out_path: str, list_dir: str | None = None) -> str:
    """拼接集锦;list_dir 指定时 concat 列表写入该目录(避免污染输出目录)。"""
    lst = (Path(list_dir) if list_dir else Path(out_path).parent) / (Path(out_path).stem + ".txt")
    lst.parent.mkdir(parents=True, exist_ok=True)
    # concat 条目用绝对路径:ffmpeg 相对*列表文件目录*解析相对路径,会加倍
    lst.write_text("\n".join(_concat_line(str(Path(c).resolve())) for c in clips), encoding="utf-8")
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
         "-i", str(lst), "-c", "copy", out_path],
        check=True,
    )
    return out_path
