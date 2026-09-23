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
    """首选编码参数(兼容旧调用);实战降级链见 ffmpeg.media_run。"""
    from shuttlecut.ffmpeg import _encoder_chain
    return _encoder_chain()[0]


def export_clips(video: str, rallies: list[Rally], out_dir: str,
                 pre_s: float = 1.5, post_s: float = 3.5,
                 progress=None) -> list[str]:
    """串行切片;解码/编码降级链由 media_run 兜底(坏驱动环境自动降档)。"""
    from shuttlecut.ffmpeg import media_run
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clips: list[str] = []
    for i, r in enumerate(rallies, start=1):
        ss = max(0.0, r.start - pre_s)
        to = r.end + post_s
        dest = out / f"rally_{i:03d}.mp4"
        media_run(lambda dec, enc: [
            "ffmpeg", "-loglevel", "error", "-y", *dec,
            "-ss", f"{ss:.3f}", "-to", f"{to:.3f}",
            "-i", video, *enc,
            "-c:a", "aac", "-movflags", "+faststart", str(dest)])
        clips.append(str(dest))
        if progress:
            progress(i, len(rallies))
    return clips


def _concat_line(path: str) -> str:
    """concat 行;路径含单引号时转义(闭引号+转义引号+重开引号)。"""
    escaped = path.replace("'", "'\\''")
    return f"file '{escaped}'"


def export_reel(clips: list[str], out_path: str, list_dir: str | None = None) -> str:
    """拼接集锦;list_dir 指定时 concat 列表写入该目录(避免污染输出目录)。"""
    from shuttlecut.ffmpeg import run_ffmpeg
    lst = (Path(list_dir) if list_dir else Path(out_path).parent) / (Path(out_path).stem + ".txt")
    lst.parent.mkdir(parents=True, exist_ok=True)
    # concat 条目用绝对路径:ffmpeg 相对*列表文件目录*解析相对路径,会加倍
    lst.write_text("\n".join(_concat_line(str(Path(c).resolve())) for c in clips), encoding="utf-8")
    run_ffmpeg(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
         "-i", str(lst), "-c", "copy", out_path])
    return out_path
