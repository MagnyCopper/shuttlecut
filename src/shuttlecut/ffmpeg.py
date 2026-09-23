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


_CAP_CACHE: dict[str, bool] = {}


def _enc_ok(name: str) -> bool:
    """功能性探测:1 帧实编(编译列表≠可用,无 GPU 机器也编译 nvenc)。"""
    if name not in _CAP_CACHE:
        r = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-f", "lavfi",
             "-i", "color=black:s=256x256:d=0.1", "-c:v", name, "-f", "null", "-"],
            capture_output=True, timeout=60)
        _CAP_CACHE[name] = r.returncode == 0
    return _CAP_CACHE[name]


def _dec_ok(hw: str) -> bool:
    """功能性探测:硬解 0.5s h264 样片(解码器必须吃真实编码流)。"""
    import tempfile
    key = f"dec:{hw}"
    if key not in _CAP_CACHE:
        ok = False
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                tmp = f.name
            subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi",
                 "-i", "color=black:s=256x256:d=0.5", "-c:v", "libx264", tmp],
                capture_output=True, timeout=60, check=True)
            r = subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-hwaccel", hw, "-i", tmp,
                 "-f", "null", "-"], capture_output=True, timeout=60)
            ok = r.returncode == 0
        except Exception:
            ok = False
        finally:
            Path(tmp).unlink(missing_ok=True)
        _CAP_CACHE[key] = ok
    return _CAP_CACHE[key]


def hwaccel_decode() -> list[str]:
    """硬解加速(4K HEVC 10bit 软解是瓶颈):cuda(NVDEC)→videotoolbox→空。
    功能性探测(编译列表会误报,无 GPU 环境实调即崩)。"""
    probe = subprocess.run(["ffmpeg", "-hide_banner", "-hwaccels"],
                           capture_output=True, text=True, check=True)
    if "cuda" in probe.stdout and _dec_ok("cuda"):
        return ["-hwaccel", "cuda"]
    if "videotoolbox" in probe.stdout and _dec_ok("videotoolbox"):
        return ["-hwaccel", "videotoolbox"]
    return []


def run_ffmpeg(cmd: list[str]) -> None:
    """统一 ffmpeg 调用:失败时携带 stderr 与恢复路径(不吞错误)。"""
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()[-4:]
        raise RuntimeError(
            f"ffmpeg 失败(rc={r.returncode}): {' '.join(cmd[:8])}…\n"
            + "\n".join(tail))


def extract_frames(video: str, outdir: str, fps: float = 5.0, width: int = 1280,
                   t_start: float | None = None, t_end: float | None = None) -> list[str]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for frame in out.glob("frame_*.jpg"):
        frame.unlink()
    cmd = ["ffmpeg", "-loglevel", "error", *hwaccel_decode()]
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
        ["ffmpeg", "-loglevel", "error", "-i", video, "-vn", "-ac", "1",  # 音频无需硬解
         "-ar", str(sr), out_wav],
        check=True,
    )
    return out_wav
