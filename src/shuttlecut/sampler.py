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


def extract_frames(video: str, outdir: str, fps: float = 5.0, width: int = 1280,
                   t_start: float | None = None, t_end: float | None = None) -> list[str]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for frame in out.glob("frame_*.jpg"):
        frame.unlink()
    cmd = ["ffmpeg", "-loglevel", "error"]
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
        ["ffmpeg", "-loglevel", "error", "-i", video, "-vn", "-ac", "1",
         "-ar", str(sr), out_wav],
        check=True,
    )
    return out_wav
