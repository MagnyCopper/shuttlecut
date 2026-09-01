import subprocess

import pytest


@pytest.fixture(scope="session")
def synth_video(tmp_path_factory: pytest.TempPathFactory) -> str:
    """20 秒 1280x720 testsrc 合成视频 + 正弦音轨,供采样/导出/精修测试复用。"""
    path = tmp_path_factory.mktemp("media") / "synth.mp4"
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=duration=20:size=1280x720:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=20",
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
            "-shortest", str(path),
        ],
        check=True,
    )
    return str(path)
