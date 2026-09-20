"""模型分发:manifest 加载 + 首用自动下载(whisper 模式,SHA-256 校验)。"""
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path


def load_manifest() -> dict:
    p = Path(__file__).parent / "model_manifest.json"
    return json.loads(p.read_text(encoding="utf-8"))


def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download_model(dest: str, manifest: dict | None = None, progress=None) -> str:
    """按 manifest 下载并校验模型到 dest;失败抛异常(不留半成品)。"""
    m = manifest or load_manifest()
    dest_p = Path(dest)
    dest_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_p.with_suffix(".part")
    url = m["url"]
    total = int(m.get("bytes") or 0)
    got = 0
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            got += len(b)
            if progress:
                pct = f"{got * 100 // total}%" if total else f"{got >> 20}MB"
                progress(f"下载 {m['version']} {pct}")
    if progress:
        progress("校验 SHA-256…")
    actual = sha256_of(str(tmp))
    if actual != m["sha256"]:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"模型校验失败: 期望 {m['sha256'][:12]}… 实际 {actual[:12]}…")
    shutil.move(str(tmp), str(dest_p))
    return str(dest_p)
