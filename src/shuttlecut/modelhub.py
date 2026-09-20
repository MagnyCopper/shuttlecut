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
    """按 manifest 下载并校验模型到 dest;失败抛异常(不留半成品)。
    匿名 HTTP 失败(如私有仓 404)时回退 gh release download(需 gh 已认证)。"""
    m = manifest or load_manifest()
    dest_p = Path(dest)
    dest_p.parent.mkdir(parents=True, exist_ok=True)
    url = m["url"]
    tmp = dest_p.with_suffix(".part")
    try:
        _download_url(url, tmp, m, progress)
    except Exception:
        _download_gh(m, tmp, progress)  # 私有仓回退
    if progress:
        progress("校验 SHA-256…")
    actual = sha256_of(str(tmp))
    if actual != m["sha256"]:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"模型校验失败: 期望 {m['sha256'][:12]}… 实际 {actual[:12]}…")
    shutil.move(str(tmp), str(dest_p))
    return str(dest_p)


def _download_url(url: str, tmp: Path, m: dict, progress) -> None:
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


def _download_gh(m: dict, tmp: Path, progress) -> None:
    """gh release download 回退(私有仓库;需 gh 认证)。"""
    import subprocess
    tag = m["url"].split("/download/")[1].split("/")[0]
    asset = m["url"].rsplit("/", 1)[1]
    if progress:
        progress(f"匿名下载失败,改用 gh(私有仓)拉取 {m['version']}…")
    r = subprocess.run(["gh", "release", "download", tag, "-R", _gh_repo(m["url"]),
                        "-p", asset, "-O", str(tmp)], capture_output=True, text=True)
    if r.returncode != 0:
        raise ValueError(f"gh 下载失败: {(r.stderr or r.stdout).strip().splitlines()[-1]}")


def _gh_repo(url: str) -> str:
    """https://github.com/<owner>/<repo>/releases/… → <owner>/<repo>"""
    parts = url.split("/")
    return f"{parts[3]}/{parts[4]}"
