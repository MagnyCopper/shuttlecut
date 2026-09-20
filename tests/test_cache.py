"""cache 子命令测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shuttlecut.cli import main


def _mk_cache(tmp_path, monkeypatch, stale=False):
    monkeypatch.chdir(tmp_path)
    w = tmp_path / "temp" / "work" / "vid"
    (w / "frames15").mkdir(parents=True)
    (w / "frames15" / "frame_000001.jpg").write_bytes(b"x" * 1000)
    (w / "diffs_cache.npy").write_bytes(b"x" * 1000)
    if stale:
        import os
        import time
        old = time.time() - 40 * 86400
        for p in w.rglob("*"):
            os.utime(p, (old, old))
    return w


def test_cache_dry_run_default(tmp_path, monkeypatch, capsys):
    _mk_cache(tmp_path, monkeypatch)
    assert main(["cache"]) == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert (tmp_path / "temp" / "work" / "vid" / "frames15").exists()


def test_cache_yes_deletes(tmp_path, monkeypatch, capsys):
    w = _mk_cache(tmp_path, monkeypatch)
    assert main(["cache", "--yes"]) == 0
    assert not (w / "frames15").exists()
    assert not (w / "diffs_cache.npy").exists()


def test_cache_older_than_skips_fresh(tmp_path, monkeypatch, capsys):
    _mk_cache(tmp_path, monkeypatch, stale=False)
    assert main(["cache", "--older-than", "30", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "无可清理" in out
    assert (tmp_path / "temp" / "work" / "vid" / "frames15").exists()
