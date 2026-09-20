"""ShuttleCut CLI.

设计规范(参考 yt-dlp/gh/ruff 社区惯例):
- 动词式子命令;常用路径零参数可用(process INPUT 即出片)
- 输出契约固定:process 恰好产出 2 个视频(all-rallies + highlights)
- 进度/诊断走 stderr,stdout 只留最终摘要
- 退出码:0 成功 / 1 处理失败 / 2 用法错误(argparse 默认)
- 副产物(JSON 元数据)仅显式 --write-metadata 生成
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from shuttlecut import __version__
from shuttlecut.ffmpeg import probe

EXAMPLES = """操作总纲(本 CLI 完全自说明,每个子命令 --help 含全部协议):
  场景决策:常规出片→process;新场馆/质量敏感→先 calibrate 再 process;
           质量验证→process --write-metadata + eval;训练新模型→tools/cuda/train_heavy.py
  模型三层:models/shuttlecut-<stem>.pt(校准,优先) → models/shuttlecut.pt(官方,
  缺失时自动下载) → models/exp/(实验,需 --model 显式指定)
  通用契约:进度→stderr;摘要→stdout;退出码 0 成功/1 失败/2 用法错

示例:
  shuttlecut process match.mp4                        # 输出 2 个视频到 ./shuttlecut-output/
  shuttlecut process match.mp4 -o exports --overwrite
  shuttlecut process match.mp4 --write-metadata      # 额外输出 rallies.json(供 eval)
  shuttlecut calibrate match.mp4                      # 新视频 5 分钟校准(协议见 calibrate --help)
"""

PROCESS_EPILOG = """输出契约(恰好 2 个文件):
  <out-dir>/<stem>-all-rallies.mp4    全部回合(时间顺序)
  <out-dir>/<stem>-highlights.mp4     精选回合(精彩度降序)
流契约:进度/警告→stderr(阶段 [1/5]..[5/5]+百分比);摘要+输出路径→stdout;
  长时间无 stdout 属正常,勿判卡死。首次处理某视频较慢(建帧/差分缓存),重复处理显著加快。
模型解析(默认无需指定):models/shuttlecut-<stem>.pt(校准优先) →
  models/shuttlecut.pt(官方;缺失时按内置 manifest 自动下载+SHA-256 校验)。
退出码:0 成功 / 1 失败 / 2 用法错。
Agent 处置规则:
  - exit 1 "输出已存在" → 征得同意后加 --overwrite 重跑
  - exit 1 "未找到时序模型"且下载失败 → 手动放置 models/shuttlecut.pt 或先 calibrate
  - exit 1 "未检出任何回合" → 视频可能无对打/场馆差异大;跑 calibrate 重适配,仍无则报告
  - stderr 出现 "[warn] 音频特征不可用"(audio 缺失) → 仅降级评分排序,非失败,忽略
  - 多模型投票(高级):--model a.pt,b.pt,c.pt(实测优于单模型)"""

CALIB_EPILOG = """两步协议(新视频实测 P/R 0.9-1.0;零训练则 0.1-0.5 逐视频抽签):
  phase 1(默认):渲染 N 张条带到 shuttlecut-output/<stem>/calib/strip_XX.jpg,
    每张含 6 帧(左上→右下时间递增,约 3s 间隔)+ calib_template.json 模板。
  人工/LLM 标注:逐帧判【近场】(画面主体球场):
    Y = 回合中(对打/发球瞬间/击球后移动取位)
    N = 停顿(走动捡球/站立休息/换场/仅远场有人打/空场)
    多球场只判近场;不确定帧取回合倾向;6 帧全 N 合法(纯休息段)。
  提交:模板每条 verdict 改为 6 值 Y/N(如 YYNNYY),不增删字段,存为
    shuttlecut-output/<stem>/calib/calib.json
  phase 2:--phase run --calib <该文件> → TTA 适配(需 GPU,5-15 分钟)
    产物 models/shuttlecut-<stem>.pt,此后 process 自动优先使用,无需 --model。
  质量守则(quality):标注一致性 > 覆盖率;<5 分钟视频 --strips 24,>25 分钟 --strips 60。
  验证回路:calibrate → process --write-metadata → eval --gt <真值>;
    未达 0.9 → 检查边界帧标注质量重标 → 重跑。

示例:
  shuttlecut calibrate match.mp4                                    # phase 1: 出 40 张条带
  shuttlecut calibrate match.mp4 --phase run --calib \\
      shuttlecut-output/<stem>/calib/calib.json                     # phase 2: TTA 适配
  shuttlecut process match.mp4                                      # 出片(自动用校准模型)
"""

EVAL_EPILOG = """用于质量验证回路:
  shuttlecut process X.MP4 --write-metadata
  shuttlecut eval shuttlecut-output/<stem>-rallies.json --gt data/ground_truth/<stem>.json
  输出 PASS/FAIL(P/R ≥0.90)与 missed/extra/fragment/boundary 分解;--record 追加台账。
  真值格式同 data/ground_truth/*.json;构建训练真值用 tools/autolabel/auto.py + 条带法。"""


def _err(msg: str) -> None:
    print(f"shuttlecut: {msg}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="shuttlecut",
        description="羽毛球整场视频 → 回合合集 + 精彩选集(恰好 2 个输出视频)。",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-V", "--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", metavar="COMMAND")

    pr = sub.add_parser(
        "process", help="切分回合并输出 2 个视频(all-rallies + highlights)",
        epilog=PROCESS_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    pr.add_argument("videos", nargs="+", metavar="INPUT", help="源视频路径(可多个)")
    pr.add_argument("-o", "--output-dir", default="shuttlecut-output", metavar="DIR",
                    help="输出目录(默认: ./shuttlecut-output)")
    pr.add_argument("--model", default=None, metavar="CKPT[,CKPT...]",
                    help="时序模型 ckpt;逗号分隔多模型启用边界投票。"
                            "默认自动查找: models/shuttlecut-<stem>.pt(校准优先) → models/shuttlecut.pt(官方)")
    pr.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    pr.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出")
    pr.add_argument("--write-metadata", action="store_true",
                    help="额外输出 <stem>-rallies.json(时间戳/评分,机器可读)")
    pr.add_argument("--quiet", action="store_true", help="抑制进度日志(仅保留最终摘要)")

    cb = sub.add_parser(
        "calibrate", help="新视频 5 分钟人工校准:条带标注 → TTA 适配模型",
        epilog=CALIB_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    cb.add_argument("video", metavar="INPUT", help="源视频路径")
    cb.add_argument("--phase", choices=["prepare", "run"], default="prepare",
                    help="prepare=渲染条带+模板(默认);run=读标注→TTA→适配模型")
    cb.add_argument("--strips", type=int, default=40, metavar="N",
                    help="条带数(默认: 40,约 5 分钟标注量)")
    cb.add_argument("--calib", metavar="FILE", help="run 阶段:标注 JSON 路径")
    cb.add_argument("--model", metavar="CKPT", help="基础模型(默认: models/shuttlecut.pt)")
    cb.add_argument("--epochs", type=int, default=3, metavar="N", help="TTA 轮数(默认: 3)")
    cb.add_argument("--lr", type=float, default=5e-5, metavar="LR", help="TTA 学习率(默认: 5e-5)")

    ev = sub.add_parser("eval", help="对比检测结果与真值(开发用)", epilog=EVAL_EPILOG,
                         formatter_class=argparse.RawDescriptionHelpFormatter)
    ev.add_argument("rallies_json", metavar="JSON", help="process --write-metadata 生成的 rallies.json")
    ev.add_argument("--gt", required=True, metavar="GT_JSON", help="真值 JSON(data/ground_truth/ 下)")
    ev.add_argument("--record", metavar="FILE", help="把本次结果追加为 Markdown 表格行")

    lb = sub.add_parser("label", help="真值标注辅助(开发用)")
    lb.add_argument("video", metavar="INPUT", help="源视频路径")
    lb.add_argument("--sheets", metavar="OUTDIR", help="生成接触表到该目录")
    lb.add_argument("--strip", nargs=2, type=float, metavar=("T0", "T1"), help="生成 [T0,T1] 秒密集帧条")
    lb.add_argument("--out", metavar="FILE", help="把 temp/label/<stem>/draft.json 存为真值")

    cc = sub.add_parser("cache", help="查看/清理可重建缓存(帧库/差分/切片)",
                        description="列出或删除 temp/work 下的可再生缓存(帧库 frames15、差分 diffs_cache、"
                        "切片 cut、音频 audio.wav)。默认 dry-run 只列出;--yes 才执行删除。"
                        "源视频(temp/*.MP4, temp/bili/)与模型不属缓存,永不触碰。")
    cc.add_argument("--yes", action="store_true", help="实际执行删除(默认 dry-run)")
    cc.add_argument("--older-than", type=float, default=0, metavar="DAYS",
                    help="仅清理 DAYS 天未访问的缓存(默认: 全部)")
    return p


# 模型命名规范(全链路):
#   models/shuttlecut.pt          —— 官方生产模型(唯一默认;由 models/exp/ 评审后晋升)
#   models/shuttlecut-<stem>.pt   —— 视频专属校准模型(calibrate 产物,存在则优先)
#   models/exp/<tag>.pt           —— 实验沙盒(train_heavy 直接产物,不对外)
DEFAULT_MODELS = ("models/shuttlecut.pt",)


def _resolve_models(stem: str, model_arg: str | None) -> list[str]:
    if model_arg:
        return [m.strip() for m in model_arg.split(",") if m.strip()]
    for cand in (f"models/shuttlecut-{stem}.pt", *DEFAULT_MODELS):
        if Path(cand).exists():
            return [cand]
    return []


def _check_input(video: str) -> bool:
    if not Path(video).exists():
        _err(f"输入视频不存在: {video}(检查路径/扩展名)")
        return False
    return True


def _ensure_default_model(progress=None) -> str | None:
    """models/shuttlecut.pt 缺失时按 manifest 自动下载(whisper 模式)。"""
    target = "models/shuttlecut.pt"
    if Path(target).exists():
        return target
    try:
        from shuttlecut.modelhub import download_model, load_manifest
        m = load_manifest()
        if not m.get("url"):
            return None
        if progress:
            progress(f"[model] 首次使用:下载 {m['version']}({m['bytes'] >> 20} MB)…")
        download_model(target, m, progress=progress)
        if progress:
            progress(f"[model] 就绪 → {target}")
        return target
    except Exception as e:
        Path(target + ".part").unlink(missing_ok=True)
        if progress:
            progress(f"[warn] 模型自动下载失败({e});请手动放置 {target}")
        return None


def process_one(video: str, out_root: str, args) -> int:
    import subprocess

    from shuttlecut.exporter import Rally, export_clips, export_reel

    if not _check_input(video):
        return 1
    stem = Path(video).stem
    meta = probe(video)
    outdir = Path(out_root)
    outdir.mkdir(parents=True, exist_ok=True)
    work = Path("temp/work") / stem

    all_path = outdir / f"{stem}-all-rallies.mp4"
    hi_path = outdir / f"{stem}-highlights.mp4"
    for target in (all_path, hi_path):
        if target.exists() and not args.overwrite:
            _err(f"输出已存在: {target}(用 --overwrite 覆盖)")
            return 1

    def progress(msg: str) -> None:
        if not args.quiet:
            print(msg, file=sys.stderr)

    ckpts = _resolve_models(stem, args.model)
    if not ckpts:
        got = _ensure_default_model(progress)
        if got:
            ckpts = [got]
        else:
            _err("未找到时序模型且自动下载失败。恢复:①检查网络后重试 ②手动放置 models/shuttlecut.pt ③先运行 calibrate(标注协议见 calibrate --help)")
            return 1
    missing = [c for c in ckpts if not Path(c).exists()]
    if missing:
        _err(f"模型不存在: {', '.join(missing)}")
        return 1


    from shuttlecut.temporal import (TWO_SCALE_DEFAULTS, boundary_vote, extract_frames15,
                                    infer_curve, load_diffs, two_scale_segments)
    frames = extract_frames15(video, str(work / "frames15"))
    if not args.quiet:
        print(f"配置: {meta.width}x{meta.height} @ {meta.fps:.0f}fps, {meta.duration_s:.0f}s "
              f"| 模型 {len(ckpts)} 个({','.join(Path(c).name for c in ckpts)}) "
              f"| 设备 {args.device} | 输出 {outdir}", file=sys.stderr)
    curves = []
    if len(ckpts) > 1 or not (work / "diffs_cache.npy").exists():
        progress(f"[1/5] 帧准备 {len(frames)} 帧…")
    diffs = load_diffs(frames, str(work / "diffs_cache.npy"),
                       progress=lambda m: progress(f"[1/5] {m}"))
    for mi, ck in enumerate(ckpts, 1):
        curves.append(infer_curve(frames, ck, device=args.device, batch=16 if args.device in ("auto", "cuda") else 8,
                                  diffs=diffs, progress=lambda m: progress(f"[2/5] 模型 {mi}/{len(ckpts)}: {m}")))
    centers, probs = curves[0]

    tune = {**TWO_SCALE_DEFAULTS, "sm": 9, "lo": 0.3, "min_len_s": 1.0, "mg": 2.0}
    if len(curves) > 1:
        segs = boundary_vote([two_scale_segments(c, p, None, None, **tune) for c, p in curves])
    else:
        segs = two_scale_segments(centers, probs, None, None, **tune)
    if not segs:
        _err("未检出任何回合。视频可能无对打内容或场馆差异过大;可跑 calibrate 重适配(见 calibrate --help),仍无则检查视频内容")
        return 1
    progress(f"[3/5] 切分完成: {len(segs)} 个回合")

    from shuttlecut.rank import score_rallies, top_rallies
    hits = None
    try:
        from shuttlecut.audio import audio_transients
        from shuttlecut.ffmpeg import extract_audio
        wav = extract_audio(video, str(work / "audio.wav"))
        hits = [tr.t for tr in audio_transients(wav)] or None
    except Exception as e:  # 无音轨时降级为纯视觉评分
        progress(f"[warn] 音频特征不可用: {e}")
    ranked = score_rallies(segs, centers, probs, hit_times=hits)

    progress(f"[4/5] 切片编码…")
    rallies = [Rally(start=a, end=b, motion_peak=float(probs.max()), confidence=1.0)
               for a, b in segs]
    clips = export_clips(video, rallies, str(work / "cut"),
                         progress=lambda i, n: progress(f"[4/5] 切片 {i}/{n}"))
    export_reel(clips, str(all_path), list_dir=str(work))
    progress(f"[5/5] 合并集锦…")
    top = top_rallies(ranked)
    pos = {id(r): i for i, r in enumerate(ranked)}
    top_clips = [clips[pos[id(r)]] for r in sorted(top, key=lambda r: r.rank)]  # 精彩度优先
    if top_clips and len(top_clips) < len(clips):
        export_reel(top_clips, str(hi_path), list_dir=str(work))
    else:
        hi_path.write_bytes(all_path.read_bytes())  # 全部即精选的退化情形

    if args.write_metadata:
        payload = {
            "video": stem,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "model": [Path(c).name for c in ckpts],
            "rallies": [{"id": i + 1, "start_s": round(a, 2), "end_s": round(b, 2)}
                        for i, (a, b) in enumerate(segs)],
            "ranking": [{"rank": r.rank, "start_s": round(r.start, 2), "end_s": round(r.end, 2),
                         "score": round(r.score, 3), "duration_s": round(r.features.duration_s, 2)}
                        for r in ranked],
        }
        (outdir / f"{stem}-rallies.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    total = sum(b - a for a, b in segs)
    print(f"{all_path}")
    print(f"{hi_path}")
    print(f"{stem}: {meta.duration_s:.0f}s → {len(segs)} 回合 {total:.0f}s "
          f"({total / meta.duration_s * 100:.0f}% 保留); 精选 top {len(top)}/{len(segs)}")
    if ranked:
        b = min(ranked, key=lambda r: r.rank)
        print(f"最精彩: {b.start:.0f}-{b.end:.0f}s ({b.features.duration_s:.0f}s)")
    return 0


def calibrate_cmd(args) -> int:
    """5 分钟校准协议:prepare 渲染条带+模板 → 人工标注 → run 组装 GT+TTA 适配。"""
    import subprocess
    import sys as _sys

    from shuttlecut.ffmpeg import probe
    from shuttlecut.temporal import extract_frames15

    if not _check_input(args.video):
        return 1

    stem = Path(args.video).stem
    outdir = Path("shuttlecut-output") / stem / "calib"
    outdir.mkdir(parents=True, exist_ok=True)
    work = Path("temp/work") / stem
    meta = probe(args.video)

    if args.phase == "prepare":
        extract_frames15(args.video, str(work / "frames15"))
        span = meta.duration_s / args.strips  # 每条带覆盖秒数
        step = min(3.0, span * 0.84 / 5)  # 帧间隔优先 3s(验证协议),短跨度自适应
        tmpl = []
        for i in range(args.strips):
            t0 = i * span + span * 0.08
            times = [t0 + k * step for k in range(6)]  # 6 帧集中(3s 间隔,覆盖 ~15s)
            imgs = []
            for j, t in enumerate(times):
                f = outdir / f"tmp_{i:02d}_{j}.jpg"
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t:.1f}", "-i", args.video,
                                "-frames:v", "1", "-vf", "scale=480:-2", "-q:v", "4", str(f)], check=True)
                imgs.append(f.name)
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y",
                            "-i", str(outdir / imgs[0]), "-i", str(outdir / imgs[1]), "-i", str(outdir / imgs[2]),
                            "-i", str(outdir / imgs[3]), "-i", str(outdir / imgs[4]), "-i", str(outdir / imgs[5]),
                            "-filter_complex",
                            "[0][1][2][3][4][5]xstack=inputs=6:layout=0_0|w0_0|w0+w1_0|0_h0|w0_h0|w0+w1_h0",
                            "-q:v", "4", str(outdir / f"strip_{i:02d}.jpg")], check=True)
            for f in imgs:
                (outdir / f).unlink(missing_ok=True)
            tmpl.append({"strip": i, "times": [round(t, 1) for t in times],
                         "verdict": "YYYYYY"})  # 用户改 Y/N
        (outdir / "calib_template.json").write_text(
            json.dumps({"video": stem, "strips": tmpl}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"条带已生成 → {outdir}\\strip_XX.jpg")
        print("标注协议:6 帧逐帧判近场,Y=回合中(对打/发球/取位) N=停顿(捡球/休息/远场);完整标准见 calibrate --help")
        print(f"逐张查看,把模板中 verdict 改为 6 值 Y/N(是/停顿),保存为 calib.json,然后运行:")
        print(f"  shuttlecut calibrate {args.video} --phase run --calib {outdir / 'calib.json'}")
        return 0

    calib = json.loads(Path(args.calib).read_text(encoding="utf-8"))
    segs = []
    for s in calib["strips"]:
        v = s["verdict"].upper().replace(" ", "")
        step = max((s["times"][-1] - s["times"][0]) / max(len(v) - 1, 1), 4.5)  # 段宽下限 4.5s>训练窗(64帧=4.27s)
        for k, ch in enumerate(v):
            if ch == "Y":
                t = s["times"][0] + k * step
                segs.append((t - step / 2, t + step / 2))
    merged = []
    for a, b in sorted(segs):
        if merged and a <= merged[-1][1] + 1.0:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    merged = [(round(a, 1), round(b, 1)) for a, b in merged if b - a >= 2.5]
    gt_path = outdir / "calib_gt.json"
    gt_path.write_text(json.dumps({"video": stem, "source": "calibrate 协议",
                                   "rallies": [{"id": i, "start_s": a, "end_s": b}
                                               for i, (a, b) in enumerate(merged, 1)]},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"标注组装 {len(merged)} 回合 → {gt_path}")

    ckpt = args.model or ("models/shuttlecut.pt" if Path("models/shuttlecut.pt").exists() else "models/exp/r3d_e16_s13.pt")
    if not Path(ckpt).exists():
        _err(f"基础模型不存在: {ckpt}")
        return 1
    adapted = f"models/shuttlecut-{stem}.pt"
    r = subprocess.run([_sys.executable, "tools/cuda/train_heavy.py",
                        "--frames", str(work / "frames15"), "--gt", str(gt_path),
                        "--init", ckpt, "--out", adapted, "--win", "64",
                        "--epochs", str(args.epochs), "--batch", "8",
                        "--device", "cuda", "--seed", "42", "--lr", str(args.lr)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        _err(f"TTA 失败: {(r.stderr or r.stdout).strip().splitlines()[-1]}")
        return 1
    print(f"适配模型 → {adapted}")
    print(f"出片: shuttlecut process {args.video}")
    return 0


def cache_cmd(args) -> int:
    """temp/work 可再生缓存的列出/清理(默认 dry-run)。"""
    import time

    work = Path("temp/work")
    if not work.exists():
        print("无缓存目录(temp/work 不存在)")
        return 0
    cache_names = {"frames15", "cut"}
    cache_files = {"diffs_cache.npy", "flow_cache.npy", "r3d_feat.npy", "audio.wav"}
    cutoff = time.time() - args.older_than * 86400 if args.older_than > 0 else None
    rows: list[tuple[str, int]] = []
    for d in sorted(work.iterdir()):
        if not d.is_dir():
            continue
        for name in cache_names:
            sub = d / name
            if sub.is_dir():
                ok = cutoff is None or sub.stat().st_atime < cutoff
                if ok:
                    rows.append((str(sub), sum(f.stat().st_size for f in sub.rglob("*") if f.is_file())))
        for fn in cache_files:
            f = d / fn
            if f.is_file():
                ok = cutoff is None or f.stat().st_atime < cutoff
                if ok:
                    rows.append((str(f), f.stat().st_size))
    if not rows:
        print("无可清理缓存" + (f"(>{args.older_than} 天未访问)" if args.older_than else ""))
        return 0
    total = sum(s for _, s in rows)
    verb = "已清理" if args.yes else "[dry-run] 将清理(--yes 执行)"
    for p, s in rows:
        print(f"  {s / 1e9:7.2f} GB  {p}")
    print(f"{verb}: {len(rows)} 项, 共 {total / 1e9:.1f} GB")
    if args.yes:
        for p, _ in rows:
            t = Path(p)
            if t.is_dir():
                import shutil
                shutil.rmtree(t, ignore_errors=True)
            else:
                t.unlink(missing_ok=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd is None:
        build_parser().print_help()
        return 1
    if args.cmd == "process":
        for video in args.videos:
            if process_one(video, args.output_dir, args) != 0:
                return 1
        return 0
    if args.cmd == "calibrate":
        return calibrate_cmd(args)
    if args.cmd == "label":
        from shuttlecut.labeling.gt import save_gt
        from shuttlecut.labeling.sheets import make_contact_sheets, make_dense_strip
        if args.sheets:
            files = make_contact_sheets(args.video, args.sheets)
            print(f"生成 {len(files)} 张接触表 → {args.sheets}")
        if args.strip:
            t0, t1 = args.strip
            make_dense_strip(args.video, t0, t1,
                             f"temp/label/{Path(args.video).stem}/strip_{t0:.0f}_{t1:.0f}.jpg")
            print(f"生成密集帧条 [{t0},{t1}]s")
        if args.out:
            rallies = json.loads(
                Path(f"temp/label/{Path(args.video).stem}/draft.json").read_text(encoding="utf-8"))
            save_gt(args.out, Path(args.video).stem, rallies)
            print(f"真值已保存 → {args.out}")
        return 0
    if args.cmd == "cache":
        return cache_cmd(args)
    if args.cmd == "eval":
        from shuttlecut.eval.evaluate import evaluate
        from shuttlecut.labeling.gt import load_gt
        det = [(r["start_s"], r["end_s"])
               for r in json.loads(Path(args.rallies_json).read_text(encoding="utf-8"))["rallies"]]
        g = [(r["start_s"], r["end_s"]) for r in load_gt(args.gt)["rallies"]]
        rep = evaluate(det, g)
        verdict = "PASS" if rep.recall >= 0.90 and rep.precision >= 0.90 else "FAIL"
        print(f"recall={rep.recall:.3f} precision={rep.precision:.3f} mae={rep.mae_s:.2f}s "
              f"({rep.n_det} 检出 / {rep.n_gt} 真值) → {verdict}")
        print(f"missed={len(rep.missed)} extra={len(rep.extra)} "
              f"fragment={len(rep.fragment)} boundary={len(rep.boundary)}")
        if args.record:
            with open(args.record, "a", encoding="utf-8") as f:
                f.write(f"| {datetime.now().isoformat(timespec='seconds')} | {args.rallies_json} | "
                        f"{rep.recall:.3f} | {rep.precision:.3f} | {rep.mae_s:.2f}s | {verdict} |\n")
        return 0 if verdict == "PASS" else 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
