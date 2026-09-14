"""ShuttleCut CLI:时序管线(帧缓存→补偿差分→R3D 概率曲线→两尺度切分→出片)。"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from shuttlecut import __version__
from shuttlecut.exporter import Rally, export_clips, export_reel
from shuttlecut.ffmpeg import probe


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="shuttlecut", description="羽毛球回合自动剪辑(时序管线)")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd")

    pr = sub.add_parser("process", help="时序管线切分回合并导出片段")
    pr.add_argument("videos", nargs="+")
    pr.add_argument("--out", default="outputs")
    pr.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    pr.add_argument("--temporal-ckpt", default=None,
                    help="时序 ckpt 路径(默认按 stem 查找 models/r3d_<stem>_w64.pt)")
    pr.add_argument("--fine-ckpt", default=None,
                    help="W24 边界模型(默认 models/r3d_<stem>_w24.pt,存在才启用)")
    pr.add_argument("--no-reel", action="store_true")

    lb = sub.add_parser("label", help="真值标注辅助工具")
    lb.add_argument("video")
    lb.add_argument("--sheets", metavar="OUTDIR", default=None)
    lb.add_argument("--strip", nargs=2, type=float, metavar=("T0", "T1"), default=None)
    lb.add_argument("--out", default=None, help="保存真值 JSON 路径")

    ev = sub.add_parser("eval", help="对比检测结果与真值")
    ev.add_argument("rallies_json")
    ev.add_argument("--gt", required=True)
    ev.add_argument("--record", default=None)
    return p


def process_one(video: str, out_root: str, args) -> int:
    from shuttlecut.temporal import extract_frames15, infer_curve, two_scale_segments

    meta = probe(video)
    stem = Path(video).stem
    outdir = Path(out_root) / stem
    work = Path("temp/work") / stem
    outdir.mkdir(parents=True, exist_ok=True)

    ckpts = [c for c in (args.temporal_ckpt or f"models/r3d_{stem}_w64.pt").split(",") if c.strip()]
    for ck in ckpts:
        if not Path(ck).exists():
            print(f"[error] 未找到时序模型 {ck};请先用 tools/cuda/train_heavy.py 训练,"
                  f"或用 --temporal-ckpt 指定路径(多模型逗号分隔启用边界投票)")
            return 1
    ckpt = ckpts[0]
    fine_ckpt = args.fine_ckpt or f"models/r3d_{stem}_w24.pt"

    frames = extract_frames15(video, str(work / "frames15"))
    print(f"[temporal] {len(frames)} 帧 @15fps,模型 {','.join(ckpts)}")
    curves = [infer_curve(frames, ck, device=args.device) for ck in ckpts]
    centers, probs = curves[0]

    fine = (infer_curve(frames, fine_ckpt, win=24, tsub=2, device=args.device)
            if Path(fine_ckpt).exists() else (None, None))

    from shuttlecut.temporal import TWO_SCALE_DEFAULTS
    tune = {**TWO_SCALE_DEFAULTS, "sm": 9, "lo": 0.3, "min_len_s": 1.0, "mg": 2.0}
    if len(curves) > 1:
        from shuttlecut.temporal import boundary_vote
        segs = boundary_vote([two_scale_segments(c, p, fine[0], fine[1], **tune)
                              for c, p in curves])
    else:
        segs = two_scale_segments(centers, probs, fine[0], fine[1], **tune)
    rallies = [Rally(start=a, end=b, motion_peak=float(probs.max()), confidence=1.0)
               for a, b in segs]
    clips = export_clips(video, rallies, str(outdir / "clips"))
    if not args.no_reel and clips:
        export_reel(clips, str(outdir / "clips" / "highlights.mp4"))

    from shuttlecut.rank import score_rallies, top_rallies
    hits = None
    try:
        from shuttlecut.audio import audio_transients
        from shuttlecut.ffmpeg import extract_audio
        wav = extract_audio(video, str(outdir / "audio.wav"))
        hits = [tr.t for tr in audio_transients(wav)] or None
    except Exception as e:  # 无音轨/ffmpeg 异常时降级为纯视觉评分
        print(f"[warn] 音频击球特征不可用: {e}")
    ranked = score_rallies(segs, centers, probs, hit_times=hits)
    payload = {
        "video": stem,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "params": {"mode": "temporal", "ckpt": ",".join(ckpts)},
        "rallies": [{"id": i + 1, "start_s": round(r.start, 2), "end_s": round(r.end, 2)}
                    for i, r in enumerate(rallies)],
        "ranking": [{"rank": r.rank, "rally_id": i + 1, "start_s": round(r.start, 2),
                     "end_s": round(r.end, 2), "score": round(r.score, 3),
                     "duration_s": round(r.features.duration_s, 2),
                     "peak": round(r.features.peak, 3), "var": round(r.features.var, 4)}
                    for i, r in enumerate(ranked)],
    }
    (outdir / "rallies.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_reel and clips:
        top = top_rallies(ranked)
        pos = {id(r): i for i, r in enumerate(ranked)}
        top_clips = [clips[pos[id(r)]] for r in sorted(top, key=lambda r: r.rank)]  # 精彩度优先
        if 0 < len(top_clips) < len(clips):
            export_reel(top_clips, str(outdir / "clips" / "highlights_top.mp4"))
    total = sum(r.end - r.start for r in rallies)
    print(f"[summary] {stem}: {meta.duration_s:.0f}s → {len(rallies)} 个回合, "
          f"共 {total:.0f}s ({total / meta.duration_s * 100:.0f}% 保留)")
    if ranked:
        b = min(ranked, key=lambda r: r.rank)
        print(f"[top1] 回合 #{b.rank}: {b.start:.0f}-{b.end:.0f}s "
              f"({b.features.duration_s:.0f}s, peak={b.features.peak:.2f})")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd is None:
        build_parser().print_help()
        return 1
    if args.cmd == "process":
        for video in args.videos:
            if process_one(video, args.out, args) != 0:
                return 1
        return 0
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
            rallies = json.loads(Path(f"temp/label/{Path(args.video).stem}/draft.json").read_text(encoding="utf-8"))
            save_gt(args.out, Path(args.video).stem, rallies)
            print(f"真值已保存 → {args.out}")
        return 0
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
    print("该子命令在后续任务实现")
    return 0
