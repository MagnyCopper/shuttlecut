import argparse
import json
from datetime import datetime
from pathlib import Path

from shuttlecut import __version__
from shuttlecut.activity import auto_roi, motion_energy, smooth
from shuttlecut.detector import detect_persons, load_persons_jsonl
from shuttlecut.exporter import export_clips, export_reel
from shuttlecut.refiner import audio_transients, refine
from shuttlecut.sampler import extract_audio, extract_frames, probe
from shuttlecut.segmenter import Rally, SegParams, segment


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="shuttlecut", description="羽毛球回合自动剪辑")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd")

    pr = sub.add_parser("process", help="切分回合并导出片段")
    pr.add_argument("videos", nargs="+")
    pr.add_argument("--out", default="outputs")
    pr.add_argument("--roi", default=None, help="X,Y,W,H 覆盖自动 ROI")
    pr.add_argument("--min-rally", type=float, default=SegParams().min_rally_s)
    pr.add_argument("--min-idle", type=float, default=SegParams().min_idle_s)
    pr.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    pr.add_argument("--no-audio-refine", action="store_true")
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


def _parse_roi(s: str | None):
    if not s:
        return None
    x, y, w, h = (float(v) for v in s.split(","))
    return (x, y, w, h)


def process_one(video: str, out_root: str, args) -> int:
    meta = probe(video)
    stem = Path(video).stem
    outdir = Path(out_root) / stem
    work = Path("temp/work") / stem
    frames = extract_frames(video, str(work / "frames"), fps=5.0, width=1280)
    persons_path = outdir / "persons.jsonl"
    rows = detect_persons(frames, frame_fps=5.0, device=args.device,
                          out_jsonl=str(persons_path))
    pmeta, rows = load_persons_jsonl(str(persons_path))
    frame_h = float(pmeta["frame_h"])

    roi = _parse_roi(args.roi)
    if roi is None:
        roi = auto_roi(rows, float(pmeta["frame_w"]), frame_h)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "roi.txt").write_text(",".join(f"{v:g}" for v in roi))
    energy = smooth(motion_energy(rows, frame_h, roi=roi), window_s=2.0, fps=5.0)

    params = SegParams(min_rally_s=args.min_rally, min_idle_s=args.min_idle)
    rallies: list[Rally] = segment(energy, params)

    if not args.no_audio_refine:
        try:
            wav = extract_audio(video, str(work / "audio.wav"))
            rallies = refine(rallies, audio_transients(wav))
        except Exception as e:  # 无音轨/解码失败 → 降级纯视觉
            print(f"[warn] 音频精修跳过: {e}")

    clips = export_clips(video, rallies, str(outdir / "clips"),
                         pre_s=params.pre_roll_s, post_s=params.post_roll_s)
    if not args.no_reel and clips:
        export_reel(clips, str(outdir / "clips" / "highlights.mp4"))

    payload = {
        "video": stem,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "params": vars(params) | {"roi": roi},
        "rallies": [
            {"id": i, "start_s": round(r.start, 2), "end_s": round(r.end, 2),
             "duration_s": round(r.end - r.start, 2), "hits": r.hits,
             "motion_peak": round(r.motion_peak, 2), "confidence": round(r.confidence, 3)}
            for i, r in enumerate(rallies, start=1)
        ],
    }
    (outdir / "rallies.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    total = sum(r.end - r.start for r in rallies)
    print(f"[summary] {stem}: {meta.duration_s:.0f}s → {len(rallies)} 个回合, "
          f"共 {total:.0f}s ({total / meta.duration_s * 100:.0f}% 保留)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd is None:
        build_parser().print_help()
        return 1
    if args.cmd == "process":
        for video in args.videos:
            process_one(video, args.out, args)
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
            rallies = json.loads(Path(f"temp/label/{Path(args.video).stem}/draft.json").read_text())
            save_gt(args.out, Path(args.video).stem, rallies)
            print(f"真值已保存 → {args.out}")
        return 0
    if args.cmd == "eval":
        from shuttlecut.eval.evaluate import evaluate
        from shuttlecut.labeling.gt import load_gt
        det = [(r["start_s"], r["end_s"])
               for r in json.loads(Path(args.rallies_json).read_text())["rallies"]]
        g = [(r["start_s"], r["end_s"]) for r in load_gt(args.gt)["rallies"]]
        rep = evaluate(det, g)
        verdict = "PASS" if rep.recall >= 0.90 and rep.precision >= 0.90 else "FAIL"
        print(f"recall={rep.recall:.3f} precision={rep.precision:.3f} mae={rep.mae_s:.2f}s "
              f"({rep.n_det} 检出 / {rep.n_gt} 真值) → {verdict}")
        print(f"missed={len(rep.missed)} extra={len(rep.extra)} "
              f"fragment={len(rep.fragment)} boundary={len(rep.boundary)}")
        if args.record:
            with open(args.record, "a") as f:
                f.write(f"| {datetime.now().isoformat(timespec='seconds')} | {args.rallies_json} | "
                        f"{rep.recall:.3f} | {rep.precision:.3f} | {rep.mae_s:.2f} | {verdict} |\n")
        return 0 if verdict == "PASS" else 2
    print("该子命令在后续任务实现")
    return 0
