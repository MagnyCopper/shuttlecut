import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from shuttlecut import __version__
from shuttlecut.activity import auto_roi, motion_energy, smooth
from shuttlecut.armed import ArmedParams, segment_armed
from shuttlecut.court import load_cal, pick_court
from shuttlecut.detector import detect_persons, load_persons_jsonl
from shuttlecut.exporter import export_clips, export_reel
from shuttlecut.posefeat import frame_features
from shuttlecut.poses import estimate_poses, load_poses_jsonl
from shuttlecut.refiner import Transient, audio_transients, refine
from shuttlecut.sampler import VideoMeta, extract_audio, extract_frames, probe
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
    pr.add_argument("--pose", action="store_true", help="姿态管线(RTMPose+球场标定)")
    pr.add_argument("--flow", action="store_true", help="光流管线(15fps 残差动作事件)")
    pr.add_argument("--temporal", action="store_true", help="时序 R3D 管线(需 models/r3d_<stem>_w64.pt)")
    pr.add_argument("--temporal-ckpt", default=None, help="时序 ckpt 路径覆盖(默认按 stem 查找)"),

    cl = sub.add_parser("calibrate", help="人工点选球场标定")
    cl.add_argument("video")
    cl.add_argument("--frame", type=float, default=500.0, help="抽帧时刻(秒)")
    cl.add_argument("--out", default="outputs")

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


def _rally_rows(rallies: list[Rally]) -> list[dict]:
    return [
        {"id": i, "start_s": round(r.start, 2), "end_s": round(r.end, 2),
         "duration_s": round(r.end - r.start, 2), "hits": r.hits,
         "motion_peak": round(r.motion_peak, 2), "confidence": round(r.confidence, 3)}
        for i, r in enumerate(rallies, start=1)
    ]


def process_one(video: str, out_root: str, args) -> int:
    meta = probe(video)
    stem = Path(video).stem
    outdir = Path(out_root) / stem
    work = Path("temp/work") / stem
    outdir.mkdir(parents=True, exist_ok=True)
    persons_path = outdir / "persons.jsonl"
    cache_path = outdir / "cache.json"
    cache_key = {"video": video, "mtime": os.path.getmtime(video),
                 "fps": 5.0, "width": 1280}
    if args.temporal:
        return _process_temporal(video, meta, args, outdir, work)
    if args.flow:
        return _process_flow(video, meta, args, outdir, work, cache_path, cache_key)
    if args.pose:
        return _process_pose(video, meta, args, outdir, work, cache_path, cache_key)
    cached = persons_path.exists() and cache_path.exists() and \
        json.loads(cache_path.read_text()) == cache_key
    if cached:
        pmeta, rows = load_persons_jsonl(str(persons_path))
        print(f"[cache] 复用检测缓存 {persons_path}")
    else:
        frames = extract_frames(video, str(work / "frames"), fps=5.0, width=1280)
        rows = detect_persons(frames, frame_fps=5.0, device=args.device,
                              out_jsonl=str(persons_path))
        pmeta, rows = load_persons_jsonl(str(persons_path))
        cache_path.write_text(json.dumps(cache_key))
    frame_h = float(pmeta["frame_h"])

    roi = _parse_roi(args.roi)
    if roi is None:
        roi = auto_roi(rows, float(pmeta["frame_w"]), frame_h)
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
        "rallies": _rally_rows(rallies),
    }
    (outdir / "rallies.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    total = sum(r.end - r.start for r in rallies)
    print(f"[summary] {stem}: {meta.duration_s:.0f}s → {len(rallies)} 个回合, "
          f"共 {total:.0f}s ({total / meta.duration_s * 100:.0f}% 保留)")
    return 0


def _process_pose(video: str, meta: VideoMeta, args, outdir: Path, work: Path,
                  cache_path: Path, cache_key: dict) -> int:
    """--pose 管线:帧→poses(缓存)→标定→特征→ARMED→精修→导出。"""
    stem = Path(video).stem
    court_path = outdir / "court.json"
    if not court_path.exists():
        print(f"缺少球场标定 {court_path},请先运行 shuttlecut calibrate <video>")
        return 1
    cal = load_cal(court_path)

    poses_path = outdir / "poses.jsonl"
    pose_key = cache_key | {"mode": "pose"}
    cached = poses_path.exists() and cache_path.exists() and \
        json.loads(cache_path.read_text()) == pose_key
    if cached:
        poses = load_poses_jsonl(str(poses_path))
        print(f"[cache] 复用姿态缓存 {poses_path}")
    else:
        frames = extract_frames(video, str(work / "frames"), fps=5.0, width=1280)
        poses = estimate_poses(frames, out_jsonl=str(poses_path))
        cache_path.write_text(json.dumps(pose_key))

    features = frame_features(poses, cal)
    transients: list[Transient] = []
    if not args.no_audio_refine:
        try:
            wav = extract_audio(video, str(work / "audio.wav"))
            transients = audio_transients(wav)
        except Exception as e:  # 无音轨/解码失败 → 降级纯视觉
            print(f"[warn] 音频精修跳过: {e}")
    params = ArmedParams()
    rallies: list[Rally] = segment_armed(features, [tr.t for tr in transients], params)
    if transients:
        rallies = refine(rallies, transients)

    seg = SegParams()
    clips = export_clips(video, rallies, str(outdir / "clips"),
                         pre_s=seg.pre_roll_s, post_s=seg.post_roll_s)
    if not args.no_reel and clips:
        export_reel(clips, str(outdir / "clips" / "highlights.mp4"))

    payload = {
        "video": stem,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "params": {"mode": "pose"} | vars(params),
        "rallies": _rally_rows(rallies),
    }
    (outdir / "rallies.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    total = sum(r.end - r.start for r in rallies)
    print(f"[summary] {stem}: {meta.duration_s:.0f}s → {len(rallies)} 个回合, "
          f"共 {total:.0f}s ({total / meta.duration_s * 100:.0f}% 保留)")
    return 0



def _process_temporal(video: str, meta: VideoMeta, args, outdir: Path, work: Path) -> int:
    """--temporal:R3D 时序管线(帧缓存→补偿差分→概率曲线→两尺度切分→导出)。"""
    from shuttlecut.segmenter import Rally, SegParams
    from shuttlecut.temporal import extract_frames15, infer_curve, two_scale_segments

    stem = Path(video).stem
    ckpt = getattr(args, "temporal_ckpt", None) or f"models/r3d_{stem}_w64.pt"
    if not Path(ckpt).exists():
        print(f"[error] 未找到时序模型 {ckpt};请先用 tools/cuda/train_heavy.py 训练,"
              f"或用 --temporal-ckpt 指定路径")
        return 1
    fine_ckpt = f"models/r3d_{stem}_w24.pt"

    frames = extract_frames15(video, str(work / "frames15"))
    print(f"[temporal] {len(frames)} 帧 @15fps,模型 {ckpt}")
    centers, probs = infer_curve(frames, ckpt, device=args.device)

    fine = (infer_curve(frames, fine_ckpt, win=24, tsub=2, device=args.device)
            if Path(fine_ckpt).exists() else (None, None))

    segs = two_scale_segments(centers, probs, fine[0], fine[1])
    rallies = [Rally(start=a, end=b, motion_peak=float(probs.max()), confidence=1.0)
               for a, b in segs]
    seg = SegParams()
    clips = export_clips(video, rallies, str(outdir / "clips"),
                         pre_s=seg.pre_roll_s, post_s=seg.post_roll_s)
    if not args.no_reel and clips:
        export_reel(clips, str(outdir / "clips" / "highlights.mp4"))

    payload = {
        "video": stem,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "params": {"mode": "temporal", "ckpt": ckpt},
        "rallies": [{"id": i + 1, "start_s": round(r.start, 2), "end_s": round(r.end, 2)}
                    for i, r in enumerate(rallies)],
    }
    (outdir / "rallies.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    total = sum(r.end - r.start for r in rallies)
    print(f"[summary] {stem}: {meta.duration_s:.0f}s → {len(rallies)} 个回合, "
          f"共 {total:.0f}s ({total / meta.duration_s * 100:.0f}% 保留)")
    return 0


def _process_flow(video: str, meta: VideoMeta, args, outdir: Path, work: Path,
                  cache_path: Path, cache_key: dict) -> int:
    """--flow 管线:15fps 光流残差→事件→ARMED→精修→导出。"""
    from shuttlecut.flowpipe import load_flow, run_flow

    flow_path = outdir / "flow.jsonl"
    flow_key = cache_key | {"mode": "flow", "fps": 15.0, "width": 960}
    cached = flow_path.exists() and cache_path.exists() and \
        json.loads(cache_path.read_text()) == flow_key
    if cached:
        rows = load_flow(str(flow_path))
        print(f"[cache] 复用光流缓存 {flow_path}")
    else:
        rows = run_flow(video, str(flow_path), fps=15.0, width=960, device=args.device)
        cache_path.write_text(json.dumps(flow_key))

    # 适配 segment_armed 输入:wrist_peak 字段复用为光流残差(内部做 median/MAD 归一);
    # n_by_side 用在场大人数克隘(≥2 人时两侧各计 min(n,2),门控等价于"场上至少双人")
    features = [
        {"t": r["t"], "n_by_side": (min(r["n_persons"], 2),) * 2 if r["n_persons"] else (0, 0),
         "wrist_peak": r["residual"], "any_ready": True}
        for r in rows
    ]

    transients: list[Transient] = []
    if not args.no_audio_refine:
        try:
            wav = extract_audio(video, str(work / "audio.wav"))
            transients = audio_transients(wav)
        except Exception as e:
            print(f"[warn] 音频精修跳过: {e}")
    params = ArmedParams()
    rallies: list[Rally] = segment_armed(features, [tr.t for tr in transients], params)
    if transients:
        rallies = refine(rallies, transients)

    seg = SegParams()
    clips = export_clips(video, rallies, str(outdir / "clips"),
                         pre_s=seg.pre_roll_s, post_s=seg.post_roll_s)
    if not args.no_reel and clips:
        export_reel(clips, str(outdir / "clips" / "highlights.mp4"))

    payload = {
        "video": Path(video).stem,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "params": {"mode": "flow"} | vars(params),
        "rallies": _rally_rows(rallies),
    }
    (outdir / "rallies.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    total = sum(r.end - r.start for r in rallies)
    print(f"[summary] {Path(video).stem}: {meta.duration_s:.0f}s → {len(rallies)} 个回合, "
          f"共 {total:.0f}s ({total / meta.duration_s * 100:.0f}% 保留)")
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
    if args.cmd == "calibrate":
        stem = Path(args.video).stem
        meta = probe(args.video)
        if args.frame >= meta.duration_s:
            print(f"[error] 无法在 {args.frame:g}s 处抽帧(视频时长仅 {meta.duration_s:.0f}s)")
            return 1
        try:
            frames = extract_frames(args.video, str(Path("temp/calib") / stem),
                                    fps=5.0, width=1280,
                                    t_start=args.frame, t_end=args.frame + 0.2)
        except subprocess.CalledProcessError:
            print(f"[error] 无法在 {args.frame:g}s 处抽帧")
            return 1
        if not frames:
            print(f"[error] 无法在 {args.frame:g}s 处抽帧(视频可能更短)")
            return 1
        outdir = Path(args.out) / stem
        outdir.mkdir(parents=True, exist_ok=True)
        pick_court(frames[0], str(outdir / "court.json"))
        print(f"标定已保存 → {outdir / 'court.json'}")
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
