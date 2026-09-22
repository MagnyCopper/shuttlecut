"""生产探针训练:全部 19 GT 视频 × 3 种子 → models/shuttlecut-probe.pt。

产物结构:{"probes": [state_dict×3], "seg": (sm,hi,lo,ml,mg), "meta": {...}}
分割参数在全量训练曲线上网格选出;推理侧另加斜率 snapping(vjepa.py)。
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from loeo_probe import (VENUES, build_probe, curve, grid_on, load_stem,  # noqa: E402
                        seg_apply, train_probe)

OUT = "models/shuttlecut-probe.pt"
SEEDS = [13, 21, 42]


def main() -> None:
    stems = [s for ss in VENUES.values() for s in ss]
    vids = [v for v in (load_stem(s) for s in stems) if v]
    print(f"训练视频: {len(vids)}/{len(stems)}, 窗数合计 {sum(len(v['feat']) for v in vids)}")
    curves = {v["stem"]: [] for v in vids}
    states = []
    for seed in SEEDS:
        m = build_probe(seed=seed).to("cuda")
        train_probe(m, vids, seed, dev="cuda")
        states.append({k: t.cpu() for k, t in m.state_dict().items()})
        for v in vids:
            curves[v["stem"]].append(curve(m, v["feat"], "cuda"))
        print(f"seed {seed} 训练完成", flush=True)
    ens = {s: np.mean(cs, axis=0) for s, cs in curves.items()}
    score, par = grid_on([(v["cent"], ens[v["stem"]], v["segs"]) for v in vids])
    print(f"全量分割参数={par} 训练侧 min(P,R)={score:.3f}")
    torch.save({
        "probes": states,
        "seg": list(par),
        "meta": {
            "feat_model": "facebook/vjepa2-vitl-fpc16-256-ssv2",
            "win": 16, "stride": 8, "fps": 15, "dim": 1024,
            "d": 256, "layers": 3, "norm": "per-video zscore",
            "snap_win_s": 1.5, "seeds": SEEDS, "trained_on": stems,
        },
    }, OUT)
    print(f"已保存 → {OUT} ({Path(OUT).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
