"""跨场馆主战场:训练=DJI-14,测试=B站场馆(逐一,特征就绪即测)。

对应档案"零训练跨场馆 0.1-0.5 抽签"的原始场景——直接可比。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from loeo_probe import VENUES, ensemble_curves, grid_on, load_stem, seg_eval  # noqa: E402

BILI = ["linzhou", "baoganghui_p1", "baoganghui_p2", "jiguang", "yygq"]
SEEDS = [13, 21, 42]


def main() -> None:
    tr = [v for v in (load_stem(s) for s in VENUES["dji"]) if v]
    assert len(tr) == 14, f"DJI-14 不齐: {len(tr)}"
    te = [v for v in (load_stem(s) for s in BILI) if v]
    if not te:
        print("[wait] 无 B站 特征")
        return
    print(f"训练=DJI-{len(tr)} 测试={[v['stem'] for v in te]}")
    probs = ensemble_curves(tr + te, SEEDS, "cuda")
    score, par = grid_on([(v["cent"], probs[v["stem"]], v["segs"]) for v in tr])
    print(f"分割参数={par} 训练侧={score:.3f}")
    for v in te:
        p, r, nd, ng = seg_eval(par, v["cent"], probs[v["stem"]], v["segs"])
        print(f"[dj2bili] {v['stem']}: P={p:.3f} R={r:.3f} 段={nd}/{ng}", flush=True)


if __name__ == "__main__":
    main()
