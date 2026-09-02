# ShuttleCut v1.6 实施计划附加(光流判别器)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development。
> 上游:v1.5 姿态判别器 5fps 下失败(见 outputs/eval_history.md);用户确认按 Oracle 原设计上 15fps 局部光流。

**Goal:** 球员框内光流残差(扣全局相机运动)→ 动作事件 → 复用 ARMED 状态机,目标 R/P≥0.90(如实报告实测上限)。

**Architecture:** 15fps@960 灰度帧 → YOLO persons(15fps,MPS)→ 每个球员 bbox 内 Farneback 光流均值 − 全局 LK-RANSAC 平移幅值 = 残差动作 → 逐视频 median/MAD 归一 → 事件(z≥k,音频±0.25s 共现降 k)→ segment_armed(适配 features)→ 既有 eval/导出。

**Tech Stack:** cv2.calcOpticalFlowFarneback、cv2.calcOpticalFlowPyrLK+estimateAffinePartial2D(RANSAC)、既有管线。

## Global Constraints

- 同前(工程内依赖/阻塞子任务/结构化提问/中文提交)
- 中间帧与检测结果全部落 temp/ 或 outputs/ 缓存;15fps 帧目录 `temp/work/<stem>/frames15/`
- 流缓存 `outputs/<stem>/flow.jsonl`(键含视频 mtime、fps=15、width=960)
- 调参只用 B1,B2 验证

---

### Task 21: flowfeat 核心残差计算(纯函数,TDD)

**Files:** Create `src/shuttlecut/flowfeat.py`、`tests/test_flowfeat.py`
**Interfaces:**
- `global_shift(prev_gray, cur_gray) -> tuple[float, float]`:goodFeaturesToTrack(整帧)→ PyrLK → RANSAC(estimateAffinePartial2D)→ 平移分量;特征不足返回 (0,0)
- `local_flow_mag(prev_gray, cur_gray, bbox: tuple[int,int,int,int]) -> float`:bbox 裁剪内 Farneback 光流幅值均值
- `residual_action(prev_gray, cur_gray, bboxes: list[tuple]) -> float`:max(local_flow_mag − ‖global_shift‖ 的范数×0.8, 0) 的最大者(全裁剪)
- [ ] TDD:合成图(平移纹理=全局运动应被扣除;框内局部运动应保留;静态=0)三场景断言
- [ ] commit `feat: flowfeat 光流残差核心`

### Task 22: 光流管线运行器(15fps+缓存)

**Files:** Create `src/shuttlecut/flowpipe.py`、`tests/test_flowpipe.py`
**Interfaces:**
- `run_flow(video: str, out_jsonl: str, fps: float = 15.0, width: int = 960, device: str = 'auto') -> list[dict]`
  流程:extract_frames(fps,width) → detect_persons(全帧,大目标 h≥10% 帧高,取每帧前 4 大) → 逐帧灰度化(cv2.imread IMREAD_GRAYSCALE)→ residual_action → 行 {"t","residual","n_persons"};缓存放 outputs
- `load_flow(path) -> list[dict]`
- [ ] TDD:synth_video 上端到端(t 步进 1/15、行 schema);残差有限非负
- [ ] commit `feat: flowpipe 15fps 光流管线`

### Task 23: --flow 模式接线与验收

**Files:** Modify `src/shuttlecut/cli.py`(+`--flow` 开关:flow 缓存→特征适配→segment_armed→refine→导出);`src/shuttlecut/armed.py` 若需(允许 features 无 n_by_side 时退化为人数门控 ≥2)
**验收:** B1 调参(k_mad∈{3,2.5,2}, end_gap 微调,一类一轮)→ 冻结 → B2 验证 → eval_history 记录;≥0.90 走收尾,<0.90 如实呈报实测上限与选项。
