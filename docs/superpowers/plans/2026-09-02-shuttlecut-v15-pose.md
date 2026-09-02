# ShuttleCut v1.5 实施计划附加(姿态判别器)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development。步骤用 checkbox 跟踪。
> 上游:2026-09-01-shuttlecut-v1.md(Task 1-15);v1 验收失败后经 Oracle 裁决(见 .superpowers/oracle-verdict.md)与用户确认,升级观测能力为 RTMPose 姿态路线。

**Goal:** 用姿态信号(手腕动作峰/准备姿态/半场归属)替代位移能量,达到段级 R/P≥0.90(或以实测上限如实交付)。

**Architecture:** poses 运行时(rtmlib,COCO 17 点)→ 球场单应(人工四角+网线标定,缓存复用)→ 球员筛选(场内+深度)→ 特征(腕速峰、姿态复位、半场占用)→ ARMED 状态机 + 音视共现 → 同一 eval 验收。

**Tech Stack:** rtmlib(onnxruntime CPU,0.06s/帧,模型在 `models/rtmlib/`,经 TORCH_HOME 指定)、OpenCV(numpy 单应)、既有管线复用。

## Global Constraints

- 同 v1 计划(工程内 venv/models/temp、阻塞子任务、结构化提问、中文 conventional commits)
- TORCH_HOME 必须在导入 rtmlib 前设为 `models/rtmlib`(规约 3)
- 姿态缓存 `outputs/<stem>/poses.jsonl`(键含视频 mtime),迭代调参不得重复推理
- 调参只用 B1,B2 只做验证(Oracle 交叉验证要求)

---

### Task 16: poses 模块(rtmlib 封装+缓存)

**Files:** Create `src/shuttlecut/poses.py`、`tests/test_poses.py`
**Interfaces:** `estimate_poses(frames: list[str], out_jsonl: str | None = None, mode='lightweight') -> list[FramePose]`;`FramePose(t: float, persons: list[PersonKps])`;`PersonKps(kps: np.ndarray (17,2), score: float)`;jsonl 行 {"t","persons":[{"kps":[[x,y]...],"score":...}]};`load_poses_jsonl(path) -> list[FramePose]`

- [ ] Step 1: 失败测试:合成帧(synth_video 抽 3 帧)上 estimate_poses 返回 3 个 FramePose,字段结构正确;jsonl roundtrip 一致(注意 kps 转 list)
- [ ] Step 2: 确认失败 → Step 3: 实现(模块级懒加载 Body,导前 os.environ['TORCH_HOME']='models/rtmlib';帧按序 t=i/5)
- [ ] Step 4: 通过 → Step 5: commit `feat: poses 模块(rtmlib 姿态+jsonl 缓存)`

### Task 17: 球场标定(人工点选+单应+缓存)

**Files:** Create `src/shuttlecut/court.py`、`tests/test_court.py`
**Interfaces:** `pick_court(frame_path: str, out_path: str) -> CourtCal`(matplotlib 交互点选 4 角[左上,右上,右下,左下]+网线中点,共 5 点;窗口标题提示顺序);`CourtCal(corners: list[tuple], net_mid: tuple)`;`save_cal/load_cal(cal, path)`(json);`to_court_xy(cal, x, y) -> tuple[float, float]`(单应映射到标准半场坐标:0..3.05(宽)×0..13.4(长),以左底角为原点);`net_line(cal) -> callable`(球场坐标中 y=6.7 为网,提供 side_of(cal, x, y) -> 0/1)。`net_mid` 仅作标定质量参考，不参与映射。

- [ ] Step 1: 失败测试:纯几何——用合成 cal(无畸变矩形)验证 to_court_xy 角点映射为 (0,0)(3.05,0)(3.05,13.4)(0,13.4)、side_of 网两侧返回不同;load/save roundtrip
- [ ] Step 2-4: TDD 实现(单应用 cv2.getPerspectiveTransform,标准坐标 X∈[0,3.05] Y∈[0,13.4])
- [ ] Step 5: 真实帧人工验证:对 B1 首帧跑 pick_court 由控制器点选,保存 `outputs/<stem>/court.json`,检查映射合理性(球网中点 y≈6.7)——此为人工关卡,控制器执行
- [ ] Step 6: commit `feat: 球场标定(四角+网线点选/单应/缓存)`

### Task 18: 姿态特征(腕速峰/姿态复位/半场占用)

**Files:** Create `src/shuttlecut/posefeat.py`、`tests/test_posefeat.py`
**Interfaces:**
- `court_players(poses, cal) -> list[FramePlayers]`(踝/髋中点映射入场内 0.15m 缓冲者,按 y 深度取最近网的 2-4 人;FramePlayers(t, players: list[tuple[kps, side])])
- `wrist_speed(poses_seq) -> list[float]`(每人右/左腕(kps 9/10)帧间速度,取帧内最大者,单位 px/s)
- `stance_ready(player) -> bool`(髋(11,12)低于膝(13,14)连线中点一定比例且踝 y 差 < 阈值 → 屈膝准备;阈值以比例表达避免绝对像素)
- `court_players(poses, cal) -> list[FramePlayers]`(按立足点场内过滤并取离网最近 4 人)
- `frame_features(poses, cal) -> list[dict]`(t, n_by_side, wrist_peak, any_ready; 跨帧按立足点最近邻匹配)——状态机输入

- [ ] Step 1: 失败测试:合成 PersonKps 构造站位/挥拍/走动三场景,断言 wrist_speed 挥拍>站位、stance_ready 站位 True 走动 False、court_players 场内深度筛选正确
- [ ] Step 2-4: TDD 实现
- [ ] Step 5: commit `feat: 姿态特征(腕速/准备姿态/半场占用/场内筛选)`

### Task 19: ARMED 状态机(音视共现+半场交替)

**Files:** Create `src/shuttlecut/armed.py`、`tests/test_armed.py`;Modify `cli.py`(process 增加 --pose 管线分支与 poses/court 缓存)
**Interfaces:** `segment_armed(features: list[dict], transients: list[float], params=ArmedParams()) -> list[Rally]`
`ArmedParams(arm_s=1.0, confirm_s=3.0, wrist_mad=3.0, wrist_mad_audio=1.5, end_gap_s=(0.4,1.2), min_rally_s=1.5, min_idle_s=1.2)`
逻辑(Oracle 设计):
- IDLE→ARMED:场地有效且每侧≥1 球员持续 arm_s
- ARMED→RALLY:腕速峰 ≥ wrist_mad×MAD(有 ±0.25s 音频瞬态时降为 wrist_mad_audio×MAD);confirm_s 内出现对侧事件则确认,否则回 ARMED
- RALLY 期间:事件半场交替维持;结束=最后事件后 end_gap_s 内无新事件且有姿态复位/连续同侧漂移 → 终点=最后事件+0.3~0.8s 回溯
- 不做固定间隙合并;min_rally_s=1.5(容纳短回合),min_idle_s=1.2

- [ ] Step 1: 失败测试:合成 features+transients 构造 IDLE/ARMED/RALLY/短回合/邻场音频污染五场景(邻场=音频峰但无腕速峰→不得进入)
- [ ] Step 2-4: TDD 实现
- [ ] Step 5: cli 接线:--pose 模式跑 poses(缓存)→标定(缺失时提示先运行 `shuttlecut calibrate <video>`,新增子命令)→特征→ARMED→refiner(边界微调保留)→导出
- [ ] Step 6: 全量测试 + commit `feat: ARMED 状态机与 --pose 管线`

### Task 20: 调参验收(B1 调参/B2 验证,交叉)

- [ ] Step 1: B1 跑 --pose 全管线 + eval;按误差分类迭代参数(只动一类/轮,记录 eval_history.md)
- [ ] Step 2: B1 达标后冻结参数,B2 只验证;再反向(B2 调参冻结,B1 验证)取交集参数
- [ ] Step 3: 双双 ≥0.90 → 走 Task 14-15 收尾;任一 <0.90 → 如实报告实测上限,带完整 eval_history 向用户结构化提问(选项:接受实测值交付 / 继续投入)
- [ ] Step 4: 更新 README/设计文档,commit
