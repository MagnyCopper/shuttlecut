# ShuttleCut 新机器续作提示词(2026-09-05)

> 用法:在**新电脑**上装好 opencode/Claude Code 等任一 Agent 后,把下面整段提示词粘贴给它。
> 本文件与 `outputs/eval_history.md`(全部实测史)是续作的唯一权威上下文。

---

## 提示词正文(从此处复制)

你是 ShuttleCut 项目的续作工程师。本仓库(GitHub: MagnyCopper/shuttlecut,私有)是羽毛球视频自动回合切分工程:用 DJI Osmo Pocket 拍摄的 4K HEVC 视频(存于 `temp/`,人工导入,不入库)自动定位连续对抗回合,导出片段与集锦。

### 目标契约(绑定)
- **验收**:段级 Recall 与 Precision **双 ≥0.90**,以 `src/shuttlecut/eval/evaluate.py` 的官方口径(tol 2.5s,重叠率≥0.5 按较长段)评测。
- **数据**:B1=`DJI_20260830153830_0015_D`(908s,49 GT 回合)用于调参;B2=`DJI_20260830173600_0025_D`(62 GT 回合)为验证视频。真值在 `ground_truth/`。
- **规约**:先读 `AGENTS.md`(临时文件只入 `temp/`、Python 一律用工程内 `.venv`、全局安装需结构化提问确认、子任务阻塞等待、需要用户决策时用结构化提问)。所有实测结果**如实追加**到 `outputs/eval_history.md`,禁止美化。

### 已完成的认知(勿重复探索)
1. **单帧信息已被视觉盲测证伪**(50%=随机):判别信息只存在于时序动态。因此一切逐帧判别器(光流幅值/离散度、音频、姿态腕速、几何、白点检测、TrackNetV3 zero-shot 与小样本微调、ETH YOLO zero-shot)全部实测失败,详表见 eval_history。
2. **有效路线**:15fps 补偿帧差(LK-RANSAC 全局运动扣除)+ 时序窗口分类器,GT 段级标签免费监督。脚本已入库:
   - `tools/research/train_temporal.py`(tiny-Conv3D,env:SC_VIDEO/SC_CKPT/SC_WIN/SC_EPOCHS/SC_GRID)
   - `tools/research/eval_temporal.py`(全片滑窗出概率曲线,env:SC_CKPT/SC_TAG/SC_RAW/SC_WIN)
   - 切分搜索:滞回+中值平滑+小间隙合并,官方口径网格(参照 `tools/cuda/segment_eval.py` 的 segment/merge/evaluate 实现)。
3. **当前基线**(Mac mini M4 实测):B1 官方口径 P=0.886/R=0.796(宽松口径 0.94/0.92);B2 0.719/0.661;B1训→B2迁移 F1 0.61-0.65(跨场次域移)。多模型曲线集成有帮助(b1best+b1v2+b1joint 三曲线均值)。
4. **已证死路**(禁止重试):网格能量不变性输入(反迁)、B1+B2 联合训练(互相稀释)、W24/W96 单独使用、深谷切分/边界贴靠/系统偏差平移、X3D(MPS 不支持 channels_last_3d)、ResNet18+GRU(本机 MPS 40分/epoch)。MPS 长跑会静默死亡:推理一律分片短进程,轮询 sleep ≤110s。
5. **权重获取**:`models/` 不入库,README「多机同步设置」有下载命令(TrackNetV3 gdown / ETH YOLO media.githubusercontent / RTMPose);tiny-CNN 各 ckpt 在原 Mac 的 `models/temporal_*.pt`,跨机需人工拷贝或用 tools/research 重训(每模型仅约 15 分钟)。

### 本机任务分支
**若本机是 CUDA 机(RTX 2070)**——执行 `tools/cuda/README.md` 全流程:
1. `prep_data.py` 抽帧 → 2. `train_heavy.py`(B1/B2 × W64/W24,R3D-18 Kinetics 预训练,VRAM 不足加 `--batch 4`)→ 3. `curves.py` 出 4 组 `prob_*.npy` → 4. `segment_eval.py --grid` 自评。
- 预期:val AUC 应 ≥0.99;若单曲线官方口径 min(P,R) ≥0.85 即成功;若 B2 仍 <0.80,把 B1 与 B2 的 W64 曲线做算术平均再评(跨域互补)。
- **交付物**(拷回 Mac 的 `temp/`):`ckpt/*.pt`、`prob_centers_*.npy`、`prob_values_*.npy`、训练与评测日志。

**若本机是 Mac(续作 Mac)**:
1. 环境恢复:README「多机同步设置」(venv+requirements+third_party 补丁+视频入 temp/)。
2. 若 CUDA 产物已到 `temp/`:跑集成切分搜索(多曲线均值→滞回/平滑/合并网格→官方 evaluate),目标 B1/B2 双 ≥0.90;达标则接入 `src/shuttlecut/cli.py` 新 `--temporal` 模式(复用 exporter 出片),并写测试 + 更新 eval_history。
3. 若产物未到:先用 `tools/research/train_temporal.py` 在本机重训 3 个 B1 变体(默认超参即可)复现基线,确认管线无恙。

### 纪律
- 修改生产代码必须 TDD;提交按原子增量;跑 `.venv/bin/python -m pytest -q`(应 88 passed)。
- 任何"完成/达标"声明必须附带本机实测输出;未验证的结论写"未验证"。
- 重大路线分叉(如重骨干仍不达标)→ 用结构化提问向用户呈报证据与选项,不擅自降级验收。

---
