# ShuttleCut 多模态 LLM 操作手册

> 本文档面向**由 LLM/Agent 操作本工具**的场景:如何选择流程、配置参数、执行视觉标注、提交结果、处理异常。人类用户只需读 README;Agent 请通读本文。

## 0. 能力速查与决策表

| 场景 | 命令 | 说明 |
|---|---|---|
| 常规出片(同场馆/训练分布内) | `process` | 零参数;自动选模型 |
| 新场馆/质量敏感视频 | `calibrate` → `process` | Agent 做 5 分钟视觉标注,实测 P/R 0.9-1.0 |
| 评测切分质量 | `process --write-metadata` → `eval` | 需要真值 JSON(data/ground_truth/) |
| 训练新模型 | `tools/cuda/train_heavy.py` | 产出 → models/exp/;晋升 → models/shuttlecut.pt |

## 1. process(核心出片)

```bash
shuttlecut process <INPUT>... [-o DIR] [--model CKPT[,CKPT...]] [--device D]
                   [--overwrite] [--write-metadata] [--quiet]
```

**模型自动解析(不要手动指定,除非明确需要多模型投票)**:
1. `models/shuttlecut-<stem>.pt`(该视频校准模型,calibrate 产物,最优先)
2. `models/shuttlecut.pt`(官方生产模型)

**输出契约(恰好 2 个文件,不得假设有其他)**:
- `<out-dir>/<stem>-all-rallies.mp4` — 全部回合,时间顺序
- `<out-dir>/<stem>-highlights.mp4` — 精选回合,精彩度降序

**流契约**:
- stderr:进度(`配置: …` + `[1/5]`~`[5/5]` 阶段 + 百分比/计数)。**长时间无 stdout 是正常的,看 stderr。**
- stdout:最终摘要(2 个输出路径 + 统计 + 最精彩回合)。**不要在 stdout 里 grep 进度。**
- 退出码:0 成功 / 1 失败(模型缺失、输出已存在、未检出回合) / 2 用法错误

**Agent 行为规则**:
- 输出已存在(exit 1, stderr 含 "用 --overwrite 覆盖"):确认用户意图后加 `--overwrite` 重跑
- "未找到时序模型":先查 `models/` 目录;无模型则告知用户需放置 `models/shuttlecut.pt` 或先 calibrate(见 §2)
- "未检出任何回合":视频可能无对打内容,或模型不适配该场馆 → 走 calibrate 流程
- 首次处理某视频较慢(抽帧+差分缓存构建);重复处理显著加快(缓存命中)
- 音频缺失只降级评分,不算失败(stderr 出现 `[warn] 音频特征不可用`)

## 2. calibrate(校准协议 —— Agent 的主要标注任务)

### 2.1 Phase 1: prepare
```bash
shuttlecut calibrate <INPUT> [--strips 40]
```
产出(固定路径):
- `shuttlecut-output/<stem>/calib/strip_00.jpg … strip_NN.jpg` — 6 帧拼条带(左上→右下时间递增,约 3s 间隔,底部标注无)
- `shuttlecut-output/<stem>/calib/calib_template.json` — 标注模板

### 2.2 视觉标注(Agent 的核心工作)
对**每一张** strip,逐帧判断**近场**(画面主体球场)是否"回合中":

| 判定 | 记号 | 视觉标准 |
|---|---|---|
| 回合中 | `Y` | 双方对打中 / 发球瞬间 / 击球后移动取位(仍属该回合) |
| 停顿 | `N` | 走动捡球 / 站立休息 / 换场 / 仅远场有人打 / 空场 |

输出恰好 6 个字符(与 6 帧一一对应),如 `"YYNNYY"`。
**多球场注意**:只判近场(最大/最清晰那片);远场打球 = `N`。
**不确定帧**:宁可选回合状态倾向(发球准备姿态=Y)。

### 2.3 提交标注
把模板每条 `verdict` 字段改好后,**完整 JSON** 保存为:
```
shuttlecut-output/<stem>/calib/calib.json
```
模板结构(不得增删字段,只改 verdict):
```json
{"video": "<stem>", "strips": [{"strip": 0, "times": [12.3, 15.3, "..."], "verdict": "YYNNYY"}, ...]}
```

### 2.4 Phase 2: run
```bash
shuttlecut calibrate <INPUT> --phase run --calib shuttlecut-output/<stem>/calib/calib.json
```
- 产出 `models/shuttlecut-<stem>.pt`(之后 process 自动优先使用,**无需 --model**)
- 需要 GPU;约 5-15 分钟(取决于视频长度);stderr 有训练日志
- 之后直接 `shuttlecut process <INPUT>` 出片

### 2.5 校准质量守则
- 40 条带是推荐密度;视频 <5 分钟可 `--strips 24`,>25 分钟可 `--strips 60`
- 标注一致性 > 覆盖率:宁可全部判对,不要猜测模糊帧
- 单张条带 6 帧全 `N` 完全合法(纯休息段)

## 3. eval / label(开发回路)

```bash
shuttlecut eval <rallies.json> --gt data/ground_truth/<stem>.json [--record file.md]
```
- rallies.json 由 `process --write-metadata` 产出
- 输出 PASS/FAIL(P/R ≥0.90)及 missed/extra/fragment/boundary 分解
- Agent 优化回路:calibrate → process --write-metadata → eval → 未达 0.9 则检查标注质量(尤其边界帧)重标 → 重跑

## 4. 训练与晋升(需要时)

```bash
python tools/cuda/train_heavy.py --frames <dir>[,<dir>...] --gt <json>[,...] \
    --out models/exp/<tag>.pt --win 64 --epochs 6 --device cuda --seed 13
```
- 实验模型一律 `models/exp/`;评审(curves.py + segment_eval.py)后 `Copy-Item` 晋升为 `models/shuttlecut.pt`
- 详见 tools/cuda/README.md 与 docs/eval-history.md(19 路线实验档案,勿重复已判负路线)

## 5. 异常处置速查

| 症状 | 原因 | Agent 动作 |
|---|---|---|
| exit 1 "输出已存在" | 覆写保护 | 征得同意后 `--overwrite` |
| exit 1 "未找到时序模型" | models/ 空 | 引导放置模型或走 calibrate |
| exit 1 "未检出任何回合" | 内容不符/域差距 | calibrate;仍无则报告用户 |
| ffmpeg 报错 | 源损坏/无音轨 | 无音轨可忽略(warn);源损坏报告用户 |
| 进度长时间停在 [2/5] | 大视频推理 | 正常;按 stderr 百分比等待 |
| GPU OOM | 显存不足 | `--device cpu`(慢)或减小视频 |
