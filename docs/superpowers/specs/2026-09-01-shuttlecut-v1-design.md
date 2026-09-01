# ShuttleCut v1 设计文档

日期:2026-09-01
状态:已确认(基于需求澄清 + 开源调研 + 三项技术验证)

## 1. 背景与目标

ShuttleCut 是一个本地 CLI 工具,用于自动剪辑 DJI Osmo Pocket 拍摄的羽毛球视频:

- **核心产出**:按回合切片的视频片段(`rally_NNN.mp4`)+ 可选合并集锦(`highlights.mp4`)
- **核心指标**:回合切分准确率 ≥ 90%(召回与精确率均需达标),片段前后留拍合理,可直接使用
- **降级后置**:正式比赛比分统计(v2+ 半自动路线)、精彩度自动评分(规则后调)

### 1.1 素材特性(决定技术选型的硬约束)

| 特性 | 实测结论 |
|---|---|
| 分辨率/编码 | 3840×2160 HEVC,单段 15-19 分钟 |
| 机位 | 场边角落低机位(0.7-1m)斜拍,**拍摄中会调整朝向** |
| 场地覆盖 | 无法稳定覆盖全部边界线,双方球员通常同框 |
| 遮挡 | 场边人员走动偶发大面积遮挡 |
| 记分牌 | 无(仅场地编号牌),比分只能靠回合结果推断 |
| 音频 | 内录音轨可用,但**多球场环境邻场击球串扰严重** |

### 1.2 运行环境

- Apple M4(10 核 GPU / 16GB 统一内存),macOS
- 工程内 `.venv/`(Python 3.12),依赖与模型全部落在工程目录内(见 `AGENTS.md` 规约)

## 2. 调研与技术验证结论

### 2.1 关键参考项目

| 项目 | 用途 |
|---|---|
| Good-Badminton(935★, Apache-2.0) | 球检测权重(yolo11s-ball.pt)、球场检测组件可复用;其回合检测基于"球场视图切换"模板匹配,**不适用本素材**(画面恒为球场) |
| Breakpoint(网球,开源) | "音频击球检测 + 静默间隙切分 + 视觉排名 + ffmpeg 导出"架构先例 |
| badminton-pipeline-repro | Apple Silicon 全流程经验:TrackNet 阈值需 0.15、TrackNet 无 MPS(约 3h/部,CPU) |
| TrackNetV3(198★) | v2 升级路径的球追踪方案 |
| kwyoke/Badminton-hit-detection | 业余机位泛化经验 + 业余标注数据集 |

### 2.2 技术验证记录(素材:temp/ 两段真实视频,产物存 temp/validation/)

| 验证项 | 结论 | 关键证据 |
|---|---|---|
| 音频击球瞬态 | 可检,但**不可独立分割** | 强瞬态 ~50 次/min,间隔中位数 0.67s;但 282-299s 段运动低谷时音频仍密集(邻场串扰实测确认) |
| YOLO11n person(MPS) | 预训练零训练即可用 | 球员置信度 0.77-0.92;26.3 FPS → 15 分钟视频约 3 分钟 |
| 运动能量信号 | 回合/间歇结构清晰 | 3 分钟窗口 10-12 次间歇(2-11s);检测框驱动,抗相机转向 |
| person 计数信号 | 无判据价值 | 观众/邻场人员干扰,人数与比赛状态弱相关 |

## 3. 架构设计

```
输入 video.mp4
 ├─ ① 采样    ffmpeg VideoToolbox 硬解 → 1280 宽 @ 5fps JPEG 帧(内存/磁盘流式)
 ├─ ② 检测    YOLO11n person 类(MPS,imgsz=1280,conf=0.25)→ 逐帧 person 框
 ├─ ③ 信号    球场 ROI 内大目标(h>12% 帧高)质心位移 → 运动能量曲线(2s 滑窗平滑)
 ├─ ④ 状态机  双阈值滞回 + 参数约束 → 回合区间列表
 ├─ ⑤ 音频精修(可选,默认开) 回合起点在状态机起点前 3s 内搜索最近击球瞬态微调;回合内瞬态计数入指标
 ├─ ⑥ 导出    ffmpeg -ss/-to 切片(前 1.5s / 后 2s 缓冲,重编码 H.264 保持边界精确)
 └─ 产出      clips/rally_NNN.mp4、clips/highlights.mp4(可选)、rallies.json、统计摘要
```

### 3.1 组件职责与边界

| 组件 | 职责 | 输入 → 输出 | 不做什么 |
|---|---|---|---|
| `sampler` | 解码与采样 | 视频 → 帧序列 + 元信息(fps/时长) | 不做检测 |
| `detector` | person 检测封装 | 帧序列 → 逐帧框 JSONL | 不做跟踪/分类 |
| `activity` | 运动能量信号 | 框 JSONL → 时间序列 | 不做切分决策 |
| `segmenter` | 状态机切分 | 能量曲线 → 回合区间 | 不依赖音频 |
| `refiner` | 音频瞬态精修 | 音轨 + 回合区间 → 修正后区间 + 击球计数 | 不独立产生回合 |
| `exporter` | 切片与集锦 | 回合区间 + 原视频 → mp4/json | 不修改检测逻辑 |
| `cli` | 参数解析与编排 | argv → 上述组件流水线 | 不含业务逻辑 |

### 3.2 状态机参数(v1 默认值,CLI 可覆盖)

- 平滑窗:2s;进入回合阈值:能量 P60;退出阈值:能量 P30(滞回)
- 最短回合:3s;最短间歇:2.5s;相邻回合间隔 < 最短间歇时合并
- 片段缓冲:头部 1.5s,尾部 2s

### 3.3 球场 ROI(可选人工标注,一次标注缓存复用)

- 首次运行自动取检测帧聚类估算主活动区域;提供 `--court-roi x,y,w,h` 手动覆盖
- 标注缓存写入输出目录(类似 Good-Badminton 的 court_annotations.txt 模式),同素材复用

## 4. CLI 接口

```
shuttlecut process <video...> [--out DIR] [--no-audio-refine] [--no-reel]
                  [--roi X,Y,W,H] [--min-rally S] [--min-idle S] [--device auto|mps|cpu]
shuttlecut label <video>            # 真值标注辅助工具(生成/编辑 ground truth JSON)
shuttlecut eval <video> [--gt FILE] # 对比 rallies.json 与真值,输出 P/R/F1
```

输出目录结构:

```
outputs/<video_stem>/
├── clips/rally_001.mp4 ...
├── clips/highlights.mp4
├── rallies.json        # [{id, start_s, end_s, duration_s, hits, motion_peak, confidence}]
├── persons.jsonl       # 逐帧检测原始数据(调试/复算)
└── roi.txt             # ROI 缓存
```

## 5. 错误处理

| 场景 | 对策 |
|---|---|
| 行人短暂遮挡(人走过) | 状态机最短回合时长容忍短暂信号丢失;遮挡 <2s 不切断回合 |
| 相机转向 | 检测框跟随人物,不依赖背景;ROI 丢失帧由滞回吸收 |
| 极端长间歇(中场休息) | 间歇上限 90s,超时拆分为独立片段组 |
| MPS 不可用 | `--device auto` 回退 CPU(速度约 1/5,功能不变) |
| 音轨缺失 | `refiner` 自动跳过,纯视觉切分 |
| 误检/漏检调节 | 全部阈值参数 CLI 暴露;rallies.json 带 confidence 供人工复核 |

## 6. 测试与验收

1. **单元测试**:sampler/detector 封装/activity 信号/状态机(合成信号用例:滞回、最短时长、合并)
2. **真值标注**:用 `shuttlecut label` 对 temp/ 两段视频人工标注回合边界(预计 30-40 个回合/段)
3. **验收指标**(v1 达标线):
   - 回合级召回 ≥ 90%(真值回合被切出,边界误差 ≤ 2s)
   - 回合级精确率 ≥ 90%(切出的片段中真实回合占比)
   - 15 分钟视频端到端处理 ≤ 8 分钟(M4)
4. **回归**:rallies.json 与真值对比进 `eval`,每次调参跑两段基准视频

## 7. 升级路径(v2+,不影响 v1 架构)

- v2a:接入 Good-Badminton yolo11s-ball.pt 做片段内球轨迹可视化(zero-shot 验证 → 需要时用其工具微调)
- v2b:击球点统计(音频瞬态 + 球轨迹融合,参考 kwyoke GRU 方案)
- v2c:半自动比分(回合归属建议 + 快捷键确认;ShuttleSensei 式 ActionFormer 为远期)
- v3:TrackNetV3 轨迹(MPS 缺失,需接受 CPU 3h/部或等后端支持)

## 8. 工程约定

遵循 `AGENTS.md`:临时产物入 `temp/`、Python 操作在工程 `.venv`、模型下载入 `models/`、全局安装需确认、子任务阻塞执行、结构化提问。

## 9. 参考项目清单

- yo-WASSUP/Good-Badminton(Apache-2.0)· qwpyyx fork
- xinyiz1226/Breakpoint(网球架构先例)
- qaz812345/TrackNetV3 · AnInsomniacy/tracknet-series-pytorch
- ychenfen/badminton-pipeline-repro(Apple Silicon 踩坑)
- kwyoke/Badminton-hit-detection(业余机位泛化)
- wywyWang/CoachAI-Projects(ShuttleSet 数据集)
- hhoao/huji-algorithm(同类产品参考)
