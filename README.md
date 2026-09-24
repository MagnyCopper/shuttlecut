# ShuttleCut

[English](README.en.md) | 简体中文

[![CI](https://github.com/MagnyCopper/shuttlecut/actions/workflows/ci.yml/badge.svg)](https://github.com/MagnyCopper/shuttlecut/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

羽毛球回合自动剪辑:DJI Osmo Pocket 拍摄的整场视频进,回合片段与集锦出,**新视频零训练直接推理**。

> **English** — ShuttleCut turns full-match badminton videos into rally reels and
> highlight clips with one command. A frozen V-JEPA 2 encoder + lightweight probe
> achieves 0.85/0.84 P/R on unseen venues with zero training. CUDA GPUs and
> Apple Silicon (Metal) supported.
## 管线

```
MP4 → 15fps 抽帧(缓存,硬解) → V-JEPA 2 冻结特征 + 3 种子探针集成(默认,跨场馆)
    → [--model 显式指定时] LK-RANSAC 补偿差分 → R3D-18 滑窗分类 → 两尺度切分
    → 边界斜率吸附 → ffmpeg 硬编出片(片段 + 集锦,自动降级链)
```

实测成绩(官方口径 `src/shuttlecut/eval/evaluate.py`,tol 2.5s / 重叠 ≥0.5×较长段):

| 场景 | P / R |
|---|---|
| 跨场馆零训练(**V-JEPA 探针,默认路径**) | **0.85 / 0.84**(DJI-14 训→B站 5 场馆;LOEO 均值 0.83/0.80) |
| 校准协议(5 分钟标注+TTA) | 1.000/1.000 与 0.898/0.800(held-out 实测) |
| R3D 路线·训练集内 | 0.83~1.00 |
| R3D 路线·同场馆家族(零训练) | 0.15~0.54(边界方差墙) |

> 历史结论(19 路线实验档案):R3D+域内特征范式下跨视频 0.1-0.5 为配置不变式;
> 路线 20(冻结基础模型特征)已击穿该墙。完整实测史(含全部失败)见 `docs/eval-history.md`。

## 安装

```powershell
git clone https://github.com/MagnyCopper/shuttlecut.git; cd shuttlecut
uv venv .venv --python 3.12; .venv\Scripts\activate
uv pip install -e .            # CPU 版 torch,开箱可用

# GPU(Windows/NVIDIA,驱动 ≥ CUDA 13.x):
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
# 国内加速:--find-links https://mirrors.aliyun.com/pytorch-wheels/cu130/ --default-index https://mirrors.aliyun.com/pypi/simple/

winget install Gyan.FFmpeg     # Windows;新开终端后 ffmpeg -version 验证
brew install ffmpeg            # macOS(M 系列 Videotoolbox 硬编硬解自动启用)
# macOS 无需 CUDA:--device 自动走 mps(Metal)
```

素材与权重(不入库,放本地):

- 视频:任意路径;权重与 HF 编码器缓存(`models/hf`)首用时**自动下载**(SHA-256 校验)

## 使用

```powershell
# 核心用法:一条命令 → 恰好 2 个视频(回合合集 + 精彩选集)
shuttlecut process temp\<视频>.MP4
# 输出: shuttlecut-output/<stem>-all-rallies.mp4 + <stem>-highlights.mp4

# 常用选项
shuttlecut process a.MP4 b.MP4 -o exports --overwrite   # 多视频/指定目录/覆盖
shuttlecut process a.MP4 --model m1.pt,m2.pt,m3.pt      # R3D 显式指定(校准/投票)
shuttlecut process a.MP4 --write-metadata               # 额外出 <stem>-rallies.json(可选)
shuttlecut process a.MP4 --post 5                       # 片尾余量(偶见球在飞被切时调大)

# 完整帮助(含示例)
shuttlecut --help / shuttlecut process --help
```

模型解析(默认无需指定,任何视频行为一致):`shuttlecut-probe.pt`(V-JEPA 探针,缺失自动下载)→ `shuttlecut.pt`(官方 R3D 兜底);校准/实验模型经 `--model` 显式使用。
退出码:`0` 成功 / `1` 处理失败 / `2` 用法错误。进度走 stderr,摘要走 stdout。
10 分钟 4K 视频冷启动约 17 分钟(RTX 2070);重复处理仅重跑切片。

## 5 分钟校准协议(单视频追求 0.9-1.0 时)

```powershell
shuttlecut calibrate temp\<新视频>.MP4                                  # 1. 渲染 40 张条带+模板
# 2. 逐张查看 shuttlecut-output\<视频>\calib\strip_XX.jpg,把 calib_template.json 中每条 verdict 改为 6 值 Y/N 序列,存为 calib.json
shuttlecut calibrate temp\<新视频>.MP4 --phase run --calib shuttlecut-output\<视频>\calib\calib.json   # 3. TTA 适配(约 10 分钟 GPU)
shuttlecut process temp\<新视频>.MP4 --model models\shuttlecut-<stem>.pt  # 4. 出片(显式使用校准模型)
```

附录:开发/评测子命令

```powershell
shuttlecut eval shuttlecut-output\<stem>-rallies.json --gt data\ground_truth\<stem>.json   # 官方口径评测
shuttlecut label temp\<视频>.MP4 --sheets temp\sheets_<视频>                              # 真值标注辅助
shuttlecut cache --yes                                                                    # 清理可重建缓存
```

## 模型命名规范(训练 → 晋升 → 使用)

```
models/
  shuttlecut.pt          # 官方 R3D 生产模型(第二优先,自动解析兜底)
  shuttlecut-probe.pt    # V-JEPA 探针(第一优先,跨场馆零校准;3 种子集成)
  shuttlecut-<stem>.pt   # 视频专属校准模型:calibrate 产物,--model 显式使用
  exp/                   # 实验沙盒:train_heavy 直接产物,不参与任何自动查找
```

生命周期:
1. **训练**:`python tools/cuda/train_heavy.py --frames ... --gt ... --out models/exp/<tag>.pt`(实验模型一律入 exp/)
2. **评审**:`tools/cuda/curves.py + segment_eval.py` 对比基线(docs/eval-history.md 台账记录)
3. **晋升**:`Copy-Item models/exp/<胜出者>.pt models/shuttlecut.pt`
4. **使用**:`shuttlecut process 视频.MP4` 零参数(探针 → 官方 R3D 两层自动)
5. **校准**(可选):`shuttlecut calibrate 视频.MP4` 产出 `models/shuttlecut-<stem>.pt`,经 `--model` 显式使用

## 目录约定

```
src/shuttlecut/          管线代码(cli/temporal/exporter/ffmpeg/audio/rank/labeling/eval/vjepa/modelhub)
tools/cuda/             训练套件(train_heavy/curves/segment_eval,见其 README)
tools/frontier/         探针实验套件(vjepa_extract/loeo_probe/train_probe_prod)
tools/autolabel/        训练 GT 构建辅助(auto.py)
data/ground_truth/      段级真值(19 份:DJI 同域 14 + B站 5 场馆)
models/                 权重区(git 忽略;首用自动下载)
shuttlecut-output/      process 默认输出目录(git 忽略)
temp/                   工作区与素材(git 忽略)
docs/                   eval-history.md(实测台账)/ specs(设计)/ history(归档)
tests/                  pytest(63 项,三平台 CI)
```

## 完整参考(操作知识全部内置于 CLI,无独立手册)

- `shuttlecut --help` / `<命令> --help` —— **唯一权威操作文档**(决策总纲/标注协议/流契约/异常处置)
- `docs/eval-history.md` —— 实验台账(19 路线完整档案 + 路线 20)
- `CHANGELOG.md` / `CONTRIBUTING.md` —— 版本史与贡献指南
- `tools/cuda/README.md` —— 训练工具手册
- `AGENTS.md` —— 工程规约(模型命名/CLI 契约/质量基线)

## 规约

见 `AGENTS.md`(临时产物入 temp/、工程内虚拟环境、全局安装先确认、子任务阻塞、结构化提问)。

## License

MIT
