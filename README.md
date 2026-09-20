# ShuttleCut

羽毛球回合自动剪辑:DJI Osmo Pocket 拍摄的整场视频进,回合片段与集锦出,**新视频零训练直接推理**。

## 管线

```
MP4 → 15fps 抽帧(缓存) → LK-RANSAC 全局运动补偿差分
    → R3D-18 滑窗分类(64 帧窗, stride 2) → W64 滞回 + W24 深谷两尺度切分
    → ffmpeg 硬编出片(片段 + 集锦)
```

实测成绩(官方口径 `src/shuttlecut/eval/evaluate.py`,tol 2.5s / 重叠 ≥0.5×较长段;多模型边界投票配方):

| 场景 | P / R |
|---|---|
| 训练集内视频(E12,10 视频 v4 GT) | 0.83~1.00 |
| 新视频同场馆家族(零训练) | 0.15~0.54(边界方差墙,见下) |
| 新视频跨场馆 tol 5s | 0.49 / 0.61 |
| 新视频 + 测试时自适配(TTA) | 最高 0.39/0.46(单视频有效不稳定) |

> 诚实结论(14 路线实验档案):R3D-18 W64 范式下跨视频边界方差 ±4-5s 不可后处理修复;
> 0.90 跨场馆目标需新范式(时序 Transformer/光流,见 docs/eval-history.md 尾部)。
同场馆新视频可用 `--tune` 自微调补强。

| 场景 | P / R |
|---|---|
| 同域(joint2 训练视频内) | 0.957/0.918 ~ 0.984/0.968 |
| **零训练跨场馆(0037)** | **0.818 / 0.964**(目标双 ≥0.90,攻坚中) |

完整实测史(全部实验含失败)见 `docs/eval-history.md`;当前方案设计见 `docs/specs/`。

## 安装

```powershell
git clone https://github.com/MagnyCopper/shuttlecut.git; cd shuttlecut
uv venv .venv --python 3.12; .venv\Scripts\activate
uv pip install -e .            # CPU 版 torch,开箱可用

# GPU(RTX 2070 实测,驱动 ≥ CUDA 13.x):
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
# 国内加速:--find-links https://mirrors.aliyun.com/pytorch-wheels/cu130/ --default-index https://mirrors.aliyun.com/pypi/simple/

winget install Gyan.FFmpeg     # 新开终端后 ffmpeg -version 验证
```

素材与权重(不入库,放本地):

- 视频:`temp/*.MP4`(命名与 `data/ground_truth/*.json` 的 video 字段一致)
- 权重:`models/<ckpt>.pt`(训练产物,不随仓库分发;用 tools/cuda/train_heavy.py 训练)

## 使用

```powershell
# 核心用法:一条命令 → 恰好 2 个视频(回合合集 + 精彩选集)
shuttlecut process temp\<视频>.MP4
# 输出: shuttlecut-output/<stem>-all-rallies.mp4 + <stem>-highlights.mp4

# 常用选项
shuttlecut process a.MP4 b.MP4 -o exports --overwrite   # 多视频/指定目录/覆盖
shuttlecut process a.MP4 --model m1.pt,m2.pt,m3.pt      # 多模型边界投票(最佳跨视频配置)
shuttlecut process a.MP4 --write-metadata               # 额外出 <stem>-rallies.json(可选)

# 完整帮助(含示例)
shuttlecut --help / shuttlecut process --help
```

退出码:`0` 成功 / `1` 处理失败(模型缺失、输出已存在等) / `2` 用法错误。
进度走 stderr,摘要与输出路径走 stdout(便于管道)。

训练与曲线工具见 `tools/cuda/README.md`(Windows RTX 2070 手册)。

## 目录约定

```
src/shuttlecut/     管线代码(cli/temporal/exporter/ffmpeg/audio/labeling/eval)
tools/              cuda 训练套件 + autolabel 自主标注
data/ground_truth/  段级真值(B1: 49 回合;B2: 62 回合;0037: 28 回合)
artifacts/curves/   实验概率曲线输出区(curves.py 默认落盘,git 忽略)
docs/               specs(现行设计)/ history(归档)/ eval-history.md(实测台账)
tests/              pytest
temp/ outputs/ models/   工作区与产物(git 忽略)
```

## 规约

见 `AGENTS.md`(临时产物入 temp/、工程内虚拟环境、全局安装先确认、子任务阻塞、结构化提问)。

## License

MIT
## 5 分钟校准协议(新视频达到 0.9-1.0 的推荐路径)

```powershell
shuttlecut calibrate temp\<新视频>.MP4                                  # 1. 渲染 40 张条带+模板
# 2. 逐张查看 shuttlecut-output\<视频>\calib\strip_XX.jpg,把 calib_template.json 中每条 verdict 改为 6 值 Y/N 序列,存为 calib.json
shuttlecut calibrate temp\<新视频>.MP4 --phase run --calib shuttlecut-output\<视频>\calib\calib.json   # 3. TTA 适配(约 10 分钟 GPU)
shuttlecut process temp\<新视频>.MP4                                    # 4. 出片(自动优先使用校准模型)
实测(u0010/u0014 held-out):条带标注+TTA 后 **P/R = 1.000/1.000 与 0.898/0.800**;零训练跨场馆则 0.1-0.5 抽签(19 路线实验档案见 docs/eval-history.md)。
# 附录:开发/评测子命令
```powershell
shuttlecut eval shuttlecut-output\<视频>-rallies.json --gt data\ground_truth\<视频>.json   # 官方口径评测(需 --write-metadata)
shuttlecut label temp\<视频>.MP4 --sheets temp\sheets_<视频>                              # 真值标注辅助
```
# 模型获取(二选一)
- 已有官方模型:放到 `models/shuttlecut.pt`(process 零参数自动使用)
- 自行训练:`tools/cuda/train_heavy.py`(见 tools/cuda/README.md),产物重命名/软链为 `models/shuttlecut.pt`
- 新视频效果提升:跑一次 `shuttlecut calibrate`(生成该视频专属模型,自动优先使用)
