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
| 跨场馆零训练(V-JEPA 2 探针,路线 20) | **0.85 / 0.84**(DJI-14 训→B站 5 场馆;tools/frontier/) |
| 新视频 + 测试时自适配(TTA) | 最高 0.39/0.46(单视频有效不稳定) |

> 诚实结论(19 路线实验档案):R3D-18 W64 范式下跨视频边界方差 ±4-5s 不可后处理修复,
> 逐视频 0.1-0.5 为配置不变式;新视频提质请走下方「5 分钟校准协议」(实测 0.9-1.0)。


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
- 权重:`models/<ckpt>.pt`(训练产物,不随仓库分发);V-JEPA 探针与官方模型首用时自动下载

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
src/shuttlecut/          管线代码(cli/temporal/exporter/ffmpeg/audio/rank/labeling/eval)
tools/cuda/             训练套件(train_heavy/curves/segment_eval,见其 README)
tools/autolabel/        训练 GT 构建辅助(auto.py)
data/ground_truth/      段级真值(16 份:DJI 同域 14 + B站 4 场馆)
models/                 shuttlecut.pt(官方)/ shuttlecut-probe.pt(V-JEPA 探针)/ shuttlecut-<stem>.pt(校准)/ exp/(实验,git 忽略)
shuttlecut-output/      process 默认输出目录(git 忽略)
artifacts/curves/       实验概率曲线输出区(git 忽略)
docs/                   specs(设计)/ history(归档)/ eval-history.md(实测台账);操作知识全在 CLI --help
tests/                  pytest(56 项)
temp/                   工作区与素材(git 忽略)
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
shuttlecut eval shuttlecut-output\<stem>-rallies.json --gt data\ground_truth\<stem>.json   # 官方口径评测(--write-metadata 先产出)
shuttlecut label temp\<视频>.MP4 --sheets temp\sheets_<视频>                              # 真值标注辅助
```
# 模型获取(二选一)
- 已有官方模型:放到 `models/shuttlecut.pt`(process 零参数自动使用)
- 自行训练:`tools/cuda/train_heavy.py`(见 tools/cuda/README.md),产物重命名/软链为 `models/shuttlecut.pt`
- 新视频效果提升:跑一次 `shuttlecut calibrate`(生成该视频专属模型,自动优先使用)

## 模型命名规范(训练 → 晋升 → 使用)

```
models/
  shuttlecut.pt          # 官方生产模型:process 的唯一默认
  shuttlecut-<stem>.pt   # 视频专属校准模型:calibrate 产物,存在则自动优先
  exp/                   # 实验沙盒:train_heavy 直接产物,不参与任何自动查找
```

生命周期:
1. **训练**:`python tools/cuda/train_heavy.py --frames ... --gt ... --out models/exp/<tag>.pt`(实验模型一律入 exp/)
2. **评审**:`tools/cuda/curves.py + segment_eval.py` 对比基线(docs/eval-history.md 台账记录)
3. **晋升**:`Copy-Item models/exp/<胜出者>.pt models/shuttlecut.pt`
4. **使用**:`shuttlecut process 视频.MP4` 零参数(校准模型 → 官方模型依次自动查找);多模型投票属高级用法 `--model models/exp/a.pt,models/exp/b.pt`
5. **校准**:`shuttlecut calibrate 视频.MP4` 产出 `models/shuttlecut-<stem>.pt`,该视频后续 process 自动使用
## 完整参考(操作知识全部内置于 CLI,无独立手册)

- `shuttlecut --help` / `<命令> --help` —— **唯一权威操作文档**(决策总纲/标注协议/流契约/异常处置)
- `docs/eval-history.md` —— 实验台账(19 路线完整档案)
- `tools/cuda/README.md` —— 训练工具手册
- `AGENTS.md` —— 工程规约(模型命名/CLI 契约/质量基线)
