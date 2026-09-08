# ShuttleCut

羽毛球回合自动剪辑:DJI Osmo Pocket 拍摄的整场视频进,回合片段与集锦出,**新视频零训练直接推理**。

## 管线

```
MP4 → 15fps 抽帧(缓存) → LK-RANSAC 全局运动补偿差分
    → R3D-18 滑窗分类(64 帧窗, stride 2) → W64 滞回 + W24 深谷两尺度切分
    → ffmpeg 硬编出片(片段 + 集锦)
```

实测成绩(官方口径 `src/shuttlecut/eval/evaluate.py`,tol 2.5s / 重叠 ≥0.5×较长段):

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
- 权重:`models/r3d_joint_b1b2_w64.pt`(必需,132MB,零训练主力)

## 使用

```powershell
# 零训练出片(核心用法)
shuttlecut process temp\<视频>.MP4 --temporal-ckpt models\r3d_joint_b1b2_w64.pt

# 评测(官方口径)
shuttlecut eval outputs\<视频>\rallies.json --gt data\ground_truth\<视频>.json

# 真值标注辅助(接触表/密集帧条)
shuttlecut label temp\<视频>.MP4 --sheets temp\sheets_<视频>

# 自主标注(模型预标 + 视觉分诊 + 音频票)
python tools\autolabel\label.py --video temp\<视频>.MP4 --ckpt models\r3d_joint_b1b2_w64.pt
```

训练与曲线工具见 `tools/cuda/README.md`(Windows RTX 2070 手册:抽帧 → train_heavy.py → curves.py → segment_eval.py)。

## 目录约定

```
src/shuttlecut/     管线代码(cli/temporal/exporter/ffmpeg/audio/labeling/eval)
tools/              cuda 训练套件 + autolabel 自主标注
data/ground_truth/  段级真值(B1: 49 回合;B2: 62 回合;0037: 28 回合)
artifacts/curves/   历史实验概率曲线(prob_*.npy,可离线复算切分与评测)
docs/               specs(现行设计)/ history(归档)/ eval-history.md(实测台账)
tests/              pytest
temp/ outputs/ models/   工作区与产物(git 忽略)
```

## 规约

见 `AGENTS.md`(临时产物入 temp/、工程内虚拟环境、全局安装先确认、子任务阻塞、结构化提问)。

## License

MIT