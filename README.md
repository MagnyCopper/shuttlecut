# ShuttleCut

DJI Osmo Pocket 羽毛球视频自动高亮剪辑:自动定位连续对抗回合片段,导出独立片段与集锦。

当前状态:时序分类器路线(补偿帧差 + 窗口分类器)已在 B1 实测段级 P=0.886/R=0.796(官方口径)与 P=0.94/R=0.92(宽松口径);重骨干(R3D-18)CUDA 微调进行中,目标段级 R/P ≥ 0.90。完整实测史见 `outputs/eval_history.md`。

## 多机同步设置(新机器)

```bash
git clone https://github.com/MagnyCopper/shuttlecut.git && cd shuttlecut

# 1. Python 环境(工程内虚拟环境)
python3 -m venv .venv
source .venv/bin/activate            # macOS/Linux
pip install -r requirements.txt      # 或 uv pip install -r requirements.txt --python .venv/bin/python
# Apple Silicon 训练/推理附带:brew install ffmpeg

# 2. 视频文件(人工导入,不入库)
mkdir -p temp videos
# 把 DJI_*.MP4 放进 temp/,命名与 ground_truth/*.json 的 video 字段一致

# 3. 模型权重(不入库,按需下载)
mkdir -p models
# - RTMPose / YOLO person:见 docs/superpowers/specs/2026-09-01-shuttlecut-v1-design.md
# - TrackNetV3 官方权重:gdown 1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA 解压到 models/tracknetv3/
# - ETH shuttle YOLO:git lfs pull(见 third_party 说明)或 curl media.githubusercontent.com .../shuttle_detection/.../best.pt → models/eth_shuttle/
# - 时序分类器 ckpt(models/temporal_*.pt):由各机器训练产物同步,或用 tools/cuda 套件重训

# 4. 第三方仓库(不入库)
git clone --depth 1 https://github.com/ZSHYC/BadmintonTrackNet.git third_party/BadmintonTrackNet
git apply tools/patches/badmintontracknet_weights_only.patch
git clone --depth 1 https://github.com/leggedrobotics/shuttle_detection.git third_party/shuttle_detection

# 5. 验证
python -m pytest -q          # 88 项测试应全绿
```

CUDA 机器(RTX 2070)训练重骨干:`tools/cuda/README.md`。

## 目录约定

- `temp/` — 视频/帧缓存/中间产物(git 忽略)
- `temp/work/<视频名>/frames15/` — 15fps 帧缓存(时序模型输入)
- `ground_truth/` — 段级真值(B1: 49 回合;B2: 62 回合)
- `outputs/<视频名>/` — 运行产物(片段/缓存,git 忽略);`outputs/eval_history.md` 入库
- `models/` — 权重(git 忽略)
- `docs/` — 设计/验收/调研文档;`tools/cuda/` — CUDA 训练套件

## 规约

见 `AGENTS.md`(临时文件入 temp/、工程内虚拟环境、全局安装需确认、子任务阻塞、结构化提问)。
