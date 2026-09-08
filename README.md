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
# - 【必需·零训练主力】R3D 时序分类器(132MB/个,超 GitHub 100MB 上限不入库,从旧机器拷贝):
#   models/r3d_joint_b1b2_w64.pt   ← joint2(B1+B2 联合),新视频零训练直出:--temporal --temporal-ckpt models/r3d_joint_b1b2_w64.pt
#   可选:models/r3d_DJI_*_w64.pt(各视频专家)、models/r3d_DJI_20260830173600_0025_D_w24.pt(B2 两尺度第二模型)
#   可选:models/r3d_joint3*_w64.pt(0031 联合实验权重,已证损害泛化,仅存档)
# - RTMPose / YOLO person / TrackNetV3 / ETH shuttle YOLO:旧路线遗留,时序管线用不到,可不装
# - 时序 tiny-CNN ckpt(models/temporal_*.pt)与概率曲线(temp/prob_*.npy):已随仓库入库,clone 即得

# 4. 第三方仓库(不入库)
git clone --depth 1 https://github.com/ZSHYC/BadmintonTrackNet.git third_party/BadmintonTrackNet
git apply tools/patches/badmintontracknet_weights_only.patch
git clone --depth 1 https://github.com/leggedrobotics/shuttle_detection.git third_party/shuttle_detection

# 5. 验证
python -m pytest -q          # 96 项测试应全绿
shuttlecut process temp/<新视频>.MP4 --temporal --temporal-ckpt models/r3d_joint_b1b2_w64.pt   # 零训练出片烟雾测试
```

## Windows(RTX 2070)迁移

推理与训练均可用,步骤同上(PowerShell):

```powershell
git clone https://github.com/MagnyCopper/shuttlecut.git; cd shuttlecut
python -m venv .venv; .venv\Scripts\activate
pip install -r requirements.txt        # requirements.txt 装的是 CPU 版 torch;要用 RTX 2070 训练/推理再执行:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
winget install Gyan.FFmpeg             # 装完重开 PowerShell,ffmpeg -version 验证
# 手动拷贝(从 Mac):
#   models/r3d_joint_b1b2_w64.pt        ← 必需(132MB)
#   temp/DJI_*.MP4                       ← 你的视频
python -m pytest -q
```

- 推理设备自动选择(MPS→CUDA→CPU),Windows 上自动走 CUDA,比 Mac MPS 更快更稳(无楔死问题)
- 零训练出片:`python -m shuttlecut.cli process temp/<视频>.MP4 --temporal --temporal-ckpt models/r3d_joint_b1b2_w64.pt`(或 pip install -e . 后直接 `shuttlecut process ...`)
- 训练升级路径(RTX 2070 快 3-5 倍):`tools/cuda/README.md` Windows 手册(含 `--resume` 断点续训);曲线 `tools/cuda/curves.py --device cuda`
- Mac 上的概率曲线 temp/prob_*.npy 已入库,Windows clone 即得,可离线复算切分/评测

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
