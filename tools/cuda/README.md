# ShuttleCut CUDA 训练套件

目标:在 RTX 2070(8GB)上用 Kinetics 预训练的 R3D-18 重骨干微调回合时序分类器,突破本机 MPS 的曲线质量瓶颈。

## 环境(一次性)

```bash
# 建议 Python 3.11/3.12 + torch cu121
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python numpy scipy
```

## 数据

本目录随包附 `ground_truth/`(B1/B2 段级真值)。把两段原始视频放进 `videos/`:

```
videos/DJI_20260830153830_0015_D.MP4   # B1
videos/DJI_20260830173600_0025_D.MP4   # B2
```

抽帧(每段约 2-3 分钟):

```bash
python prep_data.py --video videos/DJI_20260830153830_0015_D.MP4 --gt ground_truth/DJI_20260830153830_0015_D.json
python prep_data.py --video videos/DJI_20260830173600_0025_D.MP4 --gt ground_truth/DJI_20260830173600_0025_D.json
```

## 训练(每视频 × 每窗长,各约 15-40 分钟)

```bash
mkdir -p ckpt
# B1:64 帧窗(主力)+ 24 帧窗(边界)
python train_heavy.py --frames data/DJI_20260830153830_0015_D/frames15 --gt ground_truth/DJI_20260830153830_0015_D.json --out ckpt/r3d_b1_w64.pt --win 64
python train_heavy.py --frames data/DJI_20260830153830_0015_D/frames15 --gt ground_truth/DJI_20260830153830_0015_D.json --out ckpt/r3d_b1_w24.pt --win 24
# B2 同理
python train_heavy.py --frames data/DJI_20260830173600_0025_D/frames15 --gt ground_truth/DJI_20260830173600_0025_D.json --out ckpt/r3d_b2_w64.pt --win 64
python train_heavy.py --frames data/DJI_20260830173600_0025_D/frames15 --gt ground_truth/DJI_20260830173600_0025_D.json --out ckpt/r3d_b2_w24.pt --win 24
```

VRAM 不足时加 `--batch 4`。

## 曲线 + 官方口径评测

```bash
python curves.py --frames data/DJI_20260830153830_0015_D/frames15 --ckpt ckpt/r3d_b1_w64.pt --win 64 --tag b1r3d64
python curves.py --frames data/DJI_20260830153830_0015_D/frames15 --ckpt ckpt/r3d_b1_w24.pt --win 24 --tag b1r3d24
python curves.py --frames data/DJI_20260830173600_0025_D/frames15 --ckpt ckpt/r3d_b2_w64.pt --win 64 --tag b2r3d64
python curves.py --frames data/DJI_20260830173600_0025_D/frames15 --ckpt ckpt/r3d_b2_w24.pt --win 24 --tag b2r3d24

python segment_eval.py --gt ground_truth/DJI_20260830153830_0015_D.json --tag b1r3d64 --grid
python segment_eval.py --gt ground_truth/DJI_20260830173600_0025_D.json --tag b2r3d64 --grid
```

## 交付回 Mac

把以下文件拷回工程 `temp/`(其余不用):
- `ckpt/*.pt`(4 个模型)
- `prob_centers_*.npy` + `prob_values_*.npy`(4 组曲线)
- 训练/评测日志文本

我会在本机做集成切分、官方 evaluate 终评与管线接入。

## 参考指标(本机 tiny-CNN 基线,待超越)

| 视频 | 官方口径 P/R | 宽松口径 P/R |
|---|---|---|
| B1 | 0.886 / 0.796 | 0.94 / 0.92 |
| B2 | 0.719 / 0.661 | — |
| B1训→B2迁 | — | F1 0.61-0.65 |
