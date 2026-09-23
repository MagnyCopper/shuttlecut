import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "cuda"))

from train_heavy import WinDS, build_backbone  # noqa: E402


def test_r3d_18_backbone_forward():
    m, size = build_backbone("r3d_18", pretrained=False)
    m.eval()
    with torch.no_grad():
        y = m(torch.randn(2, 3, 32, size, size))
    assert y.shape == (2, 1)


def test_x3d_s_backbone_forward():
    pytest.importorskip("pytorchvideo")
    m, size = build_backbone("x3d_s", pretrained=False)
    m.eval()
    with torch.no_grad():
        y = m(torch.randn(2, 3, 32, size, size))
    assert y.shape == (2, 1)


def test_x3d_s_head_logits_not_softmax():
    pytest.importorskip("pytorchvideo")
    import torch.nn as nn
    m, _ = build_backbone("x3d_s", pretrained=False)
    assert isinstance(m.blocks[5].activation, nn.Identity)  # 结构自检:softmax 已换 Identity
    assert m.blocks[5].proj.out_features == 1


def test_winds_resizes_to_backbone_size():
    diffs = [np.random.rand(100, 90, 160).astype(np.float32)]
    items = [(0, 5, 1.0)]
    for size in (112, 160):
        x, y = WinDS(diffs, items, win=20, tsub=2, size=size)[0]  # 20//2=10>8 → 抽帧 10 帧
        assert x.shape == (3, 10, size, size)
        assert float(y) == 1.0


def test_unknown_backbone_raises():
    with pytest.raises(ValueError):
        build_backbone("resnet50")
