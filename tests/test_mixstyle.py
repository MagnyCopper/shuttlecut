import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "cuda"))

from mixstyle import _mixstyle, attach  # noqa: E402


class Tiny5D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(4, 4, 3, padding=1)
        self.blocks = nn.ModuleList([nn.Conv3d(4, 4, 3, padding=1) for _ in range(5)])
        self.layer3 = nn.Conv3d(4, 4, 3, padding=1)
        self.layer4 = nn.Conv3d(4, 4, 3, padding=1)

    def forward(self, x):
        return self.blocks[3](self.blocks[2](self.conv(x)))


def test_mixstyle_identity_in_eval_and_prob_zero():
    torch.manual_seed(0)
    x = torch.randn(4, 4, 4, 8, 8)
    assert torch.equal(_mixstyle(x, p=0.0), x)


def test_mixstyle_changes_stats_in_train():
    torch.manual_seed(0)
    x = torch.randn(4, 8, 4, 6, 6) * 3 + 1
    # 强制激活(p=1):批量内统计应被混合,输出与输入不同
    out = _mixstyle(x.clone(), p=1.0)
    assert not torch.allclose(out, x)


def test_attach_eval_identity_train_differs():
    torch.manual_seed(0)
    m = Tiny5D()
    attach(m, "x3d_s", p=1.0)
    x = torch.randn(3, 4, 4, 8, 8)
    m.eval()
    with torch.no_grad():
        assert torch.allclose(m(x), m(x))  # eval 确定性
    m.train()
    with torch.no_grad():
        y1 = m(x)
        y2 = m(x)
    assert not torch.allclose(y1, y2)  # train 随机混合


def test_attach_unknown_backbone_raises():
    with pytest.raises(ValueError):
        attach(Tiny5D(), "resnet50")


def test_r3d_style_backbone_names():
    m = Tiny5D()
    hs = attach(m, "r3d_18", p=0.0)
    assert len(hs) == 2
    for h in hs:
        h.remove()
