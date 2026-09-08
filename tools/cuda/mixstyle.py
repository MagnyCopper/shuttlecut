"""MixStyle(域风格混合)训练期增广:治混合场馆联合训练的负迁移。

Zhou et al., "Domain Generalization with MixStyle", ICLR 2021.
实现为前向钩子:训练态以概率 p 在 batch 维混合激活统计(均值/标准差),
推理态严格恒等。对 5D 视频激活 (B,C,T,H,W) 在 (T,H,W) 维统计。
"""
from __future__ import annotations

import random

import torch
from torch.utils.hooks import RemovableHandle



def _mixstyle(x: torch.Tensor, p: float = 0.5, alpha: float = 0.1) -> torch.Tensor:
    if not torch.is_tensor(x) or x.dim() != 5 or x.size(0) < 2:
        return x
    if random.random() > p:
        return x
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(x.size(0), device=x.device)
    mu = x.mean(dim=(2, 3, 4), keepdim=True)
    sig = x.std(dim=(2, 3, 4), keepdim=True)
    x_norm = (x - mu) / (sig + 1e-6)
    mu2, sig2 = mu[perm], sig[perm]
    mu_mix = lam * mu + (1 - lam) * mu2
    sig_mix = lam * sig + (1 - lam) * sig2
    return x_norm * sig_mix + mu_mix


def attach(model: torch.nn.Module, backbone: str, p: float = 0.5, alpha: float = 0.1,
           ) -> list[RemovableHandle]:
    """在骨干中后层输出上注册 MixStyle 钩子(仅 model.train() 时生效)。返回句柄列表。

    x3d_s: blocks[2] 与 blocks[3](时序中级特征,风格敏感且语义未特化)
    r3d_18: layer3 与 layer4(对应深度)
    """
    modules: list[torch.nn.Module] = []
    if backbone == "x3d_s":
        modules = [model.blocks[2], model.blocks[3]]
    elif backbone == "r3d_18":
        modules = [model.layer3, model.layer4]
    else:
        raise ValueError(f"MixStyle 未定义骨干 {backbone}")

    def hook(_m, _i, out):
        if not model.training:
            return out
        if isinstance(out, (tuple, list)):
            return out
        return _mixstyle(out, p, alpha)

    return [m.register_forward_hook(hook) for m in modules]
