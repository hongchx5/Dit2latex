"""
感知损失：VGG16 中间层特征的 MSE。
约束生成图像 I_g 的语义与打印体 I_p 一致，防止过拟合导致测试乱码。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGPerceptualLoss(nn.Module):
    """VGG16 感知损失（from relu3_3）。"""

    def __init__(self, device: str = "cuda"):
        super().__init__()
        try:
            from torchvision.models import vgg16, VGG16_Weights
        except ImportError:
            raise ImportError("torchvision required for VGG perceptual loss")

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device)
        vgg.requires_grad_(False)
        vgg.eval()

        # 取到 relu3_3（第 16 层，0-indexed）
        self.features = vgg.features[:16].to(device)
        self.features.requires_grad_(False)

        # VGG 输入归一化
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 生成图像 (B, 3, H, W) in [-1, 1]
            y: 打印体图像 (B, 3, H, W) in [-1, 1]

        Returns:
            scalar perceptual loss
        """
        # 反归一化到 [0, 1] → VGG 归一化
        x = (x * 0.5 + 0.5).clamp(0, 1)
        y = (y * 0.5 + 0.5).clamp(0, 1)

        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        feat_x = self.features(x)
        feat_y = self.features(y)

        return F.mse_loss(feat_x, feat_y)


def build_perceptual_loss(device: str = "cuda") -> nn.Module:
    """创建感知损失实例。"""
    return VGGPerceptualLoss(device=device)


# 别名，方便导入
PerceptualLoss = VGGPerceptualLoss
