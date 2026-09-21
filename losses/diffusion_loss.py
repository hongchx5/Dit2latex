"""扩散损失：标准 MSE 噪声预测损失。"""

from __future__ import annotations

import torch


def simple_diffusion_loss(noise_pred: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """
    L_simple = MSE(ε̂, ε)，直接对噪声做均方误差。

    Args:
        noise_pred: 预测噪声 (B, C, H, W)
        noise:      真实噪声 (B, C, H, W)

    Returns:
        scalar loss
    """
    return torch.nn.functional.mse_loss(noise_pred, noise)
