"""扩散损失：标准 MSE 噪声预测损失（支持可变分辨率的 token mask）。"""

from __future__ import annotations

from typing import Optional

import torch


def simple_diffusion_loss(
    noise_pred: torch.Tensor,
    noise: torch.Tensor,
    token_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    L_simple = MSE(ε̂, ε)，直接对噪声做均方误差。

    可变分辨率时，batch 内的样本被对齐到同一个 canvas，内容区之外的 token 是
    padding（token_mask=True）。这些位置的噪声既没有意义也没有监督信号，
    必须排除，否则：
      - 空白区的预测误差会稀释真实内容的梯度（分辨率越失衡稀释越严重）；
      - 模型会去"学会预测白噪声之外的东西"，白白消耗容量。

    Args:
        noise_pred: 预测噪声 (B, C, H, W)，H/W 为 latent 尺寸。
        noise:      真实噪声 (B, C, H, W)。
        token_mask: (B, N) bool，patch token 级 padding mask，True=padding。
                    N = (H // patch_size) × (W // patch_size)，patch_size 由
                    latent 与 token 网格的关系自动推出。None 时退化为普通 MSE。

    Returns:
        scalar loss
    """
    if token_mask is None:
        return torch.nn.functional.mse_loss(noise_pred, noise)

    B, C, H, W = noise_pred.shape
    N = token_mask.shape[1]

    # token 网格：grid_h × grid_w = N，且 latent = grid × patch_size
    grid_h = H // 2 if (H // 2) * (W // 2) == N else None
    # 通用做法：尝试所有能整除的 patch_size（通常就是 2）
    if grid_h is None:
        for p in (2, 1, 4):
            if H % p == 0 and W % p == 0 and (H // p) * (W // p) == N:
                grid_h = H // p
                break
    if grid_h is None:
        raise ValueError(
            f"cannot infer patch grid: noise_pred {tuple(noise_pred.shape)} vs token_mask N={N}"
        )
    grid_w = N // grid_h
    p_h, p_w = H // grid_h, W // grid_w

    # (B, N) -> (B, 1, H, W)：每个 token 铺到它覆盖的 latent 区域
    keep = (~token_mask).view(B, 1, grid_h, grid_w).to(noise_pred.dtype)
    keep = keep.repeat_interleave(p_h, dim=2).repeat_interleave(p_w, dim=3)
    keep = keep.expand(-1, C, -1, -1)

    denom = keep.sum().clamp(min=1.0)
    return ((noise_pred - noise) ** 2 * keep).sum() / denom
