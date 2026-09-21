"""
Patch Embedding（动态网格版）。

将潜在图 z_t' ∈ (B, in_channels, H, W) 转为 patch token 序列。
grid 尺寸（grid_h × grid_w）由输入形状动态决定，支持动态分桶训练：
不同桶（不同 latent 分辨率）可以复用同一 PatchEmbed。

位置信息不再由本模块提供（移除固定 sin-cos buffer），
改由 DiT 内部基于 2D 坐标的 RoPE 提供（见 models/dit/attention.py）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


class PatchEmbed(nn.Module):
    """将潜在图 z_t' ∈ (B, in_channels, H, W) 转为 patch token 序列（动态 grid）。"""

    def __init__(self, patch_size: int, in_channels: int, hidden_dim: int):
        """
        Args:
            patch_size:   每个 patch 的空间尺寸。
            in_channels:  输入通道数（8，即 [z_t; z_p]）。
            hidden_dim:   输出 token 维度。
        """
        super().__init__()
        self.patch_size = patch_size
        patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Linear(patch_dim, hidden_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, in_channels, H, W)

        Returns:
            tokens: (B, N, hidden_dim)，N = grid_h * grid_w
            grid_h: 网格行数（H // patch_size）
            grid_w: 网格列数（W // patch_size）
        """
        B, C, H, W = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, f"input {H}x{W} not divisible by patch_size {p}"
        grid_h, grid_w = H // p, W // p

        patches = rearrange(
            x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
            p1=p, p2=p,
        )  # (B, N, C*p*p)

        tokens = self.proj(patches)  # (B, N, hidden_dim)
        return tokens, grid_h, grid_w
