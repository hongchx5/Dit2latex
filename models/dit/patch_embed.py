"""
Patch Embedding（动态网格版）。

将潜在图 z_t' ∈ (B, in_channels, H, W) 转为 patch token 序列。
grid 尺寸（grid_h × grid_w）由输入形状动态决定，支持动态分桶训练：
不同桶（不同 latent 分辨率）可以复用同一 PatchEmbed。

可选：在 patchify 之前加入轻量残差局部特征增强
（Depthwise Conv(k×k) + Pointwise Conv(1×1)，残差加回）。
关闭时不创建任何参数，行为与原模型完全一致。

位置信息不再由本模块提供（移除固定 sin-cos buffer），
改由 DiT 内部基于 2D 坐标的 RoPE 提供（见 models/dit/attention.py）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


class LocalFeatureEnhance(nn.Module):
    """
    轻量残差局部特征增强。

    结构：Depthwise Conv(k×k) → Pointwise Conv(1×1) → 残差加回输入。
    Pointwise 零初始化，训练初期输出等价于原输入，便于从旧 checkpoint 继续训练。

    Args:
        channels: 输入/输出通道数（z_t' 的 8 通道）。
        kernel_size: 深度卷积核大小，必须为奇数以保持分辨率不变。
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"local_enhance kernel_size must be a positive odd number, got {kernel_size}")
        self.kernel_size = kernel_size
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=True,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        # 零初始化 pointwise，保证初始阶段等价于恒等映射
        nn.init.zeros_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pointwise(self.depthwise(x))


class PatchEmbed(nn.Module):
    """将潜在图 z_t' ∈ (B, in_channels, H, W) 转为 patch token 序列（动态 grid）。"""

    def __init__(
        self,
        patch_size: int,
        in_channels: int,
        hidden_dim: int,
        local_enhance: bool = False,
        local_enhance_kernel: int = 3,
    ):
        """
        Args:
            patch_size:   每个 patch 的空间尺寸。
            in_channels:  输入通道数（8，即 [z_t; z_p]）。
            hidden_dim:   输出 token 维度。
            local_enhance: 是否在 patchify 前做轻量残差局部增强。
            local_enhance_kernel: 增强模块深度卷积核大小。
        """
        super().__init__()
        self.patch_size = patch_size
        patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Linear(patch_dim, hidden_dim)

        if local_enhance:
            self.local_enhance = LocalFeatureEnhance(in_channels, local_enhance_kernel)
        else:
            # 关闭时不注册任何参数，state_dict 与原模型完全一致
            self.local_enhance = None

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, in_channels, H, W)

        Returns:
            tokens: (B, N, hidden_dim)，N = grid_h * grid_w
            grid_h: 网格行数（H // patch_size）
            grid_w: 网格列数（W // patch_size）
        """
        if self.local_enhance is not None:
            x = self.local_enhance(x)

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
