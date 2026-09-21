"""Multi-head Self-Attention 和 Cross-Attention 模块。

Self-Attention 支持 2D RoPE（旋转位置编码），支持动态分桶下的任意 grid 尺寸：
head_dim 前一半维度用行坐标旋转，后一半用列坐标旋转，注意力对行差/列差都敏感。

Cross-Attention（Q 来自 DiT 特征，K/V 来自风格序列）不施加 DiT grid 的 RoPE：
风格序列坐标系与 DiT grid 无关（CLIP 自带位置编码），保持原样。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ── 2D RoPE ─────────────────────────────────────────────────────────


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """标准 RoPE 旋转（前后两半配对）。"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x:   (B, H, N, D)
        cos: (N, D)
        sin: (N, D)
    Returns:
        旋转后的 (B, H, N, D)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


class RotaryPositionEmbedding2D(nn.Module):
    """
    2D 旋转位置编码：head_dim 前一半（行组）用行坐标旋转，后一半（列组）用列坐标旋转。

    支持任意 grid 尺寸：坐标 (row, col) 由输入形状决定，实时生成旋转矩阵，
    天然适配动态分桶训练（不同桶 = 不同 grid，无需预计算固定长度 buffer）。
    """

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 4 == 0, f"head_dim must be divisible by 4 for 2D RoPE, got {head_dim}"
        self.head_dim = head_dim
        self.half = head_dim // 2          # 行组 / 列组各占 head_dim // 2 维
        freq_dim = self.half // 2          # 每组内每对维度一个频率
        inv_freq = 1.0 / (base ** (torch.arange(0, freq_dim, dtype=torch.float32) / freq_dim))
        self.register_buffer("inv_freq", inv_freq)

    def _cos_sin_1d(self, pos: torch.Tensor) -> tuple:
        """
        Args:
            pos: (N,) 一维坐标（行或列）。
        Returns:
            cos, sin: (N, self.half)
        """
        angles = pos.unsqueeze(-1) * self.inv_freq  # (N, freq_dim)
        cos = torch.cos(angles).repeat_interleave(2, dim=-1)  # (N, half)
        sin = torch.sin(angles).repeat_interleave(2, dim=-1)
        return cos, sin

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        coords: torch.Tensor,
    ) -> tuple:
        """
        Args:
            q, k: (B, H, N, head_dim)
            coords: (N, 2) float，每行 [row, col] 坐标（row ∈ [0, grid_h)，col ∈ [0, grid_w)）。
        Returns:
            (q_rot, k_rot)：旋转后的 (B, H, N, head_dim)
        """
        B, H, N, D = q.shape
        assert D == self.head_dim
        h = self.half

        cos_row, sin_row = self._cos_sin_1d(coords[:, 0])
        cos_col, sin_col = self._cos_sin_1d(coords[:, 1])

        q_row = apply_rotary(q[..., :h], cos_row, sin_row)
        q_col = apply_rotary(q[..., h:], cos_col, sin_col)
        k_row = apply_rotary(k[..., :h], cos_row, sin_row)
        k_col = apply_rotary(k[..., h:], cos_col, sin_col)

        q_rot = torch.cat([q_row, q_col], dim=-1)
        k_rot = torch.cat([k_row, k_col], dim=-1)
        return q_rot, k_rot


# ── 注意力模块 ──────────────────────────────────────────────────────


class SelfAttention(nn.Module):
    """Multi-head Self-Attention + 2D RoPE。"""

    def __init__(self, dim: int, num_heads: int = 12, dropout: float = 0.0, use_rope: bool = True):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rope = use_rope

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        if use_rope:
            self.rotary = RotaryPositionEmbedding2D(self.head_dim)
        else:
            self.rotary = None

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, dim) patch token 序列。
            coords: (N, 2) float 坐标网格（row, col）。
        Returns:
            (B, N, dim)
        """
        B, N, D = x.shape
        qkv = self.qkv(x)  # (B, N, 3*D)
        qkv = rearrange(qkv, "b n (h d three) -> three b h n d", h=self.num_heads, three=3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, N, head_dim)

        if self.rotary is not None:
            q, k = self.rotary(q, k, coords)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (B, H, N, head_dim)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.proj(out)


class CrossAttention(nn.Module):
    """Multi-head Cross-Attention：Q 来自 DiT 特征，K/V 来自风格特征序列（无 RoPE）。"""

    def __init__(self, dim: int, context_dim: int, num_heads: int = 12, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_kv = nn.Linear(context_dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:               DiT 特征 (B, N, dim)
            context:         风格/caption 特征序列 (B, M, context_dim)
            key_padding_mask: (B, M) bool，True 表示忽略该 key（如 caption padding 位置）
        Returns:
            (B, N, dim)
        """
        B, N, _ = x.shape
        M = context.shape[1]

        q = self.to_q(x)  # (B, N, dim)
        kv = self.to_kv(context)  # (B, M, 2*dim)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        kv = rearrange(kv, "b m (h d two) -> two b h m d", h=self.num_heads, two=2)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            # (B, M) -> (B, 1, 1, M)，mask 位置置 -inf
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.proj(out)
