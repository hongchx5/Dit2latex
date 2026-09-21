"""
单个 DiT Block：AdaLN-Zero → Self-Attn → Style Cross-Attn → [Caption Cross-Attn] → SwiGLU MLP。

use_caption=True：SA / style-CA / caption-CA / MLP（AdaLN 4 组）
use_caption=False：SA / style-CA / MLP（AdaLN 3 组，不创建 caption 模块）
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.dit.attention import SelfAttention, CrossAttention
from models.dit.adaln import AdaLNZero, modulate


class SwiGLUMLP(nn.Module):
    """SwiGLU 前馈层。"""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mlp_ratio)
        self.w1 = nn.Linear(dim, inner_dim, bias=True)
        self.w2 = nn.Linear(dim, inner_dim, bias=True)
        self.w3 = nn.Linear(inner_dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(nn.functional.silu(self.w1(x)) * self.w2(x)))


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        cond_dim: int = 768 + 768,   # t_emb(768) + f_s_pooled(768)
        context_dim: int = 768,      # 风格序列特征维度
        caption_dim: int = 768,      # caption 序列特征维度（= hidden_dim）
        dropout: float = 0.0,
        use_caption: bool = True,
    ):
        super().__init__()
        self.use_caption = use_caption

        # AdaLN-Zero：3 组（无 caption）或 4 组（有 caption）
        num_groups = 4 if use_caption else 3
        self.adaln = AdaLNZero(cond_dim=cond_dim, hidden_dim=hidden_dim, num_groups=num_groups)

        # Self-Attention（2D RoPE）
        self.self_attn = SelfAttention(hidden_dim, num_heads, dropout=dropout)

        # Style Cross-Attention（K/V 来自风格序列）
        self.cross_attn = CrossAttention(hidden_dim, context_dim, num_heads, dropout=dropout)

        # Caption Cross-Attention（K/V 来自 latex caption 序列，带 padding mask）
        self.caption_attn = (
            CrossAttention(hidden_dim, caption_dim, num_heads, dropout=dropout)
            if use_caption else None
        )

        # MLP
        self.mlp = SwiGLUMLP(hidden_dim, mlp_ratio, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        f_s_seq: torch.Tensor,
        caption_seq: torch.Tensor,
        caption_mask: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:            patch token 序列 (B, N, hidden_dim)
            c:            条件向量 (B, cond_dim)，来自 [t_emb, f_s_pooled]
            f_s_seq:      风格特征序列 (B, M, context_dim)
            caption_seq:  latex caption 特征序列 (B, L, caption_dim)，use_caption=False 时忽略
            caption_mask: (B, L) bool，True=padding
            coords:       2D 坐标网格 (N, 2)

        Returns:
            (B, N, hidden_dim)
        """
        if self.use_caption:
            (s1, sc1, g1,          # Self-Attention
             s2, sc2, g2,          # Style Cross-Attention
             s3, sc3, g3,          # Caption Cross-Attention
             s4, sc4, g4) = self.adaln(c)  # MLP
        else:
            (s1, sc1, g1,          # Self-Attention
             s2, sc2, g2,          # Style Cross-Attention
             s3, sc3, g3) = self.adaln(c)  # MLP

        # Self-Attention 路径（2D RoPE）
        x = x + g1.unsqueeze(1) * self.self_attn(modulate(x, s1, sc1), coords)

        # Style Cross-Attention 路径
        x = x + g2.unsqueeze(1) * self.cross_attn(modulate(x, s2, sc2), f_s_seq)

        if self.use_caption:
            # Caption Cross-Attention 路径（padding 位置被 mask）
            x = x + g3.unsqueeze(1) * self.caption_attn(
                modulate(x, s3, sc3), caption_seq, caption_mask
            )
            # MLP 路径
            x = x + g4.unsqueeze(1) * self.mlp(modulate(x, s4, sc4))
        else:
            # MLP 路径（无 caption 时第 3 组是 MLP）
            x = x + g3.unsqueeze(1) * self.mlp(modulate(x, s3, sc3))

        return x
