"""
AdaLN-Zero 调制模块。

从时间嵌入和风格特征的拼接中，通过 MLP 预测 9 组调制参数：
  γ1, β1, α1 — 用于 Self-Attention 前的 LayerNorm 调制 + residual gating
  γ2, β2, α2 — 用于 Cross-Attention 前的 LayerNorm 调制 + residual gating
  γ3, β3, α3 — 用于 MLP 前的 LayerNorm 调制 + residual gating

AdaLN-Zero 的特点：α 初始化为 0，使得训练初期残差连接不起作用，
模型从恒等映射开始，训练更稳定。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AdaLNZero(nn.Module):
    """
    为单个 DiT Block 产生 AdaLN 参数。

    num_groups=3（无 caption）：SA, style-CA, MLP
    num_groups=4（有 caption）：SA, style-CA, caption-CA, MLP

    每组 3 个参数（shift, scale, gate）。
    """

    def __init__(self, cond_dim: int, hidden_dim: int, num_groups: int = 3):
        super().__init__()
        self.num_groups = num_groups
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 3 * num_groups, bias=True),
        )
        nn.init.zeros_(self.mlp[1].weight)
        nn.init.zeros_(self.mlp[1].bias)

    def forward(self, c: torch.Tensor) -> tuple:
        """
        Returns:
            num_groups 组 × (shift, scale, gate)，共 3*num_groups 个参数。
            组顺序（调用方按此解释）：
              group 1: Self-Attention
              group 2: Style Cross-Attention
              group 3: Caption Cross-Attention（仅 num_groups=4）或 MLP（num_groups=3）
              group 4: MLP（仅 num_groups=4）
        """
        params = self.mlp(c)  # (B, 3*num_groups*hidden_dim)
        return tuple(params.chunk(3 * self.num_groups, dim=-1))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    AdaLN 调制：LN → scale → shift。

    x_out = (1 + scale) * LayerNorm(x) + shift
    """
    x = nn.functional.layer_norm(x, x.shape[-1:])
    shift = shift.unsqueeze(1)
    scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift
