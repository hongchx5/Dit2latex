"""
DiT (Diffusion Transformer) 完整模型（动态分辨率版）。

输入：z_t' = [z_t; z_p] ∈ (B, 8, latent_h, latent_w)（噪声+内容条件的拼接潜在）
      f_s_seq ∈ (B, M, context_dim)（风格特征序列，用于 Style Cross-Attention）
      f_s_pooled ∈ (B, feature_dim)（风格全局特征，用于 AdaLN）
      t ∈ (B,)（扩散时间步）
      [caption_seq ∈ (B, L, caption_dim), caption_mask ∈ (B, L) bool]（可选，use_caption=True 时）

输出：ε̂ ∈ (B, 4, latent_h, latent_w)（预测的噪声）

use_caption=False 时，DiTBlock 不创建 caption Cross-Attention（AdaLN 3 组），
caption_seq/caption_mask 参数被忽略（可传 None）。

支持动态/静态分桶：latent 尺寸由输入形状决定，位置信息由 2D RoPE 提供。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dit.patch_embed import PatchEmbed
from models.dit.dit_block import DiTBlock


class TimestepEmbedder(nn.Module):
    """将扩散时间步 t 编码为 sin-cos 嵌入 + MLP。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """sin-cos 时间步嵌入（与 DiT 原版一致）。"""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.timestep_embedding(t, self.hidden_dim)
        return self.mlp(t_emb)


class DiT(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        depth: int = 12,
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        in_channels: int = 8,
        out_channels: int = 8,
        latent_dim: int = 4,
        context_dim: int = 768,
        caption_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_caption: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.use_caption = use_caption
        # caption 特征维度默认跟随 hidden_dim（CaptionEncoder 输出维度 = hidden_dim）
        if caption_dim is None:
            caption_dim = hidden_dim

        # Patch embedding（动态 grid，分辨率由输入决定）
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_dim=hidden_dim,
        )

        # 时间嵌入
        self.t_embedder = TimestepEmbedder(hidden_dim)

        # 风格池化 → AdaLN 条件
        self.style_pool_proj = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
        )

        # cond_dim = t_emb(hidden_dim) + f_s_proj(hidden_dim)
        cond_dim = hidden_dim * 2

        # DiT Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                cond_dim=cond_dim,
                context_dim=context_dim,
                caption_dim=caption_dim,
                dropout=dropout,
                use_caption=use_caption,
            )
            for _ in range(depth)
        ])

        # 输出头
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        patch_dim = out_channels * patch_size * patch_size
        self.final_proj = nn.Linear(hidden_dim, patch_dim, bias=True)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        # 1. 标准 Transformer 初始化（所有 Linear 先 xavier）
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # 2. Patch embedding 投影小权重
        nn.init.normal_(self.patch_embed.proj.weight, std=0.02)
        nn.init.zeros_(self.patch_embed.proj.bias)

        # 3. 输出投影零初始化（AdaLN-Zero 风格）
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

        # 4. 恢复 AdaLN-Zero 严格零初始化（第 1 步 xavier 会覆盖，必须重新清零）。
        #    这是"训练初期残差不起作用、模型从恒等映射开始"的稳定性机制，
        #    被覆盖会导致大模型 + fp16 下训练后期 loss 变 NaN。
        for block in self.blocks:
            nn.init.zeros_(block.adaln.mlp[1].weight)
            nn.init.zeros_(block.adaln.mlp[1].bias)

    @staticmethod
    def _make_coords(grid_h: int, grid_w: int, device: torch.device) -> torch.Tensor:
        """
        生成 2D 坐标网格（行主序，与 patch token 顺序一致）。
        Returns:
            coords: (grid_h * grid_w, 2) float，每行 [row, col]。
        """
        rows = torch.arange(grid_h, device=device, dtype=torch.float32)
        cols = torch.arange(grid_w, device=device, dtype=torch.float32)
        grid = torch.stack(torch.meshgrid(rows, cols, indexing="ij"), dim=-1)  # (grid_h, grid_w, 2)
        return grid.reshape(-1, 2)

    def forward(
        self,
        z_t_prime: torch.Tensor,
        f_s_seq: torch.Tensor,
        f_s_pooled: torch.Tensor,
        caption_seq: Optional[torch.Tensor],
        caption_mask: Optional[torch.Tensor],
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_t_prime:    (B, 8, latent_h, latent_w) — [z_t; z_p] 拼接
            f_s_seq:      (B, M, context_dim) — 风格特征序列
            f_s_pooled:   (B, context_dim) — 风格全局池化特征
            caption_seq:  (B, L, caption_dim) — latex caption 序列（use_caption=False 时忽略，可为 None）
            caption_mask: (B, L) bool — caption padding mask（同上）
            t:            (B,) — 扩散时间步

        Returns:
            epsilon: (B, 4, latent_h, latent_w) — 预测噪声（仅前 latent_dim 通道）
        """
        B = z_t_prime.shape[0]

        # Patch embedding（动态 grid）
        x, grid_h, grid_w = self.patch_embed(z_t_prime)  # (B, N, hidden_dim)

        # 2D 坐标网格（RoPE 位置信息）
        coords = self._make_coords(grid_h, grid_w, z_t_prime.device)  # (N, 2)

        # 时间嵌入
        t_emb = self.t_embedder(t)  # (B, hidden_dim)

        # 风格池化投影
        f_s_proj = self.style_pool_proj(f_s_pooled)  # (B, hidden_dim)

        # 拼接条件向量
        c = torch.cat([t_emb, f_s_proj], dim=-1)  # (B, 2 * hidden_dim)

        # 逐层处理
        for block in self.blocks:
            x = block(x, c, f_s_seq, caption_seq, caption_mask, coords)

        # 输出头
        x = self.final_norm(x)
        x = self.final_proj(x)  # (B, N, patch_dim)

        # rearrange 回 latent 空间
        p = self.patch_size
        latent_h, latent_w = grid_h * p, grid_w * p
        out = x.reshape(B, grid_h, grid_w, p, p, -1)
        out = out.permute(0, 5, 1, 3, 2, 4).contiguous()
        out = out.reshape(B, self.out_channels, latent_h, latent_w)

        # 取前 latent_dim（4）通道作为噪声预测
        return out[:, :self.latent_dim]
