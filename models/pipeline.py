"""
训练/推理共用管线：将 VAE、风格编码器、caption 编码器、DiT、扩散过程串联。

- 训练：接收 (I_p, I_s, I_t, caption_ids, caption_mask) → 计算损失
- 推理：接收 (I_p, I_s, caption_ids, caption_mask) → DDIM 采样 → I_g

条件注入：
  - 内容：z_p（打印体 latent）→ 通道拼接进 z_t_prime + DiT 内部 ControlNet 逐层注入
  - 风格：f_s_seq（4 聚合向量）→ Style Cross-Attention；f_s_pooled → AdaLN
  - caption：latex token id → CaptionEncoder → 序列 → Caption Cross-Attention

CFG dropout（训练）：统一 mask 同时丢弃风格、内容（z_p）、caption，与推理无条件分支一致。
"""

from __future__ import annotations

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.noise_schedule import NoiseSchedule
from losses.diffusion_loss import simple_diffusion_loss
from losses.perceptual_loss import PerceptualLoss


class DiTtolatexPipeline(nn.Module):
    """端到端管线。"""

    def __init__(
        self,
        vae: nn.Module,
        style_encoder: nn.Module,
        caption_encoder: nn.Module,
        dit: nn.Module,
        noise_schedule: NoiseSchedule,
        perceptual_loss: Optional[PerceptualLoss],   # weight=0 时可为 None（不构建 VGG）
        perceptual_loss_weight: float = 0.1,
        cfg_dropout_rate: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        self.vae = vae
        self.style_encoder = style_encoder
        self.caption_encoder = caption_encoder
        self.dit = dit
        self.noise_schedule = noise_schedule
        self.perceptual_loss = perceptual_loss
        self.perceptual_loss_weight = perceptual_loss_weight
        self.cfg_dropout_rate = cfg_dropout_rate
        self.device = device

    # ── 训练前向 ─────────────────────────────────────────────────────

    def forward(
        self,
        I_p: torch.Tensor,
        I_s,
        I_t: torch.Tensor,
        caption_ids: torch.Tensor,
        caption_mask: torch.Tensor,
    ) -> dict:
        """
        训练前向：计算扩散损失和感知损失。

        Args:
            I_p:          打印体图像 (B, 3, bucket_h, bucket_w)（同桶 batch）
            I_s:          风格参考图路径列表 (B,)，或 tensor 列表（offline 模式）
            I_t:          目标手写图像 (B, 3, bucket_h, bucket_w)
            caption_ids:  (B, L) long，latex token id
            caption_mask: (B, L) bool，True=padding

        Returns:
            dict with keys: 'loss', 'diff_loss', 'percep_loss'
        """
        B = I_p.shape[0]

        # 1. VAE 编码（动态分辨率）
        z_p = self.vae.encode(I_p)   # (B, 4, latent_h, latent_w)
        z_0 = self.vae.encode(I_t)   # (B, 4, latent_h, latent_w) — 目标潜在

        # 2. 风格编码（tiling + 4-query 聚合）
        f_s_seq, f_s_pooled = self.style_encoder.encode(I_s)  # (B, M, 768), (B, 768)

        # 3. caption 编码（caption_encoder 为 None 时跳过，caption_seq=None）
        caption_seq = None
        if self.caption_encoder is not None:
            caption_seq = self.caption_encoder(caption_ids, caption_mask)  # (B, L, 768)

        # 4. CFG dropout：统一 mask 同时丢弃风格、内容（z_p）、caption
        if self.training and self.cfg_dropout_rate > 0:
            mask = torch.rand(B, 1, device=I_p.device) > self.cfg_dropout_rate
            f_s_pooled = f_s_pooled * mask.float()
            f_s_seq = f_s_seq * mask.float().unsqueeze(-1)
            z_p = z_p * mask.float().view(B, 1, 1, 1)
            if caption_seq is not None:
                caption_seq = caption_seq * mask.float().unsqueeze(-1)

        # 5. 扩散：采样时间步 + 加噪
        t = self.noise_schedule.sample_timesteps(B, device=I_p.device)
        noise = torch.randn_like(z_0)
        z_t = self.noise_schedule.q_sample(z_0, t, noise)

        # 6. 拼接 z_t' = [z_t; z_p]
        z_t_prime = torch.cat([z_t, z_p], dim=1)  # (B, 8, latent_h, latent_w)

        # 7. DiT 预测噪声（含 ControlNet 内容注入 + caption cross-attn）
        noise_pred = self.dit(z_t_prime, f_s_seq, f_s_pooled, caption_seq, caption_mask, t)

        # 8. 扩散损失
        diff_loss = simple_diffusion_loss(noise_pred, noise)

        # 9. 感知损失（weight=0 或未提供损失模块时跳过 decode + VGG，避免 0*NaN 污染 total_loss）
        if self.perceptual_loss_weight > 0 and self.perceptual_loss is not None:
            alpha_bar = self.noise_schedule.alphas_cumprod[t].view(B, 1, 1, 1)
            safe_mask = (alpha_bar > 0.05).float().view(B)
            safe_count = safe_mask.sum()
            if safe_count > 0:
                z_0_pred = (z_t - (1 - alpha_bar).sqrt() * noise_pred) / alpha_bar.sqrt().clamp(min=1e-4)
                I_g_pred = self.vae.decode(z_0_pred)
                percep_loss = self.perceptual_loss(I_g_pred, I_p)
            else:
                percep_loss = torch.tensor(0.0, device=diff_loss.device)
        else:
            percep_loss = torch.tensor(0.0, device=diff_loss.device)

        # 总损失
        total_loss = diff_loss + self.perceptual_loss_weight * percep_loss

        return {
            "loss": total_loss,
            "diff_loss": diff_loss,
            "percep_loss": percep_loss,
        }

    # ── 无梯度解码辅助 ───────────────────────────────────────────────

    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """将潜在表示 z 解码为像素图像。"""
        return self.vae.decode(z)

    # ── 条件编码辅助（推理时用）─────────────────────────────────────

    def encode_content(self, I_p: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(I_p)

    def encode_style(self, I_s):
        return self.style_encoder.encode(I_s)

    def encode_caption(self, caption_ids: torch.Tensor, caption_mask: torch.Tensor):
        """caption 编码；caption_encoder 为 None 时返回 None。"""
        if self.caption_encoder is None:
            return None
        return self.caption_encoder(caption_ids, caption_mask)

    # ── 单步去噪（DDIM 用）──────────────────────────────────────────

    def predict_noise(
        self,
        z_t_prime: torch.Tensor,
        f_s_pooled: torch.Tensor,
        f_s_seq: torch.Tensor,
        caption_seq: torch.Tensor,
        caption_mask: torch.Tensor,
        t_scalar: float,
        t_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        预测噪声 ε̂（CFG 由 ddim_sample 在外部组织）。

        Args:
            z_t_prime:    (B, 8, latent_h, latent_w)
            f_s_pooled:   (B, 768)
            f_s_seq:      (B, M, 768)
            caption_seq:  (B, L, 768)
            caption_mask: (B, L) bool
            t_scalar:     时间步标量（保留兼容）
            t_tensor:     (B,) 时间步

        Returns:
            noise_pred: (B, 4, latent_h, latent_w)
        """
        return self.dit(z_t_prime, f_s_seq, f_s_pooled, caption_seq, caption_mask, t_tensor)
