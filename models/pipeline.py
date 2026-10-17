"""训练/推理共用管线：将 VAE、风格编码器、caption 编码器、DiT、扩散过程串联。"""

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
            I_p:          打印体图像 (B, 3, bucket_h, bucket_w)
            I_s:          风格参考图 (B, 3, H, W) + valid_mask (B, W)，或旧版路径列表
            I_t:          目标手写图像 (B, 3, bucket_h, bucket_w)
            caption_ids:  (B, L) long
            caption_mask: (B, L) bool，True=padding

        Returns:
            dict with keys: 'loss', 'diff_loss', 'percep_loss'
        """
        B = I_p.shape[0]

        # 1. VAE 编码
        z_p = self.vae.encode(I_p)
        z_0 = self.vae.encode(I_t)

        # 2. 风格编码
        if isinstance(I_s, tuple):
            f_s_seq, f_s_pooled = self.style_encoder.encode(I_s[0], I_s[1])
        else:
            f_s_seq, f_s_pooled = self.style_encoder.encode(I_s)

        # 3. caption 编码
        caption_seq = None
        if self.caption_encoder is not None:
            caption_seq = self.caption_encoder(caption_ids, caption_mask)

        # 4. CFG dropout：统一 mask 同时丢弃风格、内容、caption
        if self.training and self.cfg_dropout_rate > 0:
            mask = torch.rand(B, 1, device=I_p.device) > self.cfg_dropout_rate
            f_s_pooled = f_s_pooled * mask.float()
            f_s_seq = f_s_seq * mask.float().unsqueeze(-1)
            z_p = z_p * mask.float().view(B, 1, 1, 1)
            if caption_seq is not None:
                caption_seq = caption_seq * mask.float().unsqueeze(-1)

        # 5. 扩散
        t = self.noise_schedule.sample_timesteps(B, device=I_p.device)
        noise = torch.randn_like(z_0)
        z_t = self.noise_schedule.q_sample(z_0, t, noise)

        z_t_prime = torch.cat([z_t, z_p], dim=1)

        noise_pred = self.dit(z_t_prime, f_s_seq, f_s_pooled, caption_seq, caption_mask, t)

        diff_loss = simple_diffusion_loss(noise_pred, noise)

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

        total_loss = diff_loss + self.perceptual_loss_weight * percep_loss

        return {
            "loss": total_loss,
            "diff_loss": diff_loss,
            "percep_loss": percep_loss,
        }

    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """将潜在表示 z 解码为像素图像。"""
        return self.vae.decode(z)

    def encode_content(self, I_p: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(I_p)

    def encode_style(self, I_s):
        """推理时调用；接受 (tensor, valid_mask) 或路径 / PIL / list。"""
        if isinstance(I_s, tuple):
            return self.style_encoder.encode(I_s[0], I_s[1])
        return self.style_encoder.encode(I_s)

    def encode_caption(self, caption_ids: torch.Tensor, caption_mask: torch.Tensor):
        """caption 编码；caption_encoder 为 None 时返回 None。"""
        if self.caption_encoder is None:
            return None
        return self.caption_encoder(caption_ids, caption_mask)
