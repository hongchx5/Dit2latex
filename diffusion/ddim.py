"""
DDIM (Denoising Diffusion Implicit Models) 采样器。

实现确定性反向去噪过程，支持 classifier-free guidance。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from diffusion.noise_schedule import NoiseSchedule


@torch.no_grad()
def ddim_sample(
    pipeline: nn.Module,
    noise_schedule: NoiseSchedule,
    z_t: torch.Tensor,
    z_p: torch.Tensor,
    f_s_pooled: torch.Tensor,
    f_s_seq: torch.Tensor,
    caption_seq: torch.Tensor,
    caption_mask: torch.Tensor,
    num_steps: int = 50,
    eta: float = 0.0,
    cfg_scale: float = 3.0,
) -> torch.Tensor:
    """
    DDIM 采样（CFG：无条件分支同时置零内容 z_p、风格、caption）。

    Args:
        pipeline:       DiTtolatexPipeline（或兼容的预测接口）。
        noise_schedule: 噪声调度对象。
        z_t:            当前噪声潜在 (B, 4, H, W)。
        z_p:            内容条件潜在 (B, 4, H, W)。
        f_s_pooled:     风格池化特征 (B, 768)。
        f_s_seq:        风格特征序列 (B, M, 768)。
        caption_seq:    latex caption 序列特征 (B, L, 768)。
        caption_mask:   caption padding mask (B, L) bool。
        num_steps:      DDIM 步数。
        eta:            DDIM η（0=确定性，1=DDPM）。
        cfg_scale:      CFG 引导系数。

    Returns:
        z_0: 去噪后的潜在表示 (B, 4, H, W)。
    """
    device = z_t.device
    T = noise_schedule.T

    # DDIM 采样步长序列
    times = torch.linspace(T - 1, 0, num_steps, dtype=torch.long, device=device)
    time_pairs = list(zip(times[:-1], times[1:])) + [(times[-1], -1)]

    z = z_t
    for t_now, t_next in time_pairs:
        t_batch = torch.full((z.shape[0],), t_now, device=device, dtype=torch.long)
        z_t_prime = torch.cat([z, z_p], dim=1)  # (B, 8, H, W)

        # 有条件预测
        eps_cond = pipeline.predict_noise(
            z_t_prime, f_s_pooled, f_s_seq, caption_seq, caption_mask, t_now, t_batch
        )

        # 无条件预测：内容 z_p=0、风格=0、caption=0（整体条件 CFG）
        z_t_prime_uncond = torch.cat([z, torch.zeros_like(z_p)], dim=1)
        f_zero = torch.zeros_like(f_s_pooled)
        f_zero_seq = torch.zeros_like(f_s_seq)
        caption_zero = torch.zeros_like(caption_seq) if caption_seq is not None else None
        eps_uncond = pipeline.predict_noise(
            z_t_prime_uncond, f_zero, f_zero_seq, caption_zero, caption_mask, t_now, t_batch
        )

        # CFG
        eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

        # DDIM 更新
        alpha_bar_t = noise_schedule.alphas_cumprod[t_now].view(-1, 1, 1, 1)

        # 预测 z_0
        z_0_pred = (z - (1 - alpha_bar_t).sqrt() * eps) / alpha_bar_t.sqrt()

        if t_next < 0:
            z = z_0_pred
            break

        alpha_bar_next = noise_schedule.alphas_cumprod[t_next].view(-1, 1, 1, 1)

        # 确定性方向
        sigma_t = eta * ((1 - alpha_bar_next) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_next)).sqrt()
        c1 = alpha_bar_next.sqrt()
        c2 = (1 - alpha_bar_next - sigma_t ** 2).sqrt()

        z = c1 * z_0_pred + c2 * eps
        if eta > 0:
            z = z + sigma_t * torch.randn_like(z)

    return z
