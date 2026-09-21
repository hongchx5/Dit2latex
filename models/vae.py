"""
VAE 模块：潜在空间编码/解码。

- 正常模式：加载 Stable Diffusion VAE（f=8），冻结参数。
- Offline 测试模式：简单 AvgPool 下采样 + 1x1 Conv 和反向重建。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SDVAE(nn.Module):
    """Stable Diffusion VAE 封装器（正常模式）。"""

    def __init__(self, pretrained_path: str, device: str = "cuda"):
        super().__init__()
        try:
            from diffusers import AutoencoderKL
        except ImportError:
            raise ImportError("diffusers library required. Install with: pip install diffusers")

        self.vae = AutoencoderKL.from_pretrained(pretrained_path).to(device)
        self.vae.requires_grad_(False)
        self.vae.eval()
        self._latent_dim = 4
        self._f = 8  # 下采样倍率

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) → z: (B, 4, H/8, W/8), 使用 VAE 均值。"""
        posterior = self.vae.encode(x).latent_dist
        z = posterior.mode()  # 确定性编码（均值）
        return z * self.vae.config.scaling_factor

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, 4, H/8, W/8) → x: (B, 3, H, W)。"""
        z = z / self.vae.config.scaling_factor
        return self.vae.decode(z).sample

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def f(self) -> int:
        return self._f


class OfflineVAE(nn.Module):
    """Offline 测试用 VAE：简单采样 + 1x1 Conv。

    - Encoder: AvgPool2d(kernel=8, stride=8) → Conv2d(3→4, k=1)
    - Decoder: Conv2d(4→3, k=1) → Upsample(×8, nearest)
    """

    def __init__(self, latent_dim: int = 4, f: int = 8):
        super().__init__()
        self._latent_dim = latent_dim
        self._f = f

        self.encoder = nn.Sequential(
            nn.AvgPool2d(kernel_size=f, stride=f),
            nn.Conv2d(3, latent_dim, kernel_size=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, 3, kernel_size=1),
            nn.Upsample(scale_factor=f, mode="nearest"),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def f(self) -> int:
        return self._f


def build_vae(
    offline_test: bool = False,
    pretrained_path: str = "stabilityai/sd-vae-ft-ema",
    latent_dim: int = 4,
    f: int = 8,
    device: str = "cuda",
) -> nn.Module:
    """创建 VAE 实例。"""
    if offline_test:
        return OfflineVAE(latent_dim=latent_dim, f=f).to(device)
    else:
        return SDVAE(pretrained_path=pretrained_path, device=device)
