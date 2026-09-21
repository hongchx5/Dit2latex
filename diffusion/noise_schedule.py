"""
噪声调度：余弦 β 调度 + 扩散过程 q(x_t | x_0) 的工具函数。
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


class NoiseSchedule:
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = "cosine",
        device: str = "cuda",
    ):
        self.num_timesteps = num_timesteps

        if beta_schedule == "cosine":
            betas = self._cosine_beta_schedule(num_timesteps)
        elif beta_schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, num_timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        self.register(betas, device)

    def register(self, betas: torch.Tensor, device: str):
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)

        # 预计算 √ᾱ_t, √(1-ᾱ_t) 等
        self.sqrt_alphas_cumprod = alphas_cumprod.sqrt().to(device)
        self.sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt().to(device)

    @staticmethod
    def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
        """cosine schedule（参考 improved DDPM）。"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.02)

    def sample_timesteps(self, batch_size: int, device: str = "cuda") -> torch.LongTensor:
        """均匀采样 [0, T-1] 的时间步。"""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)

    def q_sample(self, x0: torch.Tensor, t: torch.LongTensor, noise: torch.Tensor) -> torch.Tensor:
        """q(x_t | x_0) 前向扩散加噪。"""
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise

    @property
    def T(self) -> int:
        return self.num_timesteps
