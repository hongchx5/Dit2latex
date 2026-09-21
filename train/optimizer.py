"""优化器和学习率调度器：AdamW + 线性 warmup + 余弦衰减。"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    betas: tuple = (0.9, 0.999),
) -> AdamW:
    """创建 AdamW 优化器，仅优化 requires_grad=True 的参数。"""
    params = [p for p in model.parameters() if p.requires_grad]
    return AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)


def build_scheduler(
    optimizer: AdamW,
    total_steps: int,
    warmup_steps: int = 5000,
    min_lr: float = 1e-6,
) -> LambdaLR:
    """
    创建 warmup + cosine decay 调度器。

    前 warmup_steps：从 0 线性增加到 base_lr。
    warmup_steps 之后：余弦衰减到 min_lr。
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # 线性 warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # 余弦衰减
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            # 从 1.0（warmup 终点）衰减到 min_lr / base_lr
            return max(min_lr / optimizer.defaults["lr"], cosine_decay)

    return LambdaLR(optimizer, lr_lambda)
