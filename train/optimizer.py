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
    style_backbone_lr_scale: float = 1.0,
) -> AdamW:
    """创建 AdamW 优化器，仅优化 requires_grad=True 的参数。

    阶段 C 时，风格编码器 backbone 使用 style_backbone_lr_scale * lr 的学习率，
    聚合头及其它模块使用 lr。
    """
    style_backbone_params = []
    other_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "style_encoder.backbone" in name:
            style_backbone_params.append(p)
        else:
            other_params.append(p)

    param_groups = [
        {"params": other_params, "lr": lr, "weight_decay": weight_decay},
    ]
    if style_backbone_params:
        param_groups.append(
            {"params": style_backbone_params, "lr": lr * style_backbone_lr_scale, "weight_decay": weight_decay}
        )

    return AdamW(param_groups, betas=betas)


def build_scheduler(
    optimizer: AdamW,
    total_steps: int,
    warmup_steps: int = 5000,
    min_lr: float = 1e-6,
) -> LambdaLR:
    """创建 warmup + cosine decay 调度器，支持多 param group 不同 base_lr。"""

    base_lrs = [g["lr"] for g in optimizer.param_groups]

    def _make_lambda(base_lr: float):
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            # scale such that each group decays from base_lr to min_lr
            return min_lr / base_lr + (1 - min_lr / base_lr) * cosine_decay
        return lr_lambda

    return LambdaLR(optimizer, [_make_lambda(base) for base in base_lrs])
