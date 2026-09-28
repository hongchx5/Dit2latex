"""优化器和学习率调度器：AdamW + 线性 warmup + 余弦衰减。"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def build_param_groups(
    model: nn.Module,
    lr: float = 1e-4,
    backbone: Optional[nn.Module] = None,
    backbone_lr_scale: float = 1.0,
    weight_decay: float = 1e-4,
) -> List[dict]:
    """
    构造参数组：backbone 用更小的学习率（阶段 C 联合微调时 backbone 需要 lr 缩放）。

    用「参数对象 id 集合」判定归属，因此在 DDP 包装下（DDP 不复制参数，
    用的是同一批 Parameter 对象）依然成立。

    Args:
        model: 完整模型（可以是 DDP 包装后的）。
        backbone: 风格编码器的 backbone；None 或 backbone_lr_scale == 1.0 时不分组。
        backbone_lr_scale: backbone 参数组相对主 lr 的缩放（如 0.5）。

    Returns:
        AdamW 用的 param_groups 列表（永远至少有一个非空组）。
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    if backbone is None or abs(backbone_lr_scale - 1.0) < 1e-12:
        return [{"params": trainable}]

    backbone_ids = {id(p) for p in backbone.parameters()}
    bb_params = [p for p in trainable if id(p) in backbone_ids]
    rest_params = [p for p in trainable if id(p) not in backbone_ids]

    groups: List[dict] = []
    if bb_params:
        groups.append({"params": bb_params, "lr": lr * backbone_lr_scale,
                       "weight_decay": weight_decay})
    if rest_params:
        groups.append({"params": rest_params, "lr": lr, "weight_decay": weight_decay})
    if not groups:
        groups = [{"params": trainable}]
    return groups


def build_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    betas: tuple = (0.9, 0.999),
    backbone: Optional[nn.Module] = None,
    backbone_lr_scale: float = 1.0,
) -> AdamW:
    """
    创建 AdamW 优化器，仅优化 requires_grad=True 的参数。

    backbone 不为 None 且 backbone_lr_scale != 1.0 时会拆成两个参数组
    （backbone 用 lr * backbone_lr_scale）。
    """
    groups = build_param_groups(
        model, lr=lr, backbone=backbone,
        backbone_lr_scale=backbone_lr_scale, weight_decay=weight_decay,
    )
    return AdamW(groups, lr=lr, weight_decay=weight_decay, betas=betas)


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
