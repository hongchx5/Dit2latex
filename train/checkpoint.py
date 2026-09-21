"""Checkpoint 管理：保存/恢复模型、优化器、调度器、EMA。"""

from __future__ import annotations

import os
from typing import Optional, Dict, Any
import torch
import torch.nn as nn


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    ema_state: Optional[Dict[str, torch.Tensor]] = None,
    step: int = 0,
    **extra,
) -> None:
    """保存完整训练状态。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint: Dict[str, Any] = {
        "step": step,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if ema_state is not None:
        checkpoint["ema_state"] = ema_state
    checkpoint.update(extra)

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    device: str = "cuda",
) -> dict:
    """恢复训练状态。返回 step 和 ema_state。"""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    step = checkpoint.get("step", 0)
    ema_state = checkpoint.get("ema_state", None)

    return {"step": step, "ema_state": ema_state}


class EMA:
    """指数移动平均 (Exponential Moving Average) 管理器。"""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self._register(model)

    def _register(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_val = self.decay * self.shadow[name] + (1 - self.decay) * param.data
                self.shadow[name] = new_val.clone().detach()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, state: Dict[str, torch.Tensor]):
        self.shadow = {k: v.clone() for k, v in state.items()}

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        """将 EMA 参数复制到模型上（推理时用）。"""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])
