"""风格编码器训练损失：DINO、SupCon、ArcFace。"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    """
    DINO 自监督损失：student 与 teacher 输出的交叉熵，带 centering 和 temperature。
    """

    def __init__(
        self,
        out_dim: int = 65536,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        teacher_temp_warmup_epochs: int = 10,
        teacher_temp_final: float = 0.07,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.student_temp = student_temp
        self.teacher_temp_base = teacher_temp
        self.teacher_temp_final = teacher_temp_final
        self.teacher_temp_warmup_epochs = teacher_temp_warmup_epochs
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def get_teacher_temp(self, epoch: int) -> float:
        if self.teacher_temp_warmup_epochs <= 0 or epoch >= self.teacher_temp_warmup_epochs:
            return self.teacher_temp_final
        return self.teacher_temp_base + (self.teacher_temp_final - self.teacher_temp_base) * (epoch / self.teacher_temp_warmup_epochs)

    def forward(
        self,
        student_out: torch.Tensor,  # (B, N_student_crops, out_dim)
        teacher_out: torch.Tensor,  # (B, N_teacher_crops, out_dim)
        epoch: int = 0,
    ) -> torch.Tensor:
        """
        student_out / teacher_out 已经过各自的 projector 并 softmax 过或 logits。
        这里假设输入是 softmax probabilities。
        """
        B = student_out.shape[0]
        student_out = student_out / self.student_temp

        teacher_temp = self.get_teacher_temp(epoch)
        teacher_out = F.softmax((teacher_out - self.center) / teacher_temp, dim=-1)
        teacher_out = teacher_out.detach()

        # 每个 student crop 与所有 teacher crops 对齐
        total_loss = 0.0
        n_terms = 0
        for i in range(student_out.shape[1]):
            for j in range(teacher_out.shape[1]):
                loss = -(teacher_out[:, j] * F.log_softmax(student_out[:, i], dim=-1)).sum(-1).mean()
                total_loss += loss
                n_terms += 1
        total_loss = total_loss / max(1, n_terms)

        # 更新 center（EMA on teacher output）
        with torch.no_grad():
            batch_center = teacher_out.mean(dim=(0, 1), keepdim=True).mean(0, keepdim=True)  # (1, out_dim)
            self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

        return total_loss


class SupConLoss(nn.Module):
    """有监督对比损失（SupCon），支持一个 anchor 对应多个 positive/negative。"""

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) L2-normalized embeddings.
            labels: (B,) writer ids.
        """
        device = features.device
        B = features.shape[0]

        # similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature  # (B, B)

        # mask: positive pairs = same label, exclude self
        labels = labels.view(-1, 1)
        mask = (labels == labels.T).float().to(device)
        mask.fill_diagonal_(0)

        # log-sum-exp trick
        sim_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - sim_max.detach()
        exp_logits = torch.exp(logits)

        # denominator: all except self
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # numerator: sum of positive log probs
        mask_sum = mask.sum(dim=1)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_sum.clamp(min=1e-8)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        return loss


class ArcFaceLoss(nn.Module):
    """ArcFace 分类头 + 损失（阶段 B 辅助损失）。"""

    def __init__(
        self,
        in_features: int = 768,
        num_classes: int = 20,
        s: float = 30.0,
        m: float = 0.5,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) L2-normalized.
            labels: (B,) long.
        """
        # normalize weights
        weight = F.normalize(self.weight.to(features.device))
        cosine = F.linear(features, weight)  # (B, C)

        sine = torch.sqrt(1.0 - torch.clamp(cosine * cosine, min=0, max=1) + 1e-8)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.s * 0.01)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        return F.cross_entropy(output, labels)
