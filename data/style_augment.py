"""
风格图增广（阶段 B 度量学习用）。

**只做笔迹相关的扰动，不破坏公式内容**——因为阶段 B 的难负样本是「不同 writer 写的
同一条公式」，如果增广改变了符号结构，正负样本对就不可比了。

允许：
  ✅ 轻微平移 / 小角度旋转（±3°）/ 轻微剪切（±0.05）/ 轻微缩放
  ✅ 笔画粗细变化（形态学 dilate / erode）
  ✅ 墨色深浅变化（等比缩放墨迹深度）

禁止：
  ❌ 颜色抖动（公式图是二值的，颜色不是风格维度）
  ❌ 随机大裁剪（会裁掉公式结构 ⇒ 内容不可比）
  ❌ 水平翻转 / 竖直翻转（破坏符号语义）

**必须在 padding 之前逐张施加**（dataset worker 内）：先增广再拼 batch，
否则平移会把内容推进 padding 区，导致 style_mask 与真实内容错位。
"""

from __future__ import annotations

import math
import random
from typing import Optional

import torch
import torch.nn.functional as F


class StyleAugment:
    """
    对单张风格图 tensor (3, H, W) in [-1, 1] 施加随机笔迹增广。

    Args:
        translate:    平移幅度（相对图像宽/高的比例）
        rotate_deg:   最大旋转角度（度）
        shear:        最大剪切系数
        scale:        最大缩放抖动（高方向）
        morph_prob:   施加形态学（笔画加粗 / 变细）的概率
        ink_prob:     施加墨色深浅抖动的概率
        fg_threshold: 前景阈值（[-1,1] 域；默认 0.0）
        prob:         整体施加概率（1.0 = 总是增广）
    """

    def __init__(
        self,
        translate: float = 0.02,
        rotate_deg: float = 3.0,
        shear: float = 0.05,
        scale: float = 0.03,
        morph_prob: float = 0.15,
        ink_prob: float = 0.20,
        fg_threshold: float = 0.0,
        prob: float = 1.0,
    ):
        self.translate = float(translate)
        self.rotate_deg = float(rotate_deg)
        self.shear = float(shear)
        self.scale = float(scale)
        self.morph_prob = float(morph_prob)
        self.ink_prob = float(ink_prob)
        self.fg_threshold = float(fg_threshold)
        self.prob = float(prob)

    def __call__(self, x: torch.Tensor, rng: Optional[random.Random] = None) -> torch.Tensor:
        """
        Args:
            x:   (3, H, W) in [-1, 1]
            rng: 随机数发生器；None 用全局 random（训练时建议按样本播种保证可复现）
        Returns:
            (3, H, W) in [-1, 1]，尺寸不变
        """
        r = rng if rng is not None else random
        if self.prob < 1.0 and r.random() > self.prob:
            return x

        C, H, W = x.shape
        x4 = x.unsqueeze(0)

        # 1) 仿射：旋转 + 剪切 + 缩放 + 平移（区域外填 +1.0 = 背景色白）
        angle = r.uniform(-self.rotate_deg, self.rotate_deg)
        sh = r.uniform(-self.shear, self.shear)
        sc = r.uniform(1.0 - self.scale, 1.0 + self.scale)
        dx = r.uniform(-self.translate, self.translate) * W
        dy = r.uniform(-self.translate, self.translate) * H
        x4 = _affine(x4, angle, sh, sc, dx, dy)

        y = x4.squeeze(0)

        # 2) 笔画粗细（形态学 dilate / erode）
        if self.morph_prob > 0 and r.random() < self.morph_prob:
            mode = 0 if r.random() < 0.5 else 1
            y = _morph(y, self.fg_threshold, mode)

        # 3) 墨色深浅（等比缩放墨迹深度，不改变结构）
        if self.ink_prob > 0 and r.random() < self.ink_prob:
            k = r.uniform(0.85, 1.15)
            y = 1.0 - (1.0 - y) * k

        return y.clamp(-1.0, 1.0)


def _affine(
    x: torch.Tensor,
    angle_deg: float,
    shear: float,
    scale: float,
    dx: float,
    dy: float,
) -> torch.Tensor:
    """
    仿射变换（旋转 + 剪切 + 缩放 + 平移）。**区域外填充 +1.0（背景色白）**。

    Args:
        x: (1, C, H, W)
    Returns:
        (1, C, H, W)
    """
    _, C, H, W = x.shape
    device, dtype = x.device, x.dtype

    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    # 前向：y = s · R(a) · [[1, sh], [0, 1]] · x + t
    A = torch.tensor(
        [[ca, ca * shear - sa],
         [sa, sa * shear + ca]],
        dtype=torch.float32, device=device,
    ) * float(scale)
    A_inv = torch.inverse(A)                       # (2, 2)

    ys = torch.arange(H, dtype=torch.float32, device=device) - (H - 1) / 2.0
    xs = torch.arange(W, dtype=torch.float32, device=device) - (W - 1) / 2.0
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)

    # 输出像素 p（相对中心、像素单位）→ 输入坐标 A⁻¹(p - t)
    px = torch.stack([gx - dx, gy - dy], dim=-1)                 # (H, W, 2)
    src = px @ A_inv.transpose(0, 1).to(torch.float32)           # (H, W, 2)

    grid = torch.stack(
        [src[..., 0] / max(1e-6, (W - 1) / 2.0),
         src[..., 1] / max(1e-6, (H - 1) / 2.0)],
        dim=-1,
    ).unsqueeze(0).to(dtype)                                     # (1, H, W, 2)

    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    # padding_mode="zeros" 在区域外填 0（[-1,1] 下是灰色，会引入虚假边缘）
    # ⇒ 用同一 grid 采样一张全 1 图得到 inside mask，区域外补回 +1.0（白）
    ones = torch.ones_like(x)
    inside = F.grid_sample(ones, grid, mode="nearest", padding_mode="zeros", align_corners=True)
    return out + (1.0 - inside) * 1.0


def _morph(x: torch.Tensor, fg_threshold: float, mode: int) -> torch.Tensor:
    """
    形态学笔画粗细变化。mode=0 加粗（dilate），mode=1 变细（erode）。

    只改动「因形态学而翻转」的像素，其余像素保持原灰度（不全图二值化）。
    """
    ink = (x.mean(dim=0, keepdim=True) < fg_threshold).to(torch.float32)   # (1, H, W)
    k = 3
    if mode == 0:                                  # dilate：邻域有墨 ⇒ 该点有墨
        ink2 = F.max_pool2d(ink, k, stride=1, padding=k // 2)
    else:                                          # erode：邻域全是墨 ⇒ 该点有墨
        ink2 = 1.0 - F.max_pool2d(1.0 - ink, k, stride=1, padding=k // 2)
    ink2 = (ink2 > 0.5)

    new_ink = (ink2 & (ink < 0.5))[0]              # (H, W)
    new_bg = ((~ink2) & (ink > 0.5))[0]

    y = x.clone()
    if new_ink.any():
        y[:, new_ink] = -1.0
    if new_bg.any():
        y[:, new_bg] = 1.0
    return y
