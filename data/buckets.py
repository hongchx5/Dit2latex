"""
FiT 可变分辨率尺寸预算工具。

取消固定桶列表，改为按 patch 预算动态计算内容图目标尺寸。
核心逻辑：
- 给定原图 (H, W)，在保持长宽比的前提下缩放；
- 缩放后尺寸必须对齐到 multiple（=8，满足 VAE 8 倍下采样）；
- 缩放后的 patch 数（按 content 图 16×16 像素计）不超过 max_patch；
- 优先放大（原图 patch 不足预算时），但不超过预算；
- 缩放后高度低于 min_height 的图被丢弃。

返回目标尺寸统一为 (H, W) 像素，且 H % multiple == 0、W % multiple == 0。
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from PIL import Image

import torch
from torchvision import transforms as T


def count_patches(H: int, W: int, patch_size: int = 16) -> int:
    """
    按 content 图像素计算 patch 数量。
    patch 数为整除：((H // patch_size) * (W // patch_size))。
    """
    if H <= 0 or W <= 0:
        return 0
    return (H // patch_size) * (W // patch_size)


def _round_to_multiple(x: int, multiple: int) -> int:
    """将整数对齐到 multiple 的最近倍数，至少为 multiple。"""
    if x <= 0:
        return multiple
    return int(round(x / multiple)) * multiple


def fit_size_to_patch_budget(
    H: int,
    W: int,
    max_patch: int = 192,
    patch_size: int = 16,
    multiple: int = 8,
) -> Optional[Tuple[int, int]]:
    """
    按 FiT patch 预算计算内容图目标尺寸 (H', W')。

    规则：
    1. H' 和 W' 都是 multiple 的倍数；
    2. W' / H' ≈ W / H（保持宽高比，由 H' 等比推导 W' 后再对齐）；
    3. count_patches(H', W', patch_size) <= max_patch；
    4. 在满足约束的前提下取最大尺寸。

    若 max_patch <= 0 或原图尺寸非正，返回 None。
    若找不到满足 min_height 约束的结果，调用方应丢弃该图。

    Args:
        H: 原图高度（像素）。
        W: 原图宽度（像素）。
        max_patch: 单图 patch 数上限。
        patch_size: 像素级 patch 尺寸（content 图 16×16 = 1 patch）。
        multiple: 目标尺寸对齐粒度（默认 8，适配 VAE 下采样）。

    Returns:
        (H', W') 或 None。
    """
    if max_patch <= 0 or H <= 0 or W <= 0:
        return None

    def size_at(hp: int) -> Tuple[int, int]:
        # 由 H' 等比推导理想 W'，再对齐到 multiple
        wp = int(round(W * hp / float(H)))
        wp = _round_to_multiple(wp, multiple)
        # 保证至少 1 个 patch 宽度
        min_w_pixels = max(multiple, patch_size)
        wp = max(wp, min_w_pixels)
        return hp, wp

    # 上界：按面积预算 + 宽高比估算一个安全的 H 上限
    upper = int(math.sqrt(max_patch * patch_size * patch_size * H / float(W))) + multiple * 2
    upper = max(multiple, _round_to_multiple(upper, multiple))

    lo, hi, best = multiple, upper, None
    while lo <= hi:
        mid = ((lo + hi) // 2 // multiple) * multiple
        if mid < multiple:
            mid = multiple
        hp, wp = size_at(mid)
        if count_patches(hp, wp, patch_size) <= max_patch:
            best = (hp, wp)
            lo = mid + multiple
        else:
            hi = mid - multiple

    return best


def resize_to_fit(
    image: Image.Image,
    target_h: int,
    target_w: int,
) -> Image.Image:
    """
    将图像等比缩放到目标尺寸。允许轻微形变（误差由 multiple 对齐导致）。
    放大用 BICUBIC，缩小用 BILINEAR。

    Args:
        image: 输入 PIL Image（RGB）。
        target_h: 目标高度（像素）。
        target_w: 目标宽度（像素）。

    Returns:
        缩放后的 PIL Image(target_w, target_h)。
    """
    w, h = image.size
    # 按面积判断放大/缩小
    if target_h * target_w >= h * w:
        resample = Image.BICUBIC
    else:
        resample = Image.BILINEAR
    return image.resize((target_w, target_h), resample)


def fit_transform_pil(
    image: Image.Image,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """
    内容图 FiT 预处理：按目标尺寸等比缩放并归一化到 [-1, 1]。

    Args:
        image: 输入 PIL Image。
        target_h: 目标高度。
        target_w: 目标宽度。

    Returns:
        tensor (3, target_h, target_w) in [-1, 1]。
    """
    img = image.convert("RGB")
    resized = resize_to_fit(img, target_h, target_w)
    tensor = T.ToTensor()(resized)  # [0, 1]
    tensor = tensor * 2.0 - 1.0     # [-1, 1]
    return tensor


def fit_transform_tensor(
    tensor: torch.Tensor,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """
    tensor 版 FiT 变换。

    Args:
        tensor: (3, H, W) in [-1, 1]。
        target_h: 目标高度。
        target_w: 目标宽度。

    Returns:
        tensor (3, target_h, target_w) in [-1, 1]。
    """
    _, h, w = tensor.shape
    resample = "bicubic" if target_h * target_w >= h * w else "bilinear"
    tensor = tensor.unsqueeze(0)
    tensor = torch.nn.functional.interpolate(
        tensor, size=(target_h, target_w), mode=resample, align_corners=False,
    )
    return tensor.squeeze(0)


# 旧接口兼容：保留符号但改为使用 fit_transform_pil 的语义
# 不再返回 FillInfo，因为不再做居中填充

def bucket_transform_pil(
    image: Image.Image,
    bucket_h: int,
    bucket_w: int,
) -> Tuple[torch.Tensor, None]:
    return fit_transform_pil(image, bucket_h, bucket_w), None


def bucket_transform_tensor(
    tensor: torch.Tensor,
    bucket_h: int,
    bucket_w: int,
) -> Tuple[torch.Tensor, None]:
    return fit_transform_tensor(tensor, bucket_h, bucket_w), None


def bucket_to_latent(
    bucket_h: int,
    bucket_w: int,
    vae_f: int = 8,
) -> Tuple[int, int]:
    """桶像素尺寸 → VAE latent 尺寸 (latent_h, latent_w)。"""
    return bucket_h // vae_f, bucket_w // vae_f


def bucket_to_grid(
    bucket_h: int,
    bucket_w: int,
    vae_f: int = 8,
    patch_size: int = 2,
) -> Tuple[int, int]:
    """桶像素尺寸 → DiT latent patch grid (grid_h, grid_w)。"""
    latent_h, latent_w = bucket_to_latent(bucket_h, bucket_w, vae_f)
    return latent_h // patch_size, latent_w // patch_size


def bucket_tokens(
    bucket_h: int,
    bucket_w: int,
    vae_f: int = 8,
    patch_size: int = 2,
) -> int:
    """给定像素尺寸对应的 DiT patch token 数量。"""
    grid_h, grid_w = bucket_to_grid(bucket_h, bucket_w, vae_f, patch_size)
    return grid_h * grid_w

