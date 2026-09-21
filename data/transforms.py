"""
图像预处理：
- 内容图（I_p / I_t）：按桶尺寸等比缩放 + 白色居中填充（resize_and_pad_to_bucket）。
- 风格图（I_s）：平铺切分（tiling）——等比缩放到 height=224，不足 224 右填充，
  以 stride=168 滑动切出 224×224 小块，最后剩余不足 224 的块直接丢弃。
"""

from __future__ import annotations

import math
from typing import List, Tuple

from PIL import Image
import torch
from torchvision import transforms as T

from data.buckets import resize_and_pad_to_bucket, FillInfo

# ── 内容图：bucket 变换 ─────────────────────────────────────────────


def bucket_transform_pil(
    image: Image.Image,
    bucket_h: int,
    bucket_w: int,
) -> Tuple[torch.Tensor, FillInfo]:
    """
    将 PIL 图像等比例缩放并白色居中填充到桶尺寸，返回归一化 tensor 与填充信息。

    Args:
        image: 输入 PIL Image。
        bucket_h: 桶高度。
        bucket_w: 桶宽度。

    Returns:
        (tensor (3, bucket_h, bucket_w) in [-1, 1], info)
    """
    img = image.convert("RGB")
    canvas, info = resize_and_pad_to_bucket(img, bucket_h, bucket_w)
    tensor = T.ToTensor()(canvas)          # [0, 1]
    tensor = tensor * 2.0 - 1.0            # [-1, 1]
    return tensor, info


def bucket_transform_tensor(
    tensor: torch.Tensor,
    bucket_h: int,
    bucket_w: int,
) -> Tuple[torch.Tensor, FillInfo]:
    """
    tensor 版 bucket 变换（用于已有 tensor 的输入，如推理管道）。

    Args:
        tensor: (3, H, W) in [-1, 1]。
        bucket_h: 桶高度。
        bucket_w: 桶宽度。

    Returns:
        (tensor (3, bucket_h, bucket_w), info)
    """
    _, h, w = tensor.shape
    canvas, info = resize_and_pad_to_bucket(
        T.ToPILImage()(tensor * 0.5 + 0.5), bucket_h, bucket_w
    )
    out = T.ToTensor()(canvas) * 2.0 - 1.0
    return out, info


# ── 风格图：tiling 预处理 ───────────────────────────────────────────


def tile_image_to_tensors(
    image: Image.Image,
    tile_size: int = 224,
    stride: int = 168,
) -> List[torch.Tensor]:
    """
    将风格图平铺切分为 1×T 个 (3, tile_size, tile_size) 的归一化 tensor。

    流程（用户确认）：
    1. 等比例缩放到 height = tile_size；若缩放后 width < tile_size，右侧白色填充到 tile_size。
    2. 从左到右按 stride 滑动切出 tile_size×tile_size 小块。
    3. 最后一块剩余 width 不足 tile_size 时直接丢弃。

    Args:
        image: 输入 PIL Image。
        tile_size: 切块边长（224）。
        stride: 滑动步长（168）。

    Returns:
        tiles: (T, 3, tile_size, tile_size) in [-1, 1] 的列表。
    """
    img = image.convert("RGB")
    w, h = img.size

    # 等比缩放到 height = tile_size
    scale = tile_size / h
    new_w = max(tile_size, int(round(w * scale)))
    new_h = tile_size
    resized = img.resize((new_w, new_h), Image.BICUBIC)

    # 若 width 不足 tile_size，右侧白色填充
    if new_w < tile_size:
        canvas = Image.new("RGB", (tile_size, tile_size), (255, 255, 255))
        canvas.paste(resized, (0, 0))
        resized = canvas
        new_w = tile_size

    # 滑动切块：起点 i*stride，最后剩余不足 tile_size 丢弃
    tiles: List[torch.Tensor] = []
    x0 = 0
    while x0 + tile_size <= new_w:
        tile = resized.crop((x0, 0, x0 + tile_size, tile_size))
        tensor = T.ToTensor()(tile)   # [0, 1]
        tensor = tensor * 2.0 - 1.0   # [-1, 1]
        tiles.append(tensor)
        x0 += stride

    if not tiles:
        # 极端情况：宽度 < tile_size（缩放后被填充到 tile_size，正常不会走到这里）
        raise ValueError(f"tiling produced no tiles for image of size {img.size}")

    return tiles


def tiles_to_batch(
    tiles: List[torch.Tensor],
) -> torch.Tensor:
    """tile tensor 列表 → 批量 tensor (T, 3, tile_size, tile_size)。"""
    return torch.stack(tiles, dim=0)


# ── 旧接口（兼容保留）───────────────────────────────────────────────


def resize_and_pad(
    image: Image.Image,
    target_size: int = 256,
    fill_color: int = 255,
) -> Image.Image:
    """
    等比例缩放图像使长边 = target_size，白色填充短边居中（旧版正方形变换）。

    Args:
        image: 输入 PIL Image。
        target_size: 目标正方形边长。
        fill_color: 填充像素值（0-255），默认 255（白色）。

    Returns:
        正方形 PIL Image。
    """
    w, h = image.size
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = image.resize((new_w, new_h), Image.BICUBIC)

    canvas = Image.new("RGB", (target_size, target_size), (fill_color, fill_color, fill_color))
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))

    return canvas


def get_transform(image_size: int = 256) -> T.Compose:
    """
    返回旧版训练/推理用的正方形图像变换 pipeline（兼容保留）。

    - PIL → RGB → resize_and_pad → Tensor → [-1, 1] 归一化
    """
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Lambda(lambda img: resize_and_pad(img, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """将 [-1, 1] tensor 反归一化到 [0, 1]。"""
    return (tensor * 0.5 + 0.5).clamp(0, 1)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """将 [0, 1] 或 [-1, 1] tensor 转为 PIL Image。"""
    if tensor.min() < 0:
        tensor = denormalize(tensor)
    arr = (tensor * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)
