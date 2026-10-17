"""
图像预处理：
- 内容图（I_p / I_t）：FiT 动态缩放 → 目标尺寸 (H', W')，归一化到 [-1, 1]。
- 风格图（I_s）：平铺切分（tiling）——等比缩放到 height=224，不足 224 右填充，
  以 stride=168 滑动切出 224×224 小块，最后剩余不足 224 的块直接丢弃。
"""

from __future__ import annotations

from typing import List, Tuple

from PIL import Image
import torch
from torchvision import transforms as T

from data.buckets import fit_transform_pil


def bucket_transform_pil(
    image: Image.Image,
    bucket_h: int,
    bucket_w: int,
) -> Tuple[torch.Tensor, None]:
    """兼容旧接口的 FiT 内容图变换（不再填充，不再返回 FillInfo）。"""
    tensor = fit_transform_pil(image, bucket_h, bucket_w)
    return tensor, None


def bucket_transform_tensor(
    tensor: torch.Tensor,
    bucket_h: int,
    bucket_w: int,
) -> Tuple[torch.Tensor, None]:
    """tensor 版兼容接口。"""
    from data.buckets import fit_transform_tensor
    out = fit_transform_tensor(tensor, bucket_h, bucket_w)
    return out, None


# ── 风格图：tiling 预处理 ───────────────────────────────────────────


def tile_image_to_tensors(
    image: Image.Image,
    tile_size: int = 224,
    stride: int = 168,
) -> List[torch.Tensor]:
    """
    将风格图平铺切分为 1×T 个 (3, tile_size, tile_size) 的归一化 tensor。
    """
    img = image.convert("RGB")
    w, h = img.size

    scale = tile_size / h
    new_w = max(tile_size, int(round(w * scale)))
    new_h = tile_size
    resized = img.resize((new_w, new_h), Image.BICUBIC)

    if new_w < tile_size:
        canvas = Image.new("RGB", (tile_size, tile_size), (255, 255, 255))
        canvas.paste(resized, (0, 0))
        resized = canvas
        new_w = tile_size

    tiles: List[torch.Tensor] = []
    x0 = 0
    while x0 + tile_size <= new_w:
        tile = resized.crop((x0, 0, x0 + tile_size, tile_size))
        tensor = T.ToTensor()(tile)
        tensor = tensor * 2.0 - 1.0
        tiles.append(tensor)
        x0 += stride

    if not tiles:
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
    """等比例缩放图像使长边 = target_size，白色填充短边居中（旧版正方形变换）。"""
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
    """返回旧版训练/推理用的正方形图像变换 pipeline。"""
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
