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


# ── 新版风格图：整图固定高度 + 动态宽度预处理 ─────────────────────────


def preprocess_style_image(
    image: Image.Image,
    image_height: int = 64,
    max_width: int = 512,
    random_crop: bool = False,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    风格图预处理：等比缩放到固定高度，宽度动态；超出 max_width 时裁剪。

    Args:
        image: PIL Image（任意模式）。
        image_height: 目标高度 H。
        max_width: 最大允许宽度。
        random_crop: 是否随机裁剪一段（训练用）；False 为中心裁剪（推理用）。

    Returns:
        tensor: (3, H, W)，归一化到 [-1, 1]，白底 +1.0。
        original_size: (W, H) 缩放后的原始尺寸（裁剪前）。
    """
    import random
    img = image.convert("RGB")
    w, h = img.size
    scale = image_height / h
    new_w = max(1, int(round(w * scale)))
    new_h = image_height
    resized = img.resize((new_w, new_h), Image.BICUBIC)

    if new_w > max_width:
        if random_crop:
            start = random.randint(0, new_w - max_width)
        else:
            start = (new_w - max_width) // 2
        resized = resized.crop((start, 0, start + max_width, new_h))
        new_w = max_width

    tensor = T.ToTensor()(resized) * 2.0 - 1.0  # (3, H, W), ∈ [-1, 1]
    return tensor, (new_w, new_h)


def batch_style_images(
    tensors: List[torch.Tensor],
    fill_value: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    把高度相同、宽度不同的风格图 tensor batch 在一起，右侧 padding。

    Args:
        tensors: [(3, H, W_i)]，均已归一化到 [-1, 1]。
        fill_value: padding 填充值，必须用背景色 +1.0（白）。

    Returns:
        batched: (B, 3, H, W_max)
        valid_mask: (B, W_max) bool，True 表示真实图像区域。
    """
    if not tensors:
        raise ValueError("empty tensor list")
    H = tensors[0].shape[1]
    assert all(t.shape[1] == H for t in tensors), "all style tensors must have the same height"
    max_w = max(t.shape[2] for t in tensors)
    B = len(tensors)

    batched = torch.full((B, 3, H, max_w), fill_value, dtype=tensors[0].dtype)
    valid_mask = torch.zeros((B, max_w), dtype=torch.bool)

    for i, t in enumerate(tensors):
        w = t.shape[2]
        batched[i, :, :, :w] = t
        valid_mask[i, :w] = True

    return batched, valid_mask
