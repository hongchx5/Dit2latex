"""
动态分桶（aspect-ratio bucketing）工具。

桶尺寸是物理图（像素）尺寸，格式统一为 (H, W)。
默认桶集合覆盖宽图分布，所有边长满足 8×（VAE 下采样）与 2×（patch）整除：
    (64, 1024), (96, 512), (128, 512), (128, 384), (256, 256)
其中 96 是 32 的倍数（非严格 64 倍数），但 8 和 16 的整除均满足。

预处理语义（用户确认）：
- "恰好容入"：等比缩放后至少一条边等于桶对应边，另一条边不超过桶边；
- 填充：内容居中，对称填充；水平方向奇数余量右侧多一列，垂直方向奇数余量下侧多一行；
- 小于桶的图不放大（scale 取 min 比例，只缩小或保持原尺寸）。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from PIL import Image

# (H, W) —— 注意顺序：高在前、宽在后
DEFAULT_BUCKETS: List[Tuple[int, int]] = [
    (64, 1024),
    (96, 512),
    (128, 512),
    (128, 384),
    (256, 256),
]

FillInfo = Tuple[int, int, int, int, float]
# (offset_x, offset_y, new_w, new_h, scale) —— 用于推理时裁剪回内容区域


def pick_bucket(
    w: int,
    h: int,
    buckets: Sequence[Tuple[int, int]] = DEFAULT_BUCKETS,
) -> Tuple[int, int]:
    """
    选择与图片宽高比最匹配的桶（填充面积最小）。

    Args:
        w: 原图宽度（像素）。
        h: 原图高度（像素）。
        buckets: 桶集合，每项为 (bucket_h, bucket_w)。

    Returns:
        (bucket_h, bucket_w)
    """
    best_bucket: Optional[Tuple[int, int]] = None
    best_pad_area = float("inf")

    for bucket_h, bucket_w in buckets:
        # contain 缩放：至少一条边恰好等于桶边
        scale = min(bucket_h / h, bucket_w / w)
        new_h = h * scale
        new_w = w * scale
        pad_area = bucket_h * bucket_w - new_w * new_h
        if pad_area < best_pad_area:
            best_pad_area = pad_area
            best_bucket = (bucket_h, bucket_w)

    assert best_bucket is not None
    return best_bucket


def resize_and_pad_to_bucket(
    image: Image.Image,
    bucket_h: int,
    bucket_w: int,
    fill_color: int = 255,
    resample: int = Image.BICUBIC,
) -> Tuple[Image.Image, FillInfo]:
    """
    等比缩放图像使某条边恰好容入桶，剩余方向白色居中对称填充。

    Args:
        image: 输入 PIL Image（RGB）。
        bucket_h: 桶高度（像素）。
        bucket_w: 桶宽度（像素）。
        fill_color: 填充像素值，默认 255（白色）。
        resample: 缩放插值方式。

    Returns:
        (canvas, info)，其中 canvas 为 (bucket_h, bucket_w) 的填充后图像，
        info = (offset_x, offset_y, new_w, new_h, scale) 供推理裁剪使用。
    """
    w, h = image.size
    scale = min(bucket_h / h, bucket_w / w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # 保证不超出桶
    new_w = min(new_w, bucket_w)
    new_h = min(new_h, bucket_h)

    resized = image.resize((new_w, new_h), resample)

    canvas = Image.new("RGB", (bucket_w, bucket_h), (fill_color, fill_color, fill_color))

    # 居中对称填充：奇数余量右侧/下侧多一列/行
    pad_x = bucket_w - new_w
    pad_y = bucket_h - new_h
    offset_x = pad_x // 2          # 左留白；pad_x 为奇数时右多一列
    offset_y = pad_y // 2          # 上留白；pad_y 为奇数时下多一行

    canvas.paste(resized, (offset_x, offset_y))

    info: FillInfo = (offset_x, offset_y, new_w, new_h, scale)
    return canvas, info


def crop_to_content(
    image: Image.Image,
    info: FillInfo,
) -> Image.Image:
    """
    按 FillInfo 从填充后的桶图像中裁出内容区域（推理时还原原始宽高比）。

    Args:
        image: 桶尺寸的图像（生成结果）。
        info: resize_and_pad_to_bucket 返回的填充信息。

    Returns:
        内容区域图像（宽高比与原图一致）。
    """
    offset_x, offset_y, new_w, new_h, _ = info
    return image.crop((offset_x, offset_y, offset_x + new_w, offset_y + new_h))


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
    """桶像素尺寸 → DiT patch token 网格 (grid_h, grid_w)。"""
    latent_h, latent_w = bucket_to_latent(bucket_h, bucket_w, vae_f)
    return latent_h // patch_size, latent_w // patch_size


def bucket_tokens(
    bucket_h: int,
    bucket_w: int,
    vae_f: int = 8,
    patch_size: int = 2,
) -> int:
    """桶对应的 patch token 数量。"""
    grid_h, grid_w = bucket_to_grid(bucket_h, bucket_w, vae_f, patch_size)
    return grid_h * grid_w
