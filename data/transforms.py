"""
图像预处理：
- 内容图（I_p / I_t）：
  - 静态分桶：按桶尺寸等比缩放 + 白色居中填充（bucket_transform_pil）；
  - FiT token 预算：缩放到自身 fit 尺寸后居中贴进 canvas（fit_transform_pil），
    返回内容区在 token 网格中的位置，供 padding mask 使用。
- 风格图（I_s）：平铺切分（tiling）——等比缩放到 height=224，不足 224 右填充，
  以 stride=168 滑动切出 224×224 小块，最后剩余不足 224 的块直接丢弃。
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

from PIL import Image
import torch
from torchvision import transforms as T

from data.buckets import (
    resize_and_pad_to_bucket,
    FillInfo,
    ContentRect,
    aligned_offset,
    floor_to_unit,
    token_unit,
)

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


def fit_transform_pil(
    image: Image.Image,
    fit_h: int,
    fit_w: int,
    canvas_h: Optional[int] = None,
    canvas_w: Optional[int] = None,
    vae_f: int = 8,
    patch_size: int = 2,
    fill_color: int = 255,
    resample: int = Image.BICUBIC,
) -> Tuple[torch.Tensor, FillInfo, ContentRect]:
    """
    FiT token 预算版内容图变换：等比缩放到 fit 尺寸，再居中贴进 canvas。

    与 bucket_transform_pil 的区别：后者是「contain 缩放填满桶」，会把小图放大；
    这里是「先按 token 预算算出每张图自己的 fit 尺寸并精确缩放到该尺寸」，
    canvas 只是为了让同 batch 的 tensor 形状一致，多出来的部分是 padding
    （由 token_mask 屏蔽，不进注意力、不算损失）。

    推理时 canvas 缺省 = fit 尺寸 ⇒ 无 padding，模型看到的就是纯内容网格。

    Args:
        image:    输入 PIL Image。
        fit_h, fit_w: fit_size_to_token_budget 算出的目标内容尺寸（unit 的倍数）。
        canvas_h, canvas_w: 同 batch 统一的画布尺寸；None 表示等于 fit（无 padding）。
        vae_f, patch_size: 用于把像素坐标换算成 token 网格坐标。

    Returns:
        (tensor (3, canvas_h, canvas_w) in [-1, 1], info, rect)
        rect = (canvas_gh, canvas_gw, row0, col0, content_gh, content_gw)
    """
    unit = token_unit(vae_f, patch_size)
    canvas_h = fit_h if canvas_h is None else canvas_h
    canvas_w = fit_w if canvas_w is None else canvas_w

    img = image.convert("RGB")
    w, h = img.size

    # 1) 缩放到 fit 尺寸；若 fit 比 canvas 还大（极端分箱），按 contain 收一点
    scale = min(1.0, canvas_h / float(fit_h), canvas_w / float(fit_w))
    new_h = floor_to_unit(fit_h * scale, unit)
    new_w = floor_to_unit(fit_w * scale, unit)
    resized = img.resize((new_w, new_h), resample)

    # 2) 居中（按 unit 对齐）贴进 canvas
    canvas = Image.new("RGB", (canvas_w, canvas_h), (fill_color, fill_color, fill_color))
    offset_x = aligned_offset(canvas_w, new_w, unit)
    offset_y = aligned_offset(canvas_h, new_h, unit)
    canvas.paste(resized, (offset_x, offset_y))

    tensor = T.ToTensor()(canvas)      # [0, 1]
    tensor = tensor * 2.0 - 1.0        # [-1, 1]

    info: FillInfo = (offset_x, offset_y, new_w, new_h, new_h / max(1.0, float(h)))
    rect: ContentRect = (
        canvas_h // unit, canvas_w // unit,
        offset_y // unit, offset_x // unit,
        new_h // unit, new_w // unit,
    )
    return tensor, info, rect


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


# ── 风格图：整图输入预处理（ConvNeXt 风格编码器用）────────────────
#
# 与 tiling 的区别：不再切块，而是「等比缩放到固定基准高度 H + 动态宽度」，
# 保留宽高比（公式图长宽比可到 1:10+，压扁会毁掉笔迹特征）。
# 归一化沿用项目的 [-1, 1]（白底 +1 / 墨迹 -1），与 I_p / I_t 一致。


def style_resize_to_height(
    image: Image.Image,
    height: int = 64,
    max_width: Optional[int] = None,
    crop: str = "none",
    resample: int = Image.BICUBIC,
) -> Image.Image:
    """
    风格图：等比缩放到高度 = height（**绝不压扁**，宽度按原宽高比推导）。

    Args:
        image:   输入 PIL Image（任意模式，内部转 RGB）。
        height:  基准高度 H。
        max_width: 宽度上限；缩放后宽度超过它时按 crop 裁一段：
                   - "none"  ：不裁剪（宽度保持，可能超过 max_width）
                   - "random"：随机裁一段（训练）
                   - "center"：居中裁一段（验证 / 推理）
        crop:    见 max_width 说明。
        resample: 重采样方式，默认 BICUBIC（与项目其余部分一致）。

    Returns:
        PIL Image，尺寸 (height, W)，W ≥ 1。
    """
    img = image.convert("RGB")
    w, h = img.size
    if h <= 0 or w <= 0:
        raise ValueError(f"empty style image: size={img.size}")

    new_w = max(1, int(round(w * float(height) / float(h))))
    resized = img.resize((new_w, height), resample)

    if max_width is not None and new_w > max_width and crop != "none":
        if crop == "center":
            x0 = (new_w - max_width) // 2
        elif crop == "random":
            x0 = random.randint(0, new_w - max_width)
        else:
            raise ValueError(f"crop must be 'none'/'random'/'center', got {crop!r}")
        resized = resized.crop((x0, 0, x0 + max_width, height))

    return resized


def style_transform_pil(
    image: Image.Image,
    height: int = 64,
    max_width: Optional[int] = None,
    crop: str = "none",
    resample: int = Image.BICUBIC,
) -> torch.Tensor:
    """
    风格图 → tensor：等比缩放到高度 height 后归一化到 [-1, 1]。

    Returns:
        tensor (3, height, W) in [-1, 1]（白底 +1，墨迹 -1）
    """
    resized = style_resize_to_height(
        image, height=height, max_width=max_width, crop=crop, resample=resample
    )
    tensor = T.ToTensor()(resized)   # [0, 1]
    return tensor * 2.0 - 1.0        # [-1, 1]


def style_transform_path(
    path: str,
    height: int = 64,
    max_width: Optional[int] = None,
    crop: str = "center",
    resample: int = Image.BICUBIC,
) -> torch.Tensor:
    """风格图路径 → tensor (3, height, W) in [-1, 1]（默认居中裁剪，给验证/推理用）。"""
    with Image.open(path) as img:
        return style_transform_pil(
            img, height=height, max_width=max_width, crop=crop, resample=resample
        )


def pad_style_batch(
    styles: List[torch.Tensor],
    pad_value: float = 1.0,
    pad_side: str = "right",
    max_width: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    batch 内风格图右（或左）padding 到统一宽度。

    **padding 值必须是 +1.0（背景色白）**：0 在 [-1,1] 下是灰色，
    会引入虚假边缘，且会被前景 mask 误判成墨迹。

    Args:
        styles:   [(3, H, W_i) in [-1,1], ...]，H 必须一致。
        pad_value: padding 填充值，默认 +1.0。
        pad_side: "right"（默认，公式左端结构更密集）或 "left"。
        max_width: 宽度硬上限（超过则从内容侧截断）。

    Returns:
        (style_tensor (B, 3, H, W_max), style_mask (B, W_max) bool)
        style_mask: True = 真实区域，False = padding。
    """
    if not styles:
        raise ValueError("pad_style_batch got an empty list")

    B = len(styles)
    C, H, _ = styles[0].shape
    w_max = max(s.shape[2] for s in styles)
    if max_width is not None:
        w_max = min(w_max, int(max_width))

    out = styles[0].new_full((B, C, H, w_max), float(pad_value))
    mask = torch.zeros((B, w_max), dtype=torch.bool)

    for i, s in enumerate(styles):
        w = min(s.shape[2], w_max)
        s = s[:, :, :w]
        if pad_side == "left":
            out[i, :, :, w_max - w:] = s
            mask[i, w_max - w:] = True
        else:   # right
            out[i, :, :, :w] = s
            mask[i, :w] = True

    return out, mask


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
