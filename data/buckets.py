"""
分桶（aspect-ratio bucketing）工具。

两套方案并存：

【A. 静态分桶（旧）】
桶尺寸是物理图（像素）尺寸，格式统一为 (H, W)。
默认桶集合覆盖宽图分布，所有边长满足 8×（VAE 下采样）与 2×（patch）整除：
    (64, 1024), (96, 512), (128, 512), (128, 384), (256, 256)

预处理语义（用户确认）：
- "恰好容入"：等比缩放后至少一条边等于桶对应边，另一条边不超过桶边；
- 填充：内容居中，对称填充；水平方向奇数余量右侧多一列，垂直方向奇数余量下侧多一行；
- 小于桶的图不放大（scale 取 min 比例，只缩小或保持原尺寸）。

【B. FiT 式 token 预算（新，可变分辨率）】
不再把所有图压到同一个固定尺寸，而是：
  1. 每张图按自身面积等比缩放，使 grid_h × grid_w ≤ max_tokens（只缩小不放大）；
  2. 按宽高比自动分箱，每箱取一个 canvas（也满足 token 预算），batch 内同 canvas；
  3. 样本缩放到自己的 fit 尺寸后居中贴进 canvas，空白 token 用 mask 屏蔽
     （注意力里当 key 被 mask 掉，损失里也不计数）。
1 个 patch token = vae_f × patch_size 像素边长（默认 8×2 = 16px）。
"""

from __future__ import annotations

import math
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


# ══════════════════════════════════════════════════════════════════
# FiT 式 token 预算（可变分辨率）
# ══════════════════════════════════════════════════════════════════

# 内容区在 token 网格中的位置：(canvas_gh, canvas_gw, row0, col0, content_gh, content_gw)
# —— 自带 canvas 尺寸，collate 端无需再知道 vae_f / patch_size。
ContentRect = Tuple[int, int, int, int, int, int]


def token_unit(vae_f: int = 8, patch_size: int = 2) -> int:
    """1 个 patch token 对应的像素边长 = vae_f × patch_size（默认 16px）。"""
    return vae_f * patch_size


def floor_to_unit(x: float, unit: int, minimum: Optional[int] = None) -> int:
    """把像素长度向下取整到 unit 的整数倍（至少 minimum，默认一个 unit）。"""
    lo = unit if minimum is None else minimum
    return max(lo, int(math.floor(float(x) / unit)) * unit)


def fit_size_to_token_budget(
    w: int,
    h: int,
    max_tokens: int = 256,
    vae_f: int = 8,
    patch_size: int = 2,
    only_downscale: bool = True,
) -> Tuple[int, int]:
    """
    按 token 预算等比缩放：使 grid_h × grid_w ≤ max_tokens。

    缩放系数只由「面积」决定：scale = sqrt(max_tokens × unit² / (w × h))，
    only_downscale=True 时再取 min(1, scale)（小图不放大）。
    两条边都向下对齐到 unit = vae_f × patch_size，保证 latent / patch 网格整除。

    Args:
        w, h:        原图宽高（像素）。
        max_tokens:  patch token 预算（grid_h × grid_w 的上界）。
        vae_f:       VAE 下采样倍率。
        patch_size:  DiT patch 边长。
        only_downscale: True 时只允许缩小。

    Returns:
        (fit_h, fit_w) 像素尺寸，均为 unit 的整数倍。
    """
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
    unit = token_unit(vae_f, patch_size)
    budget_px = float(max_tokens) * unit * unit
    area = max(1.0, float(w) * float(h))

    scale = math.sqrt(budget_px / area)
    if only_downscale:
        scale = min(1.0, scale)

    fit_h = floor_to_unit(h * scale, unit)
    fit_w = floor_to_unit(w * scale, unit)

    # 取整后复核预算（浮点误差 / 极端长宽比可能让 unit 对齐后的面积回弹）
    while (fit_h // unit) * (fit_w // unit) > max_tokens and fit_w > unit:
        fit_w -= unit
    while (fit_h // unit) * (fit_w // unit) > max_tokens and fit_h > unit:
        fit_h -= unit
    return fit_h, fit_w


def size_to_grid(
    h: int,
    w: int,
    vae_f: int = 8,
    patch_size: int = 2,
) -> Tuple[int, int]:
    """像素尺寸 → patch token 网格 (grid_h, grid_w)。"""
    unit = token_unit(vae_f, patch_size)
    return h // unit, w // unit


def aligned_offset(total: int, content: int, unit: int) -> int:
    """
    居中填充的左/上偏移，并向下对齐到 unit 的整数倍
    （保证内容区在 token 网格上整数对齐，剩余余量留在右/下侧）。
    """
    off = max(0, (total - content) // 2)
    return off - (off % unit)


def build_aspect_bins(
    items: Sequence[Tuple[str, int, int]],
    max_tokens: int = 256,
    num_bins: int = 8,
    vae_f: int = 8,
    patch_size: int = 2,
    only_downscale: bool = True,
) -> List[Tuple[int, int, List[str]]]:
    """
    按宽高比自动分箱：每箱给一个满足 token 预算的 canvas 尺寸。

    做法：先对每张图算 fit 尺寸，再按宽高比 r = w/h 排序后「等数量切分」成
    num_bins 箱（数据密集的比例区间自然分得更细）。每箱的 canvas 取箱内
    fit 的最大 grid_h / 最大 grid_w；若两者相乘超出预算，则收缩 grid_w 保证
    canvas 本身也不超预算（极少数样本会被 canvas 再压一点，由 contain 缩放兜住）。

    Args:
        items: [(name, w, h), ...]
        num_bins: 期望箱数（会被样本数截断）。

    Returns:
        [(canvas_h, canvas_w, [names...]), ...]
    """
    unit = token_unit(vae_f, patch_size)
    scored = []
    for name, w, h in items:
        fit_h, fit_w = fit_size_to_token_budget(
            w, h, max_tokens, vae_f, patch_size, only_downscale
        )
        ratio = float(w) / max(1.0, float(h))
        scored.append((ratio, name, fit_h, fit_w))
    scored.sort(key=lambda x: (x[0], x[1]))

    n = len(scored)
    k = max(1, min(int(num_bins), n))
    # 等数量切分，余数摊到前几箱
    counts = [(n + i) // k for i in range(k)]

    bins: List[Tuple[int, int, List[str]]] = []
    idx = 0
    for cnt in counts:
        group = scored[idx: idx + cnt]
        idx += cnt
        if not group:
            continue
        gh_c = max(fh // unit for _, _, fh, fw in group)
        gw_c = max(fw // unit for _, _, fh, fw in group)
        if gh_c * gw_c > max_tokens:
            gh_c = min(gh_c, max_tokens)
            gw_c = max(1, max_tokens // max(1, gh_c))
        bins.append((gh_c * unit, gw_c * unit, [g[1] for g in group]))

    # 相邻箱可能算出相同 canvas：合并，减少 DataLoader 数量
    merged: List[Tuple[int, int, List[str]]] = []
    for canvas_h, canvas_w, names in bins:
        if merged and merged[-1][0] == canvas_h and merged[-1][1] == canvas_w:
            merged[-1][2].extend(names)
        else:
            merged.append((canvas_h, canvas_w, list(names)))
    return merged
