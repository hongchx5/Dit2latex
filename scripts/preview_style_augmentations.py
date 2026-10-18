
#!/usr/bin/env python
"""风格数据增强预览脚本。

为手写公式图生成多种笔迹风格增强效果预览图：
    1. 笔画粗细（Stroke Thickness）：形态学 dilate / erode
    2. 倾斜角（Slant）：仿射 shear
    3. 字高（Character Height）：横向/纵向拉伸
    4. 圆润度（Roundness）：高斯平滑 + 二值化
    5. 连笔程度（Ligature）：骨架收缩或局部膨胀模拟连笔

用法：
    python scripts/preview_style_augmentations.py --input_dir data/test --output_dir preview_output
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from scipy.ndimage import distance_transform_edt


def pil_to_bin(img: Image.Image) -> Tuple[np.ndarray, Tuple[int, int]]:
    """PIL 图像 -> 二值 numpy (H, W)，墨迹=1，背景=0，并返回原始尺寸。"""
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.float32)
    # 简单阈值：假设公式图为二值或接近二值；中值阈值对灰度图也鲁棒
    threshold = np.median(arr)
    binary = (arr < threshold).astype(np.float32)  # 墨=1，背景=0
    return binary, arr.shape


def bin_to_pil(binary: np.ndarray) -> Image.Image:
    """二值 numpy (H, W) -> PIL RGB，白底黑字。"""
    uint8 = ((1 - binary) * 255).astype(np.uint8)
    return Image.fromarray(uint8).convert("RGB")


def crop_to_content(binary: np.ndarray, pad: int = 2) -> np.ndarray:
    """裁剪到内容区域，去除上下左右大边白边。"""
    ys = np.where(binary.sum(axis=1) > 0)[0]
    xs = np.where(binary.sum(axis=0) > 0)[0]
    if len(ys) == 0 or len(xs) == 0:
        return binary
    y1, y2 = max(0, ys[0] - pad), min(binary.shape[0], ys[-1] + 1 + pad)
    x1, x2 = max(0, xs[0] - pad), min(binary.shape[1], binary.shape[1] + 1 + pad)
    return binary[y1:y2, x1:x2]


def stroke_thickness(binary: np.ndarray, thickness: float) -> np.ndarray:
    """
    调整笔画粗细。
    thickness > 0: 膨胀变粗；thickness < 0: 腐蚀变细。
    """
    if abs(thickness) < 0.01:
        return binary
    iterations = max(1, int(round(abs(thickness) * 2)))
    if thickness > 0:
        # dilate: 墨区域扩大
        for _ in range(iterations):
            binary = ndimage.grey_dilation(binary, size=(3, 3))
    else:
        for _ in range(iterations):
            binary = ndimage.grey_erosion(binary, size=(3, 3))
    return (binary > 0.5).astype(np.float32)


def slant(binary: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    对图像做水平 shear（倾斜）。angle_deg 为正：向右倾斜。
    """
    if abs(angle_deg) < 0.1:
        return binary
    angle_rad = math.radians(angle_deg)
    shear = math.tan(angle_rad)
    h, w = binary.shape
    # 估计新宽度并做 shear
    new_w = int(w + abs(shear) * h)
    # 使用 PIL affine 更方便：在二值图上重采样
    pil_img = bin_to_pil(binary)
    # 变换矩阵 (a, b, c, d, e, f): x' = ax + by + c, y' = dx + ey + f
    # shear in x direction: x' = x + shear*y, y' = y
    # 需要扩宽图像
    transform = (1, shear, -shear * h / 2 if shear > 0 else 0, 0, 1, 0)
    # 使用 nearest 保持二值
    out = pil_img.transform((new_w, h), Image.AFFINE, transform, resample=Image.NEAREST)
    # 转回二值
    gray = np.array(out.convert("L"), dtype=np.float32)
    return (gray < 128).astype(np.float32)


def character_height(binary: np.ndarray, h_scale: float, w_scale: float) -> np.ndarray:
    """
    字高/字宽缩放：h_scale 纵向，w_scale 横向。
    h_scale > 1 字更高，w_scale > 1 字更宽。
    """
    if abs(h_scale - 1) < 0.01 and abs(w_scale - 1) < 0.01:
        return binary
    h, w = binary.shape
    new_h = max(8, int(round(h * h_scale)))
    new_w = max(8, int(round(w * w_scale)))
    pil_img = bin_to_pil(binary)
    resized = pil_img.resize((new_w, new_h), Image.BICUBIC)
    gray = np.array(resized.convert("L"), dtype=np.float32)
    return (gray < 128).astype(np.float32)


def roundness(binary: np.ndarray, sigma: float) -> np.ndarray:
    """
    圆润度：用高斯平滑 + 阈值，使尖角变圆。
    sigma 越大越圆。
    """
    if sigma < 0.1:
        return binary
    smoothed = ndimage.gaussian_filter(binary.astype(np.float32), sigma=sigma)
    # 阈值可自适应：保持大致墨面积
    threshold = 0.5
    return (smoothed > threshold).astype(np.float32)


def ligature(binary: np.ndarray, strength: float) -> np.ndarray:
    """
    连笔程度：对距离较近的笔画做局部膨胀，模拟连笔。
    strength 控制膨胀半径。
    """
    if strength < 0.01:
        return binary
    radius = max(1, int(round(strength * 3)))
    # 使用 distance_transform：对背景像素计算到墨的距离，距离小的背景像素填充为墨
    dist = distance_transform_edt(1 - binary)
    return ((dist <= radius) | (binary > 0.5)).astype(np.float32)


def make_preview_grid(
    original: np.ndarray,
    augmented_rows: List[Tuple[str, List[Tuple[str, np.ndarray]]]],
    cell_height: int = 96,
    cell_width: int = 384,
    label_width: int = 180,
    pad: int = 4,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    生成预览大图：每行是一个增强类型，每列是不同强度。
    第一列为原图。
    """
    n_rows = len(augmented_rows)
    n_cols = max(len(cols) for _, cols in augmented_rows) + 1  # +1 for original

    total_h = n_rows * (cell_height + pad) + pad
    total_w = label_width + n_cols * (cell_width + pad) + pad

    canvas = np.full((total_h, total_w, 3), bg_color, dtype=np.uint8)

    def paste(binary_img: np.ndarray, y: int, x: int):
        h, w = binary_img.shape
        # 等比缩放到 cell 内，保持比例
        scale = min(cell_height / h, cell_width / w)
        new_h = max(2, int(round(h * scale)))
        new_w = max(2, int(round(w * scale)))
        pil_img = bin_to_pil(binary_img).resize((new_w, new_h), Image.BICUBIC)
        arr = np.array(pil_img)
        y_off = y + (cell_height - new_h) // 2
        x_off = x + (cell_width - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = arr

    # 第一行放原图和列标题
    # 后续每行一个增强类型
    for row_idx, (row_label, cols) in enumerate(augmented_rows):
        y = pad + row_idx * (cell_height + pad)
        # 标签
        label_pil = Image.new("RGB", (label_width, cell_height), bg_color)
        # 写标签（简单用 PIL 文本）
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(label_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
        draw.text((10, cell_height // 2 - 10), row_label, fill=(0, 0, 0), font=font)
        canvas[y:y + cell_height, pad:pad + label_width] = np.array(label_pil)

        # 原图列
        x = pad + label_width
        paste(original, y, x)

        # 增强图
        for col_idx, (col_label, aug_img) in enumerate(cols):
            x = pad + label_width + (col_idx + 1) * (cell_width + pad)
            paste(aug_img, y, x)

    return Image.fromarray(canvas)


def process_image(input_path: str, output_dir: str):
    """为单张输入图生成所有增强预览并保存。"""
    img = Image.open(input_path)
    binary, _ = pil_to_bin(img)
    binary = crop_to_content(binary, pad=4)

    rows = []

    # 1. 笔画粗细
    rows.append(("Stroke Thickness", [
        ("thin (-0.8)", stroke_thickness(binary.copy(), -0.8)),
        ("normal", binary.copy()),
        ("thick (+0.6)", stroke_thickness(binary.copy(), 0.6)),
        ("very thick (+1.2)", stroke_thickness(binary.copy(), 1.2)),
    ]))

    # 2. 倾斜角
    rows.append(("Slant Angle", [
        ("left -10", slant(binary.copy(), -10)),
        ("left -5", slant(binary.copy(), -5)),
        ("normal", binary.copy()),
        ("right +5", slant(binary.copy(), 5)),
        ("right +10", slant(binary.copy(), 10)),
    ]))

    # 3. 字高（横向/纵向拉伸）
    rows.append(("Char Height/Width", [
        ("tall", character_height(binary.copy(), 1.3, 1.0)),
        ("short", character_height(binary.copy(), 0.7, 1.0)),
        ("wide", character_height(binary.copy(), 1.0, 1.4)),
        ("narrow", character_height(binary.copy(), 1.0, 0.7)),
    ]))

    # 4. 圆润度
    rows.append(("Roundness", [
        ("sharp", binary.copy()),
        ("slight", roundness(binary.copy(), 0.7)),
        ("medium", roundness(binary.copy(), 1.2)),
        ("strong", roundness(binary.copy(), 1.8)),
    ]))

    # 5. 连笔程度
    rows.append(("Ligature", [
        ("none", binary.copy()),
        ("light", ligature(binary.copy(), 0.5)),
        ("medium", ligature(binary.copy(), 1.0)),
        ("heavy", ligature(binary.copy(), 1.8)),
    ]))

    preview = make_preview_grid(binary, rows, cell_height=120, cell_width=420, label_width=200)

    os.makedirs(output_dir, exist_ok=True)
    stem = Path(input_path).stem
    out_path = os.path.join(output_dir, f"{stem}_augment_preview.png")
    preview.save(out_path)
    print(f"Saved preview: {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview handwriting style augmentations")
    parser.add_argument("--input_dir", type=str, default="data/test", help="Directory of input formula images")
    parser.add_argument("--output_dir", type=str, default="preview_output", help="Directory to save preview images")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    files = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith(exts)])
    if not files:
        print(f"No images found in {args.input_dir}")
        return

    print(f"Found {len(files)} images in {args.input_dir}")
    for f in files:
        process_image(os.path.join(args.input_dir, f), args.output_dir)
    print(f"All previews saved to {args.output_dir}")


if __name__ == "__main__":
    main()
