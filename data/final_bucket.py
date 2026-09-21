"""统计 print、style1、style2 图片的统一裁剪尺寸。

选择一个统一宽高比，使所有图片中心裁剪后平均保留面积最大，
然后根据图片高度中位数生成对应的 (H, W)。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import median
from typing import Iterable, List, Tuple

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def collect_sizes(data_root: Path) -> Tuple[List[Tuple[int, int]], List[Path]]:
	"""递归读取三个目标目录中的图片尺寸，返回 (宽, 高) 列表和失败文件。"""
	sizes: List[Tuple[int, int]] = []
	failed: List[Path] = []

	for folder_name in ("print", "style1", "style2"):
		folder = data_root / folder_name
		if not folder.is_dir():
			raise FileNotFoundError(f"目录不存在: {folder}")

		for path in sorted(folder.rglob("*")):
			if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
				continue
			try:
				with Image.open(path) as image:
					width, height = image.size
				if width > 0 and height > 0:
					sizes.append((width, height))
			except (OSError, ValueError):
				failed.append(path)

	return sizes, failed


def retained_area_fraction(width: int, height: int, target_ratio: float) -> float:
	"""返回按 target_ratio 裁剪时，原图面积的保留比例。"""
	source_ratio = width / height
	if source_ratio >= target_ratio:
		return target_ratio / source_ratio
	return source_ratio / target_ratio


def choose_ratio(sizes: Iterable[Tuple[int, int]]) -> float:
	"""从候选原图比例中选择平均保留面积最大的比例。"""
	sizes = list(sizes)
	candidates = sorted({width / height for width, height in sizes})
	return max(
		candidates,
		key=lambda ratio: sum(
			retained_area_fraction(width, height, ratio)
			for width, height in sizes
		) / len(sizes),
	)


def align_down(value: float, multiple: int) -> int:
	"""将尺寸向下对齐到 multiple，至少保留一个 multiple。"""
	return max(multiple, int(value // multiple) * multiple)


def main() -> None:
	parser = argparse.ArgumentParser(description="统计统一裁剪比例并输出 H、W")
	parser.add_argument(
		"data_root",
		nargs="?",
		type=Path,
		default=Path(__file__).resolve().parent,
		help="包含 print/style1/style2 的目录，默认是当前脚本所在目录",
	)
	parser.add_argument(
		"--multiple",
		type=int,
		default=8,
		help="H、W 的对齐倍数，默认 8",
	)
	args = parser.parse_args()
	if args.multiple <= 0:
		parser.error("--multiple 必须是正整数")

	sizes, failed = collect_sizes(args.data_root)
	if not sizes:
		raise RuntimeError("没有找到可读取的图片")

	target_ratio = choose_ratio(sizes)
	median_height = median(height for _, height in sizes)
	output_h = align_down(median_height, args.multiple)
	output_w = align_down(output_h * target_ratio, args.multiple)
	actual_ratio = output_w / output_h
	average_retained = sum(
		retained_area_fraction(width, height, target_ratio)
		for width, height in sizes
	) / len(sizes)

	print(f"图片数量: {len(sizes)}")
	print(f"读取失败: {len(failed)}")
	print(f"最佳宽高比 W/H: {target_ratio:.6f}")
	print(f"裁剪尺寸 H, W: {output_h}, {output_w}")
	print(f"实际宽高比 W/H: {actual_ratio:.6f}")
	print(f"平均保留面积: {average_retained:.2%}")
	if failed:
		print("失败文件示例:", failed[0])


if __name__ == "__main__":
	main()
