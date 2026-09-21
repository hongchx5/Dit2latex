"""按配置桶尺寸给 print 图片分类，并生成带 LaTeX caption 的清单。"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image

try:
    import yaml
except ImportError as exc:
    raise SystemExit("需要安装 PyYAML：pip install pyyaml") from exc


ImageBucket = Tuple[int, int]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_config(config_path: Path) -> tuple[List[ImageBucket], ImageBucket]:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    data_config = config.get("data", {})
    all_size = data_config.get("all_size")
    train_size = data_config.get("train_size")
    if not all_size:
        raise ValueError(f"配置缺少 data.all_size: {config_path}")
    if not train_size or len(train_size) != 2:
        raise ValueError(f"配置缺少有效的 data.train_size: {config_path}")

    buckets = [tuple(map(int, size)) for size in all_size]
    selected_train_size = tuple(map(int, train_size))
    if selected_train_size not in buckets:
        raise ValueError(
            f"data.train_size={selected_train_size} 不在 data.all_size={buckets} 中"
        )
    return buckets, selected_train_size


def load_captions(caption_path: Path) -> Dict[str, str]:
    captions: Dict[str, str] = {}
    with caption_path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if "\t" in line:
                name, formula = line.split("\t", 1)
            else:
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    print(f"[warning] 忽略 caption 第 {line_number} 行: {line}")
                    continue
                name, formula = parts
            captions[Path(name.strip()).stem] = formula.strip()
    return captions


def pick_bucket(width: int, height: int, buckets: Sequence[ImageBucket]) -> ImageBucket:
    """使用项目现有分桶规则：等比容入后，选择填充面积最小的桶。"""
    best_bucket = buckets[0]
    best_padding = float("inf")
    for bucket_height, bucket_width in buckets:
        scale = min(bucket_height / height, bucket_width / width)
        content_area = (height * scale) * (width * scale)
        padding = bucket_height * bucket_width - content_area
        if padding < best_padding:
            best_padding = padding
            best_bucket = (bucket_height, bucket_width)
    return best_bucket


def natural_sort_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def resolve_config(project_root: Path, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path if explicit_path.is_absolute() else project_root / explicit_path
    candidates = (
        project_root / "config" / "config-default.yaml",
        project_root / "config-default.yaml",
        project_root / "config" / "default.yaml",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("找不到 config-default.yaml 或 config/default.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="配置文件路径")
    parser.add_argument("--data-root", type=Path, help="数据目录，默认使用项目下的 data")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bucket_caption.txt"),
        help="输出文件名，默认写入 data/bucket_caption.txt",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_root = args.data_root or project_root / "data"
    if not data_root.is_absolute():
        data_root = project_root / data_root
    config_path = resolve_config(project_root, args.config)
    buckets, train_size = load_config(config_path)
    captions = load_captions(data_root / "caption.txt")

    grouped: defaultdict[ImageBucket, List[str]] = defaultdict(list)
    missing_captions: List[str] = []
    image_paths = sorted(
        (path for path in (data_root / "print").iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: natural_sort_key(path.name),
    )
    for image_path in image_paths:
        with Image.open(image_path) as image:
            bucket = pick_bucket(image.width, image.height, buckets)
        if image_path.stem not in captions:
            missing_captions.append(image_path.name)
            continue
        grouped[bucket].append(image_path.name)

    output_path = args.output if args.output.is_absolute() else data_root / args.output
    lines = [
        f"{bucket[0]}x{bucket[1]} {name.split('.')[0]} {captions[Path(name).stem]}"
        for bucket in sorted(grouped)
        for name in grouped[bucket]
    ]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    print(f"配置: {config_path}")
    print(f"data.train_size: {train_size}")
    print(f"图片总数: {len(image_paths)}，已写入: {len(lines)}，缺少 caption: {len(missing_captions)}")
    print(f"输出: {output_path}")
    if missing_captions:
        print("[warning] 缺少 caption 的图片示例: " + ", ".join(missing_captions[:10]))


if __name__ == "__main__":
    main()
