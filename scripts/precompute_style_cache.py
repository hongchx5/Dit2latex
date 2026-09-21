"""
预计算风格图 tile 特征缓存。

用法：
    python scripts/precompute_style_cache.py \
        --data_root ./data \
        --cache_dir ./style_cache \
        --model_name /path/to/clip-vit-base-patch32 \
        --device cuda:0

说明：
    - 遍历 data_root 下所有 style* 目录中的图片；
    - 每张图平铺切分（tiling）后过冻结 CLIP，得到 (T, feature_dim) 特征；
    - 命中缓存（路径 + tiling 参数一致）则跳过，未命中的写入缓存；
    - 缓存由训练/推理时的 TiledCLIPStyleEncoder 直接读取（键相同）。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.style_cache import cache_key, load_tile_feats, save_tile_feats
from models.style_encoder import TiledCLIPStyleEncoder


def collect_style_images(data_root: str) -> list:
    root = Path(data_root)
    images = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and entry.name.lower().startswith("style"):
            for f in sorted(entry.iterdir()):
                if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    images.append(str(f))
    return images


def main():
    parser = argparse.ArgumentParser(description="Precompute style tile feature cache")
    parser.add_argument("--data_root", type=str, required=True, help="Data root containing style* dirs")
    parser.add_argument("--cache_dir", type=str, required=True, help="Cache output directory")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--tile_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=168)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    # 构造编码器（只使用冻结 CLIP 提取部分；聚合模块不参与预计算）
    encoder = TiledCLIPStyleEncoder(
        model_name=args.model_name,
        tile_size=args.tile_size,
        stride=args.stride,
        cache_dir=None,
        device=args.device,
    )

    images = collect_style_images(args.data_root)
    if not images:
        print(f"No style images found under {args.data_root}")
        return

    hits = 0
    computed = 0
    total_tiles = 0

    for path in images:
        key = cache_key(path, args.tile_size, args.stride)
        if load_tile_feats(args.cache_dir, key) is not None:
            hits += 1
            continue

        feats = encoder.extract_tile_feats(path)  # (T, feature_dim)
        save_tile_feats(args.cache_dir, key, feats)
        computed += 1
        total_tiles += feats.shape[0]
        print(f"[computed] {path} -> {feats.shape[0]} tiles, dim={feats.shape[1]}")

    print(f"\nDone: {len(images)} images, {hits} cache hits, {computed} newly computed, "
          f"{total_tiles} total tiles.")


if __name__ == "__main__":
    main()
