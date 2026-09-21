"""
风格图 tile 特征缓存。

缓存内容：每张风格图经平铺切分 + CLIP（冻结）后的 tile 特征 (T, feature_dim)。
聚合模块（可学习）不缓存，训练时在线计算。

缓存键 = md5(绝对路径 + tiling 参数)，保证：
  - 路径或 tiling 参数（tile_size / stride）变化时自动失效；
  - 训练与推理共用同一份缓存。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import torch


def cache_key(path: str, tile_size: int, stride: int) -> str:
    """生成缓存文件名（不含扩展名）。"""
    raw = f"{os.path.abspath(path)}|tile={tile_size}|stride={stride}"
    return hashlib.md5(raw.encode()).hexdigest()


def cache_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, f"{key}.pt")


def load_tile_feats(cache_dir: Optional[str], key: str) -> Optional[torch.Tensor]:
    """
    从缓存加载 tile 特征。

    Args:
        cache_dir: 缓存目录；None 或目录不存在时视为未命中。
        key: cache_key 生成的键。

    Returns:
        tile 特征 (T, feature_dim)，未命中时返回 None。
    """
    if cache_dir is None:
        return None
    path = cache_path(cache_dir, key)
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return None


def save_tile_feats(cache_dir: str, key: str, feats: torch.Tensor) -> str:
    """
    保存 tile 特征到缓存（原子写：临时文件 + rename，避免多进程并发冲突）。

    Args:
        cache_dir: 缓存目录。
        key: cache_key 生成的键。
        feats: (T, feature_dim) tensor。

    Returns:
        保存的完整路径。
    """
    os.makedirs(cache_dir, exist_ok=True)
    final_path = cache_path(cache_dir, key)
    tmp_path = final_path + f".tmp{os.getpid()}"
    torch.save(feats.detach().cpu(), tmp_path)
    os.replace(tmp_path, final_path)
    return final_path
