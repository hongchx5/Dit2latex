"""
风格编码器：从手写风格参考图像提取笔迹特征。

- 正常模式：TiledCLIPStyleEncoder —— 风格图平铺切分（tiling）后逐块过冻结 CLIP，
  再用 4 个可学习全局查询做 Cross-Attention 聚合，输出：
      f_s_seq   (B, 4, feature_dim)   4 个聚合向量（DiT Cross-Attention 的 K/V）
      f_s_pooled (B, feature_dim)     concat 后 MLP 4 倍压缩的全局特征（AdaLN 条件）
  tile 特征缓存：命中缓存直接用，未命中临时计算（可选写回）。
- Offline 模式：简单 CNN → FC → 768 维向量（M=1）。

位置编码（tile 聚合）：一行 tiles，alpha = i/(T-1)（T=1 时 alpha=0.5），
pos = alpha * 2π，标准 1D 正弦位置编码，加到 tile 特征（K/V 侧）。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from data.transforms import tile_image_to_tensors, tiles_to_batch
from data.style_cache import cache_key, load_tile_feats, save_tile_feats


def _tile_pos_embedding(T: int, feature_dim: int, device: torch.device) -> torch.Tensor:
    """
    一行 tile 的 1D 正弦位置编码。

    pos = alpha * 2π，alpha = i/(T-1)（T=1 时 alpha=0.5）。
    归一化到 [0, 1] 使位置编码不依赖 T 的绝对大小，对外推友好。

    Args:
        T: tile 数量。
        feature_dim: 特征维度。
        device: 目标设备。

    Returns:
        pos_emb: (T, feature_dim)
    """
    if T == 1:
        alpha = torch.tensor([0.5], dtype=torch.float32, device=device)
    else:
        alpha = torch.arange(T, dtype=torch.float32, device=device) / (T - 1)
    pos = alpha * (2.0 * math.pi)  # (T,)

    half = feature_dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, dtype=torch.float32, device=device) / half
    )  # (half,)
    angles = pos.unsqueeze(-1) * freqs.unsqueeze(0)  # (T, half)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (T, feature_dim)


class TiledCLIPStyleEncoder(nn.Module):
    """
    平铺切分 + CLIP（冻结） + 可学习 4-query 聚合 的风格编码器。

    输出接口（供 pipeline 使用）：
        encode(paths) -> (f_s_seq (B, 4, feature_dim), f_s_pooled (B, feature_dim))
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        feature_dim: int = 768,
        tile_size: int = 224,
        stride: int = 168,
        cache_dir: Optional[str] = None,
        num_query: int = 4,
        device: str = "cuda",
    ):
        super().__init__()
        try:
            from transformers import CLIPVisionModel, CLIPImageProcessor
        except ImportError:
            raise ImportError("transformers library required. Install with: pip install transformers")

        self.clip = CLIPVisionModel.from_pretrained(model_name).to(device)
        self.clip.requires_grad_(False)
        self.clip.eval()

        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.feature_dim = self.clip.config.hidden_size  # 768
        assert self.feature_dim == feature_dim, \
            f"CLIP hidden_size {self.feature_dim} != feature_dim {feature_dim}"

        self.tile_size = tile_size
        self.stride = stride
        self.cache_dir = cache_dir
        self.num_query = num_query

        # ── 可学习聚合模块（参与梯度训练）──
        # 4 个全局查询向量（跨样本共享）
        # 注意：必须用叶子张量构造 Parameter，否则梯度不会填充（optimizer 不更新）
        self.queries = nn.Parameter(torch.empty(num_query, self.feature_dim).normal_(0, 0.02))

        self.q_proj = nn.Linear(self.feature_dim, self.feature_dim, bias=True)
        self.kv_proj = nn.Linear(self.feature_dim, self.feature_dim * 2, bias=True)
        self.out_proj = nn.Linear(self.feature_dim, self.feature_dim, bias=True)
        self.attn_dropout = nn.Dropout(0.0)

        # 4 个聚合向量 concat → LayerNorm → MLP 4 倍压缩 → 全局特征
        self.agg_norm = nn.LayerNorm(num_query * self.feature_dim)
        self.agg_mlp = nn.Sequential(
            nn.Linear(num_query * self.feature_dim, self.feature_dim, bias=True),
            nn.SiLU(),
            nn.Linear(self.feature_dim, self.feature_dim, bias=True),
        )

        # 聚合模块整体移动到目标设备（CLIP 已在其上；避免参数留在 CPU）
        self.to(device)

    # ── CLIP tile 特征提取（冻结）────────────────────────────────────

    @torch.no_grad()
    def extract_tile_feats(self, path: str) -> torch.Tensor:
        """
        单张风格图：tiling → 逐块 CLIP → (T, feature_dim)。

        纯计算、不查询缓存（预计算脚本用）。
        """
        with Image.open(path) as img:
            tiles = tile_image_to_tensors(img, self.tile_size, self.stride)
        tiles_t = tiles_to_batch(tiles).to(next(self.clip.parameters()).device)  # (T, 3, 224, 224) [-1,1]

        # CLIP 期望 [0, 1] 像素值
        pixel_values = tiles_t * 0.5 + 0.5
        outputs = self.clip(pixel_values=pixel_values)
        return outputs.pooler_output  # (T, feature_dim)

    def _get_tile_feats(self, path: str) -> torch.Tensor:
        """查缓存；未命中则临时计算（命中缓存时训练/推理共用）。"""
        key = cache_key(path, self.tile_size, self.stride)
        feats = load_tile_feats(self.cache_dir, key)
        if feats is not None:
            return feats.to(next(self.clip.parameters()).device)
        feats = self.extract_tile_feats(path)
        if self.cache_dir is not None:
            try:
                save_tile_feats(self.cache_dir, key, feats)
            except Exception:
                pass  # 缓存写入失败不影响训练
        return feats

    # ── 聚合（可学习，在线计算）──────────────────────────────────────

    def _aggregate(self, tile_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tile_feats: (T, feature_dim) 单张风格图的 tile 特征。
        Returns:
            f_s: (num_query, feature_dim) 聚合向量（作为 f_s_seq 的 1 个样本）
            pooled: (feature_dim,) 全局特征（f_s_pooled 的 1 个样本）
        """
        T, D = tile_feats.shape
        device = tile_feats.device

        # 位置编码（加到 tile 特征 / K-V 侧），让 query 感知 tile 顺序
        pos_emb = _tile_pos_embedding(T, D, device)  # (T, D)
        x = tile_feats + pos_emb

        # Cross-Attention：4 个全局查询 → 4 个聚合向量
        q = self.q_proj(self.queries)  # (num_query, D)
        kv = self.kv_proj(x)           # (T, 2D)
        k, v = kv.chunk(2, dim=-1)     # (T, D), (T, D)

        attn = (q @ k.transpose(-1, -2)) / math.sqrt(D)  # (num_query, T)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        agg = attn @ v                  # (num_query, D)
        agg = self.out_proj(agg)        # (num_query, D)

        # concat → LN → MLP 4 倍压缩 → 全局特征
        pooled = self.agg_mlp(self.agg_norm(agg.reshape(-1)))  # (D,)

        return agg, pooled

    # ── 对外接口 ─────────────────────────────────────────────────────

    def encode_from_paths(self, paths: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        注意：聚合模块（_aggregate）参与梯度训练，本方法不在 no_grad 下运行；
        CLIP 前向部分已由 extract_tile_feats 内部的 @torch.no_grad() 单独冻结。

        Args:
            paths: (B,) 风格图路径列表（每个样本 1 张风格图）。
        Returns:
            f_s_seq:   (B, num_query, feature_dim)
            f_s_pooled: (B, feature_dim)
        """
        device = next(self.clip.parameters()).device
        seq_list: List[torch.Tensor] = []
        pooled_list: List[torch.Tensor] = []
        for path in paths:
            tile_feats = self._get_tile_feats(path)  # (T, D)
            agg, pooled = self._aggregate(tile_feats)
            seq_list.append(agg)
            pooled_list.append(pooled)

        f_s_seq = torch.stack(seq_list, dim=0)        # (B, num_query, D)
        f_s_pooled = torch.stack(pooled_list, dim=0)  # (B, D)
        return f_s_seq, f_s_pooled

    def encode(self, source: Union[str, List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        统一入口（pipeline 调用）。支持单路径或路径列表。

        Args:
            source: 风格图路径 str 或 list[str]。
        Returns:
            (f_s_seq, f_s_pooled)
        """
        if isinstance(source, str):
            return self.encode_from_paths([source])
        return self.encode_from_paths(list(source))


class CLIPStyleEncoder(nn.Module):
    """旧版单图 CLIP 风格编码器（已废弃，仅保留兼容；新训练请用 TiledCLIPStyleEncoder）。"""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        super().__init__()
        try:
            from transformers import CLIPVisionModel, CLIPImageProcessor
        except ImportError:
            raise ImportError("transformers library required. Install with: pip install transformers")

        self.clip = CLIPVisionModel.from_pretrained(model_name).to(device)
        self.clip.requires_grad_(False)
        self.clip.eval()

        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.feature_dim = self.clip.config.hidden_size  # 768

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W) in [-1, 1]。
        Returns:
            f_s: (B, feature_dim)
        """
        pixel_values = (images * 0.5 + 0.5).clamp(0, 1)
        pixel_values = F.interpolate(pixel_values, size=(224, 224), mode="bicubic", antialias=True)
        outputs = self.clip(pixel_values=pixel_values)
        return outputs.pooler_output


class OfflineStyleEncoder(nn.Module):
    """Offline 测试用风格编码器：轻量 CNN + FC（输出 M=1 的序列）。"""

    def __init__(self, feature_dim: int = 768, image_size: int = 256):
        super().__init__()
        self.feature_dim = feature_dim

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 128
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 32
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 16
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, feature_dim)

    def encode(self, images: Union[torch.Tensor, List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: (B, 3, H, W) 堆叠 tensor 或长度 B 的 tensor 列表（风格图尺寸可不同）。
        Returns:
            f_s_seq:   (B, 1, feature_dim)
            f_s_pooled: (B, feature_dim)
        """
        if isinstance(images, torch.Tensor):
            batch = [images]
        else:
            batch = list(images)

        pooled_list = [self._encode_single(x) for x in batch]
        f_s_pooled = torch.stack(pooled_list, dim=0)   # (B, feature_dim)
        f_s_seq = f_s_pooled.unsqueeze(1)              # (B, 1, feature_dim)
        return f_s_seq, f_s_pooled

    def _encode_single(self, x: torch.Tensor) -> torch.Tensor:
        """单张风格图 → (feature_dim,)。"""
        out = self.conv(x.unsqueeze(0))  # (1, 256, 1, 1)
        out = out.flatten(1)             # (1, 256)
        return self.fc(out).squeeze(0)   # (feature_dim,)


def build_style_encoder(
    offline_test: bool = False,
    model_name: str = "openai/clip-vit-base-patch32",
    feature_dim: int = 768,
    image_size: int = 256,
    tile_size: int = 224,
    stride: int = 168,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
) -> nn.Module:
    """创建风格编码器实例。"""
    if offline_test:
        return OfflineStyleEncoder(feature_dim=feature_dim, image_size=image_size).to(device)
    else:
        return TiledCLIPStyleEncoder(
            model_name=model_name,
            feature_dim=feature_dim,
            tile_size=tile_size,
            stride=stride,
            cache_dir=cache_dir,
            device=device,
        )
