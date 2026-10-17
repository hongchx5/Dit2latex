
"""
风格编码器：从手写风格参考图像提取笔迹特征。

- 新版默认：ConvNeXtStyleEncoder —— 整图输入 + 前景感知 4-query 聚合，
  用 ConvNeXt-T 作为 backbone（stem stride 改为 2，总下采样 16），
  输出与 FiT pipeline 兼容的：
      f_s_seq    (B, num_query, feature_dim)
      f_s_pooled (B, feature_dim)
- 旧版保留：TiledCLIPStyleEncoder / CLIPStyleEncoder / OfflineStyleEncoder（兼容）。

归一化约定：输入图像 ∈ [-1, 1]，白底 = +1，墨迹 = -1。
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from data.transforms import (
    tile_image_to_tensors,
    tiles_to_batch,
    preprocess_style_image,
    batch_style_images,
)
from data.style_cache import cache_key, load_tile_feats, save_tile_feats


def _tile_pos_embedding(T: int, feature_dim: int, device: torch.device) -> torch.Tensor:
    """
    一行 tile 的 1D 正弦位置编码（仅旧 TiledCLIPStyleEncoder 使用）。
    """
    if T == 1:
        alpha = torch.tensor([0.5], dtype=torch.float32, device=device)
    else:
        alpha = torch.arange(T, dtype=torch.float32, device=device) / (T - 1)
    pos = alpha * (2.0 * math.pi)

    half = feature_dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, dtype=torch.float32, device=device) / half
    )
    angles = pos.unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def _normalized_2d_sine_pos_embedding(
    H: int,
    W: int,
    feature_dim: int,
    device: torch.device,
    temperature: float = 10000.0,
) -> torch.Tensor:
    """
    归一化 2D 正弦位置编码，变宽友好。
    """
    assert feature_dim % 4 == 0, f"feature_dim must be divisible by 4, got {feature_dim}"
    half = feature_dim // 2
    quarter = feature_dim // 4

    row_norm = torch.arange(H, dtype=torch.float32, device=device) / max(1, H - 1)
    col_norm = torch.arange(W, dtype=torch.float32, device=device) / max(1, W - 1)

    freqs = torch.exp(
        -math.log(temperature) * torch.arange(0, quarter, dtype=torch.float32, device=device) / quarter
    )

    row_angles = row_norm.unsqueeze(-1) * freqs.unsqueeze(0)
    col_angles = col_norm.unsqueeze(-1) * freqs.unsqueeze(0)

    row_emb = torch.cat([torch.sin(row_angles), torch.cos(row_angles)], dim=-1)
    col_emb = torch.cat([torch.sin(col_angles), torch.cos(col_angles)], dim=-1)

    row_emb = row_emb.unsqueeze(1).expand(H, W, half)
    col_emb = col_emb.unsqueeze(0).expand(H, W, half)
    return torch.cat([row_emb, col_emb], dim=-1).reshape(H * W, feature_dim)


class _ForegroundAwareAggregator(nn.Module):
    """
    4-query Cross-Attention + 前景 mask + null token 聚合头。
    """

    def __init__(
        self,
        feature_dim: int = 768,
        num_query: int = 4,
        num_heads: int = 12,
        dropout: float = 0.0,
        fg_threshold: float = 0.0,
        total_stride: int = 16,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_query = num_query
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.fg_threshold = fg_threshold
        self.total_stride = total_stride

        assert feature_dim % num_heads == 0, (
            f"feature_dim {feature_dim} must be divisible by num_heads {num_heads}"
        )

        # 注意：必须用叶子张量构造 Parameter，否则梯度不会填充
        self.queries = nn.Parameter(torch.empty(num_query, feature_dim).normal_(0, 0.02))

        self.q_proj = nn.Linear(feature_dim, feature_dim, bias=True)
        self.kv_proj = nn.Linear(feature_dim, feature_dim * 2, bias=True)
        self.out_proj = nn.Linear(feature_dim, feature_dim, bias=True)
        self.attn_dropout = nn.Dropout(dropout)

        # null token：永远可见，防止全背景图导致 softmax 数值异常
        self.null_token = nn.Parameter(torch.empty(1, feature_dim).normal_(0, 0.02))

        self.agg_norm = nn.LayerNorm(num_query * feature_dim)
        self.agg_mlp = nn.Sequential(
            nn.Linear(num_query * feature_dim, feature_dim, bias=True),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, D, H, W = x.shape
        device = x.device

        # 1. 构造前景/有效 mask（必须在原始输入图上做，特征图 x 已经 stride=16，无法再放 16x kernel）
        if images is None:
            images = x
        fg = (images < self.fg_threshold).float()
        fg = (fg.mean(dim=1, keepdim=True) > 0.5).float()
        fg_ds = F.max_pool2d(fg, kernel_size=self.total_stride, stride=self.total_stride)
        keep = fg_ds.squeeze(1).bool()

        if valid_mask is not None:
            vm = valid_mask.unsqueeze(1).unsqueeze(2).float()
            vm_ds = F.max_pool1d(
                vm.view(B, 1, -1),
                kernel_size=self.total_stride,
                stride=self.total_stride,
            )
            vm_ds = vm_ds.squeeze(1).bool()  # (B, W')
            vm_ds = vm_ds.unsqueeze(1).expand_as(keep)  # (B, H, W')
            keep = keep & vm_ds

        # 2. flatten + 加 2D 位置编码
        tokens = x.view(B, D, H * W).permute(0, 2, 1)
        pos_emb = _normalized_2d_sine_pos_embedding(H, W, D, device)
        tokens = tokens + pos_emb.unsqueeze(0)

        # 3. 追加 null token 到 K/V
        null = self.null_token.expand(B, 1, D)
        kv_tokens = torch.cat([tokens, null], dim=1)

        # 4. Cross-Attention
        q = self.q_proj(self.queries).unsqueeze(0).expand(B, -1, -1)
        kv = self.kv_proj(kv_tokens)
        k, v = kv.chunk(2, dim=-1)

        q = q.view(B, self.num_query, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        keep_flat = keep.reshape(B, 1, 1, H * W).expand(B, 1, self.num_query, H * W)
        null_visible = torch.ones(B, 1, self.num_query, 1, dtype=torch.bool, device=device)
        visible = torch.cat([keep_flat, null_visible], dim=-1)
        attn = attn.masked_fill(~visible, -1e4)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        agg = attn @ v
        agg = agg.transpose(1, 2).reshape(B, self.num_query, D)
        agg = self.out_proj(agg)

        # 5. pooled
        pooled = self.agg_mlp(self.agg_norm(agg.reshape(B, -1)))
        return agg, pooled


class ConvNeXtStyleEncoder(nn.Module):
    """
    新版风格编码器：ConvNeXt-T backbone + 前景感知 4-query 聚合。
    """

    def __init__(
        self,
        feature_dim: int = 768,
        num_query: int = 4,
        image_height: int = 64,
        max_width: int = 512,
        pretrained: bool = True,
        fg_threshold: float = 0.0,
        num_heads: int = 12,
        dropout: float = 0.0,
        device: str = "cuda",
    ):
        super().__init__()
        try:
            from torchvision import models
        except ImportError:
            raise ImportError("torchvision is required for ConvNeXtStyleEncoder")

        self.feature_dim = feature_dim
        self.num_query = num_query
        self.image_height = image_height
        self.max_width = max_width
        self.fg_threshold = fg_threshold
        self.device = device

        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        convnext = models.convnext_tiny(weights=weights)

        # 把 stem 第一个卷积 stride 从 4 改为 2，总下采样变为 16
        first_conv = convnext.features[0][0]
        if isinstance(first_conv, nn.Conv2d):
            first_conv.stride = (2, 2)
            first_conv.padding = (1, 1)
        else:
            for m in convnext.features[0].modules():
                if isinstance(m, nn.Conv2d):
                    m.stride = (2, 2)
                    m.padding = (1, 1)
                    break

        self.backbone = convnext.features
        backbone_out_dim = 768
        if backbone_out_dim != feature_dim:
            self.proj = nn.Conv2d(backbone_out_dim, feature_dim, kernel_size=1)
        else:
            self.proj = nn.Identity()

        self.aggregator = _ForegroundAwareAggregator(
            feature_dim=feature_dim,
            num_query=num_query,
            num_heads=num_heads,
            dropout=dropout,
            fg_threshold=fg_threshold,
            total_stride=16,
        )
        self.to(device)

    def _preprocess_paths(self, paths: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        tensors: List[torch.Tensor] = []
        for path in paths:
            tensor, _ = preprocess_style_image(
                Image.open(path),
                image_height=self.image_height,
                max_width=self.max_width,
                random_crop=False,
            )
            tensors.append(tensor)
        return batch_style_images(tensors)

    def encode(
        self,
        source: Union[str, Image.Image, torch.Tensor, List[str], List[torch.Tensor]],
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(source, str):
            return self.encode([source])

        if isinstance(source, Image.Image):
            tensor, _ = preprocess_style_image(
                source.convert("RGB"),
                image_height=self.image_height,
                max_width=self.max_width,
                random_crop=False,
            )
            tensor = tensor.unsqueeze(0)
            valid_mask = torch.ones(1, tensor.shape[-1], dtype=torch.bool)
            return self.encode(tensor, valid_mask)

        if isinstance(source, torch.Tensor):
            images = source.to(self.device)
            assert valid_mask is not None, "must provide valid_mask when source is a tensor"
            valid_mask = valid_mask.to(self.device)
        elif isinstance(source, list) and len(source) == 0:
            raise ValueError("empty source list")
        elif isinstance(source, list) and isinstance(source[0], str):
            images, valid_mask = self._preprocess_paths(source)
            images = images.to(self.device)
            valid_mask = valid_mask.to(self.device)
        elif isinstance(source, list) and isinstance(source[0], torch.Tensor):
            images, valid_mask = batch_style_images(source)
            images = images.to(self.device)
            valid_mask = valid_mask.to(self.device)
        else:
            raise TypeError(f"unsupported source type: {type(source)}")

        B, C, H, W = images.shape
        assert H == self.image_height, f"input height {H} != configured {self.image_height}"
        if W > self.max_width:
            print(f"[WARN] style width {W} exceeds max_width {self.max_width}; center cropping")
            start = (W - self.max_width) // 2
            images = images[:, :, :, start:start + self.max_width]
            valid_mask = valid_mask[:, start:start + self.max_width]

        features = self.backbone(images)
        features = self.proj(features)
        return self.aggregator(features, valid_mask, images=images)


class TiledCLIPStyleEncoder(nn.Module):
    """
    平铺切分 + CLIP（冻结） + 可学习 4-query 聚合 的旧版风格编码器。
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
            raise ImportError("transformers library required")

        self.clip = CLIPVisionModel.from_pretrained(model_name).to(device)
        self.clip.requires_grad_(False)
        self.clip.eval()

        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.feature_dim = self.clip.config.hidden_size
        assert self.feature_dim == feature_dim

        self.tile_size = tile_size
        self.stride = stride
        self.cache_dir = cache_dir
        self.num_query = num_query

        self.queries = nn.Parameter(torch.empty(num_query, self.feature_dim).normal_(0, 0.02))
        self.q_proj = nn.Linear(self.feature_dim, self.feature_dim, bias=True)
        self.kv_proj = nn.Linear(self.feature_dim, self.feature_dim * 2, bias=True)
        self.out_proj = nn.Linear(self.feature_dim, self.feature_dim, bias=True)
        self.attn_dropout = nn.Dropout(0.0)

        self.agg_norm = nn.LayerNorm(num_query * self.feature_dim)
        self.agg_mlp = nn.Sequential(
            nn.Linear(num_query * self.feature_dim, self.feature_dim, bias=True),
            nn.SiLU(),
            nn.Linear(self.feature_dim, self.feature_dim, bias=True),
        )

        self.to(device)

    @torch.no_grad()
    def extract_tile_feats(self, path: str) -> torch.Tensor:
        with Image.open(path) as img:
            tiles = tile_image_to_tensors(img, self.tile_size, self.stride)
        tiles_t = tiles_to_batch(tiles).to(next(self.clip.parameters()).device)
        pixel_values = tiles_t * 0.5 + 0.5
        outputs = self.clip(pixel_values=pixel_values)
        return outputs.pooler_output

    def _get_tile_feats(self, path: str) -> torch.Tensor:
        key = cache_key(path, self.tile_size, self.stride)
        feats = load_tile_feats(self.cache_dir, key)
        if feats is not None:
            return feats.to(next(self.clip.parameters()).device)
        feats = self.extract_tile_feats(path)
        if self.cache_dir is not None:
            try:
                save_tile_feats(self.cache_dir, key, feats)
            except Exception:
                pass
        return feats

    def _aggregate(self, tile_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T, D = tile_feats.shape
        device = tile_feats.device
        pos_emb = _tile_pos_embedding(T, D, device)
        x = tile_feats + pos_emb

        q = self.q_proj(self.queries)
        kv = self.kv_proj(x)
        k, v = kv.chunk(2, dim=-1)

        attn = (q @ k.transpose(-1, -2)) / math.sqrt(D)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        agg = attn @ v
        agg = self.out_proj(agg)
        pooled = self.agg_mlp(self.agg_norm(agg.reshape(-1)))
        return agg, pooled

    def encode_from_paths(self, paths: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.clip.parameters()).device
        seq_list: List[torch.Tensor] = []
        pooled_list: List[torch.Tensor] = []
        for path in paths:
            tile_feats = self._get_tile_feats(path)
            agg, pooled = self._aggregate(tile_feats)
            seq_list.append(agg)
            pooled_list.append(pooled)
        f_s_seq = torch.stack(seq_list, dim=0)
        f_s_pooled = torch.stack(pooled_list, dim=0)
        return f_s_seq, f_s_pooled

    def encode(self, source: Union[str, List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(source, str):
            return self.encode_from_paths([source])
        return self.encode_from_paths(list(source))


class CLIPStyleEncoder(nn.Module):
    """
    旧版单图 CLIP 风格编码器（已废弃，仅保留兼容）。
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        super().__init__()
        try:
            from transformers import CLIPVisionModel, CLIPImageProcessor
        except ImportError:
            raise ImportError("transformers library required")

        self.clip = CLIPVisionModel.from_pretrained(model_name).to(device)
        self.clip.requires_grad_(False)
        self.clip.eval()

        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.feature_dim = self.clip.config.hidden_size

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        pixel_values = (images * 0.5 + 0.5).clamp(0, 1)
        pixel_values = F.interpolate(pixel_values, size=(224, 224), mode="bicubic", antialias=True)
        outputs = self.clip(pixel_values=pixel_values)
        return outputs.pooler_output


class OfflineStyleEncoder(nn.Module):
    """
    Offline 测试用风格编码器：轻量 CNN + FC（输出 M=1 的序列）。
    """

    def __init__(self, feature_dim: int = 768, image_size: int = 256):
        super().__init__()
        self.feature_dim = feature_dim

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, feature_dim)

    def encode(self, images: Union[torch.Tensor, List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(images, torch.Tensor):
            batch = [images]
        else:
            batch = list(images)

        pooled_list = [self._encode_single(x) for x in batch]
        f_s_pooled = torch.stack(pooled_list, dim=0)
        f_s_seq = f_s_pooled.unsqueeze(1)
        return f_s_seq, f_s_pooled

    def _encode_single(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x.unsqueeze(0))
        out = out.flatten(1)
        return self.fc(out).squeeze(0)


def build_style_encoder(
    offline_test: bool = False,
    model_name: str = "openai/clip-vit-base-patch32",
    feature_dim: int = 768,
    image_size: int = 256,
    tile_size: int = 224,
    stride: int = 168,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
    use_convnext: bool = True,
    image_height: int = 64,
    max_width: int = 512,
    pretrained: bool = True,
    fg_threshold: float = 0.0,
    num_heads: int = 12,
    dropout: float = 0.0,
    stage_b_checkpoint: Optional[str] = None,
) -> nn.Module:
    """
    创建风格编码器实例。
    """
    if offline_test:
        return OfflineStyleEncoder(feature_dim=feature_dim, image_size=image_size).to(device)
    if use_convnext:
        encoder = ConvNeXtStyleEncoder(
            feature_dim=feature_dim,
            num_query=4,
            image_height=image_height,
            max_width=max_width,
            pretrained=pretrained,
            fg_threshold=fg_threshold,
            num_heads=num_heads,
            dropout=dropout,
            device=device,
        )
        if stage_b_checkpoint is not None:
            ckpt = torch.load(stage_b_checkpoint, map_location=device, weights_only=True)
            state = ckpt.get("encoder_state_dict", ckpt.get("model_state_dict", ckpt))
            missing, unexpected = encoder.load_state_dict(state, strict=False)
            print(f"[style_encoder] loaded stage-B checkpoint: {stage_b_checkpoint}")
            if missing:
                print(f"  missing keys: {missing}")
            if unexpected:
                print(f"  unexpected keys: {unexpected}")
        return encoder
    return TiledCLIPStyleEncoder(
        model_name=model_name,
        feature_dim=feature_dim,
        tile_size=tile_size,
        stride=stride,
        cache_dir=cache_dir,
        num_query=4,
        device=device,
    )
