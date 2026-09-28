"""
风格编码器：从手写风格参考图像提取笔迹特征。

- 正常模式（默认）：ConvNeXtStyleEncoder —— 自训 ConvNeXt-T（**整图输入**）
  + 前景感知 4-query 聚合，输出：
      f_s_seq   (B, 4, feature_dim)   4 个聚合向量（DiT Cross-Attention 的 K/V）
      f_s_pooled (B, feature_dim)     concat 后 MLP 4 倍压缩的全局特征（AdaLN 条件）
  与旧版的关键差异：
      * 不再 tiling，整图等比缩放到固定高度 H（保留宽高比，绝不压扁）；
      * backbone 可训练（阶段 A/B/C 都要反传），因此**不做特征缓存**；
      * ConvNeXt-T 总 stride 从 32 改成 **16**（stem stride 4→2），否则 H=64 时
        特征图高度只有 2，4 个 query 几乎没有信息可用；
      * 前景 mask 用 max_pool 下采样（avg 会把细笔画稀释成 0）；
      * K/V 序列末尾追加 1 个可学习 null token（永远可见），避免整行被 mask 掉后
        softmax 出 NaN —— 本项目已踩过 caption mask 的同类坑，且当前用 bf16。
- 基线（可切换）：TiledCLIPStyleEncoder —— 冻结 CLIP ViT-B/32 + 224 tiling
  + 4-query 聚合，带 tile 特征缓存。保留用于效果对照。
- Offline 模式：简单 CNN → FC → 768 维向量（M=1）。

位置编码：
  - tiling 版：一行 tiles，alpha = i/(T-1)（T=1 时 alpha=0.5），pos = alpha * 2π，
    标准 1D 正弦位置编码，加到 tile 特征（K/V 侧）。
  - ConvNeXt 版：**归一化 2D 正弦**（row/col 归一化到 [0,1] 再 × 2π），
    不依赖绝对尺寸，对变宽友好。
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from data.transforms import (
    pad_style_batch,
    style_transform_pil,
    tile_image_to_tensors,
    tiles_to_batch,
)
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


# ══════════════════════════════════════════════════════════════════════
#  ConvNeXt 风格编码器（整图输入 + 前景感知聚合）
# ══════════════════════════════════════════════════════════════════════

# ConvNeXt 各规格 stage4 输出通道数（用于判断是否要做通道投影）
_CONVNEXT_FEAT_CH = {
    "convnext_tiny": 768,
    "convnext_small": 768,
    "convnext_base": 1024,
    "convnext_large": 1536,
}


def _forward_convnext_features(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    取 stage4 输出特征图 (B, C, H', W')（不经过 avgpool / classifier）。

    兼容两种 torchvision 布局：
      - stem 单独作为 `backbone.stem`，`backbone.features` = 4 个 stage；
      - stem 是 `backbone.features[0]`。
    """
    stem = getattr(backbone, "stem", None)
    if stem is not None:
        x = stem(x)
        feats = backbone.features
    else:
        feats = backbone.features
    for layer in feats:
        x = layer(x)
    return x


def _patch_convnext_stem(backbone: nn.Module, stem_stride: int = 2) -> bool:
    """
    **关键改造**：把 ConvNeXt stem 的 stride 从 4 改成 2，使总 stride = 16（而非 32）。

    stem 权重 shape 为 (out_ch, 3, 4, 4)，只由 (out_ch, in_ch, k, k) 决定，
    stride 是运行时参数 ⇒ **ImageNet 预训练权重仍可直接加载，无需丢弃**。

    kernel=4 时若 stride 从 4 改 2 而 padding 仍为 0，输出尺寸会变成
    floor((H-4)/2)+1 = H/2 - 1（H=64 → 31 而非 32），后续每层都会差一点，
    最终特征图与 mask 对不齐。所以这里同步把 padding 设为 (k - stride) // 2 = 1，
    保证输出严格是 H/2。

    Returns: 是否成功定位并修改 stem。
    """
    target: Optional[nn.Conv2d] = None
    for m in backbone.modules():
        if isinstance(m, nn.Conv2d) and m.in_channels == 3 and m.kernel_size[0] == 4:
            target = m
            break
    if target is None:
        return False
    k = target.kernel_size[0]
    target.stride = (stem_stride, stem_stride)
    target.padding = ((k - stem_stride) // 2,) * 2
    return True


def _infer_total_stride(backbone: nn.Module) -> int:
    """统计 backbone 的总下采样倍率（stem 的 stride × 各 stage 下采样层的 stride）。"""
    total = 1
    for m in backbone.modules():
        if not isinstance(m, nn.Conv2d):
            continue
        s = m.stride[0]
        if s <= 1:
            continue
        if m.in_channels == 3 and m.kernel_size[0] == 4:   # stem
            total *= s
        elif m.kernel_size[0] == 2:                        # stage 间的 downsample 层
            total *= s
    return max(1, total)


def _build_convnext_backbone(
    name: str = "convnext_tiny",
    pretrained: bool = True,
    stem_stride: int = 2,
) -> Tuple[nn.Module, int]:
    """
    构建 ConvNeXt backbone（去掉分类头），并把 stem stride 改成 2（总 stride 16）。

    Returns:
        (backbone, feat_channels)
    """
    try:
        from torchvision import models as tv_models
    except ImportError as e:  # pragma: no cover
        raise ImportError("torchvision required for ConvNeXtStyleEncoder. pip install torchvision") from e

    getter = {
        "convnext_tiny": tv_models.convnext_tiny,
        "convnext_small": tv_models.convnext_small,
        "convnext_base": tv_models.convnext_base,
        "convnext_large": tv_models.convnext_large,
    }.get(name)
    if getter is None:
        raise ValueError(
            f"unsupported backbone {name!r}; "
            f"choose from {sorted(_CONVNEXT_FEAT_CH)}"
        )

    weights_arg = None
    if pretrained:
        enum_cls = {
            "convnext_tiny": "ConvNeXt_Tiny_Weights",
            "convnext_small": "ConvNeXt_Small_Weights",
            "convnext_base": "ConvNeXt_Base_Weights",
            "convnext_large": "ConvNeXt_Large_Weights",
        }[name]
        try:
            weights_arg = getattr(tv_models, enum_cls).IMAGENET1K_V1
        except Exception:
            weights_arg = None

    model = None
    if weights_arg is not None:
        try:
            model = getter(weights=weights_arg)
        except Exception as e:      # 断网 / 权重下载失败
            warnings.warn(f"[style_encoder] 加载 ImageNet 预训练权重失败，将从零初始化: {e}")
    if model is None:
        try:
            model = getter(weights=None)
        except TypeError:           # 旧版 torchvision 只有 pretrained= 参数
            try:
                model = getter(pretrained=False)
            except Exception as e:
                raise RuntimeError(f"failed to build {name}: {e}") from e
        if pretrained:
            warnings.warn(
                f"[style_encoder] {name} 未加载到 ImageNet 预训练权重（从零初始化）。"
                "阶段 A(DINO) 的训练步数需要相应提高。"
            )

    # 分类头用不到（阶段 B 的 ArcFace 头在训练脚本里），直接丢掉省参数
    model.classifier = nn.Identity()

    if not _patch_convnext_stem(model, stem_stride=stem_stride):
        warnings.warn("[style_encoder] 未能定位 ConvNeXt stem（kernel=4 的首层 conv），"
                      "总 stride 可能仍是 32，请检查 torchvision 版本。")

    feat_ch = _CONVNEXT_FEAT_CH.get(name)
    if feat_ch is None:
        with torch.no_grad():
            side = stem_stride * 16
            feat_ch = int(_forward_convnext_features(model, torch.zeros(1, 3, side, side)).shape[1])
    return model, feat_ch


def _sincos_pos_embed_2d(
    h: int,
    w: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    归一化 2D 正弦位置编码：row/col 归一化到 [0,1] 后 × 2π。

    **不依赖绝对尺寸**，因此对任意宽度的公式图都友好（变宽友好）。

    Returns:
        pos: (h * w, dim)，行主序（与 flatten 后的 token 顺序一致）
    """
    half = dim // 2                      # 行 / 列各占 dim/2
    freq_dim = max(1, half // 2)         # 每组内每对维度一个频率（sin + cos 占 2 维）
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, freq_dim, dtype=torch.float32, device=device) / freq_dim
    )  # (freq_dim,)

    row = torch.arange(h, dtype=torch.float32, device=device) / max(1, h - 1) * (2.0 * math.pi)
    col = torch.arange(w, dtype=torch.float32, device=device) / max(1, w - 1) * (2.0 * math.pi)

    ang_r = row.view(h, 1, 1) * freqs.view(1, 1, freq_dim)   # (h, 1, f)
    ang_c = col.view(1, w, 1) * freqs.view(1, 1, freq_dim)   # (1, w, f)
    ang_r = ang_r.expand(h, w, freq_dim)
    ang_c = ang_c.expand(h, w, freq_dim)

    emb = torch.cat(
        [torch.sin(ang_r), torch.cos(ang_r), torch.sin(ang_c), torch.cos(ang_c)], dim=-1
    )  # (h, w, 4*freq_dim)
    emb = emb.reshape(h * w, -1)

    if emb.shape[-1] < dim:              # dim/2 为奇数时补零对齐
        pad = emb.new_zeros(h * w, dim - emb.shape[-1])
        emb = torch.cat([emb, pad], dim=-1)
    return emb.to(dtype).reshape(h * w, dim)


def _match_spatial(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """
    把 (B, 1, H0, W0) 的 mask 对齐到 (B, 1, h, w)：多则裁、少则补 0（背景）。

    卷积取整与 max_pool 取整在极端尺寸下可能差 1 格，这里兜底。
    """
    if x.shape[-2:] == (h, w):
        return x
    if x.shape[-2] > h:
        x = x[..., :h, :]
    if x.shape[-1] > w:
        x = x[..., :, :w]
    pad_h = h - x.shape[-2]
    pad_w = w - x.shape[-1]
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x


class ConvNeXtStyleEncoder(nn.Module):
    """
    整图输入 + ConvNeXt-T + 前景感知 4-query 聚合 的风格编码器。

    预处理（dataset 侧已完成）：等比缩放到高度 H，宽度动态，batch 内右侧 padding（+1.0 白）。

    前向：
        ConvNeXt-T(stem stride 2 ⇒ 总 stride 16) → stage4 特征图 (B, 768, H', W')
        → flatten + 归一化 2D 正弦位置编码
        → 末尾追加 1 个可学习 null token（永远可见，NaN 安全防护）
        → 前景 mask（max_pool 下采样）× valid_mask 抑制背景 / padding
        → num_query 个可学习 query 做 Cross-Attention → f_s_seq (B, M, D)
        → concat(M*D) → LayerNorm → MLP → f_s_pooled (B, D)

    输出接口（供 pipeline 使用，与 TiledCLIPStyleEncoder 完全一致）：
        encode(source) -> (f_s_seq (B, num_query, feature_dim), f_s_pooled (B, feature_dim))
    """

    #: 被抑制位置的 attention logits 加值。用 -1e4 而不是 -inf：
    #: bf16 下 -inf 经过 softmax 仍可能产出 NaN（本项目已在 caption mask 上踩过）。
    MASK_BIAS = -1.0e4

    def __init__(
        self,
        backbone: str = "convnext_tiny",
        pretrained: bool = True,
        feature_dim: int = 768,
        height: int = 64,
        num_query: int = 4,
        fg_threshold: float = 0.0,
        max_width: Optional[int] = None,
        pad_side: str = "right",
        stem_stride: int = 2,
        num_heads: int = 1,
        device: str = "cuda",
    ):
        super().__init__()
        self.backbone_name = backbone
        self.feature_dim = int(feature_dim)
        self.height = int(height)
        self.num_query = int(num_query)
        self.fg_threshold = float(fg_threshold)
        self.max_width = int(max_width) if max_width else 8 * int(height)
        self.pad_side = pad_side
        self.num_heads = int(num_heads)
        assert self.feature_dim % self.num_heads == 0, \
            f"feature_dim {self.feature_dim} must be divisible by num_heads {self.num_heads}"

        self.backbone, feat_ch = _build_convnext_backbone(
            backbone, pretrained=pretrained, stem_stride=stem_stride
        )
        self.total_stride = _infer_total_stride(self.backbone)
        # ConvNeXt-T 的 stage4 通道数恰好 768 = feature_dim，默认无需投影
        self.proj = (
            nn.Conv2d(feat_ch, self.feature_dim, kernel_size=1)
            if feat_ch != self.feature_dim else nn.Identity()
        )

        # ── 可学习聚合模块（参与梯度训练）──
        # 注意：必须用叶子张量构造 Parameter，否则梯度不会填充（optimizer 不更新）
        self.queries = nn.Parameter(torch.empty(num_query, self.feature_dim).normal_(0, 0.02))
        # null token：永远可见，防止整行 logits 被压到 -1e4 后出 NaN
        self.null_token = nn.Parameter(torch.empty(self.feature_dim).normal_(0, 0.02))

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

    # ── 权重加载 ─────────────────────────────────────────────────────

    def load_pretrained(self, path: str, strict: bool = True) -> None:
        """
        加载阶段 A/B 产出的权重（scripts/pretrain_style_dino.py / train_style_encoder.py）。

        兼容三种 ckpt 形态：{"model": state_dict} / {"state_dict": ...} / 裸 state_dict。
        strict=False 时会自动去掉 DINO/ArcFace 等训练期专用头的键。
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            state = ckpt.get("model", ckpt.get("state_dict", ckpt.get("encoder", ckpt)))
        else:
            state = ckpt
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]

        result = self.load_state_dict(state, strict=strict)
        if strict:
            return
        if result.missing_keys:
            print(f"[style_encoder] 缺失键 {len(result.missing_keys)} 个（随机初始化）: "
                  f"{result.missing_keys[:5]}{' ...' if len(result.missing_keys) > 5 else ''}")
        if result.unexpected_keys:
            print(f"[style_encoder] 未使用键 {len(result.unexpected_keys)} 个（训练期专用头）: "
                  f"{result.unexpected_keys[:5]}{' ...' if len(result.unexpected_keys) > 5 else ''}")

    def set_backbone_trainable(self, trainable: bool = True) -> None:
        """冻结 / 解冻 backbone（阶段 C 必须为 True；仅训聚合头时可先冻结）。"""
        self.backbone.requires_grad_(bool(trainable))

    # ── 输入预处理 ───────────────────────────────────────────────────

    def _prepare(
        self,
        source: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], str, Image.Image, Sequence],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        把各种输入形态归一化为 (x (B,3,H,W), valid_mask (B,W) 或 None)。

        支持：
          - (tensor, mask) 二元组：训练 collate 的输出
          - tensor (B,3,H,W) 或 (3,H,W)
          - 路径 str / PIL Image（推理直接喂图时用）
          - 上述元素的 list
        """
        # 训练 collate 的二元组：(x (B,3,H,W) 4D, mask (B,W) 2D)
        # 判据必须同时看两个元素的 ndim：否则「长度恰好为 2 的 tensor 列表」
        # （batch=2 时的旧接口）会被误判成 (tensor, mask)。
        if isinstance(source, (tuple, list)) and len(source) == 2 \
                and torch.is_tensor(source[0]) and torch.is_tensor(source[1]) \
                and source[0].dim() == 4 and source[1].dim() == 2:
            return source[0], source[1]

        items = source if isinstance(source, (list, tuple)) else [source]
        if len(items) == 0:
            raise ValueError("empty style input")

        tensors: List[torch.Tensor] = []
        for it in items:
            if isinstance(it, str) or isinstance(it, Path):
                tensors.append(style_transform_pil(
                    Image.open(it), height=self.height, max_width=self.max_width, crop="center"
                ))
            elif isinstance(it, Image.Image):
                tensors.append(style_transform_pil(
                    it, height=self.height, max_width=self.max_width, crop="center"
                ))
            elif torch.is_tensor(it):
                t = it
                if t.dim() == 3:
                    t = t.unsqueeze(0)
                tensors.append(t)
            else:
                raise TypeError(f"unsupported style input type: {type(it)!r}")

        if len(tensors) == 1:
            return tensors[0], None
        # 多张（宽度可能不同）→ 主进程内 padding（值 +1.0 = 背景色白）
        return pad_style_batch(
            tensors, pad_value=1.0, pad_side=self.pad_side, max_width=self.max_width
        )

    # ── 前向 ─────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:           (B, 3, H, W_max) in [-1, 1]，padding 为 +1.0（白）
            valid_mask:  (B, W_max) bool，True = 真实区域；None 表示无 padding
        Returns:
            (f_s_seq (B, num_query, D), f_s_pooled (B, D))
        """
        B, _, H, W = x.shape
        D = self.feature_dim
        nh = self.num_heads
        hd = D // nh

        feats = self.proj(_forward_convnext_features(self.backbone, x))   # (B, D, H', W')
        Hf, Wf = feats.shape[-2], feats.shape[-1]
        s = self.total_stride

        # ── 前景 mask：必须用 max_pool 下采样 ──
        # avg_pool 会把细笔画稀释成接近 0（等于没用）；max 能保住「这一格里有墨迹」。
        fg = (x.min(dim=1, keepdim=True).values < self.fg_threshold).to(torch.float32)
        fg_ds = _match_spatial(F.max_pool2d(fg, kernel_size=s, stride=s), Hf, Wf)
        keep = fg_ds > 0.5

        # padding 区填充的是背景色（白），fg 在那里自然为 0 ⇒ 前景 mask 已覆盖 pad mask。
        # valid_mask 作为额外保险，两者取逻辑与。
        if valid_mask is not None:
            vm = valid_mask.to(device=x.device, dtype=torch.float32).view(B, 1, 1, W)
            vm = vm.expand(B, 1, H, W)
            vm_ds = _match_spatial(F.max_pool2d(vm, kernel_size=s, stride=s), Hf, Wf)
            keep = keep & (vm_ds > 0.5)

        # ── tokens + 归一化 2D 正弦位置编码 ──
        tokens = feats.flatten(2).transpose(1, 2)                      # (B, N, D)
        tokens = tokens + _sincos_pos_embed_2d(Hf, Wf, D, x.device, feats.dtype).unsqueeze(0)

        # ── 追加 null token（永远可见）──
        null = self.null_token.view(1, 1, D).expand(B, 1, D)
        kv_in = torch.cat([tokens, null], dim=1)                       # (B, N+1, D)

        k, v = self.kv_proj(kv_in).chunk(2, dim=-1)                    # (B, N+1, D) ×2
        q = self.q_proj(self.queries)                                  # (M, D)

        keep_flat = keep.flatten(1)                                    # (B, N)
        keep_flat = torch.cat(
            [keep_flat, torch.ones(B, 1, dtype=torch.bool, device=x.device)], dim=1
        )                                                              # (B, N+1)

        if nh > 1:
            # 注意用 reshape 而不是 view：chunk 出来的 k/v 是非连续视图，view 会报错
            qh = q.reshape(self.num_query, nh, hd).transpose(0, 1).unsqueeze(0) \
                  .expand(B, nh, self.num_query, hd)
            kh = k.reshape(B, -1, nh, hd).transpose(1, 2)
            vh = v.reshape(B, -1, nh, hd).transpose(1, 2)
            logits = torch.matmul(qh, kh.transpose(-1, -2)) / math.sqrt(hd)   # (B, nh, M, N+1)
            logits = logits.masked_fill(~keep_flat.unsqueeze(1).unsqueeze(2), self.MASK_BIAS)
            attn = torch.softmax(logits, dim=-1)
            attn = self.attn_dropout(attn)
            agg = torch.matmul(attn, vh)                                      # (B, nh, M, hd)
            agg = agg.transpose(1, 2).reshape(B, self.num_query, D)
        else:
            logits = torch.matmul(
                q.unsqueeze(0).expand(B, self.num_query, D), k.transpose(-1, -2)
            ) / math.sqrt(D)                                                  # (B, M, N+1)
            logits = logits.masked_fill(~keep_flat.unsqueeze(1), self.MASK_BIAS)
            attn = torch.softmax(logits, dim=-1)
            attn = self.attn_dropout(attn)
            agg = torch.matmul(attn, v)                                       # (B, M, D)

        f_s_seq = self.out_proj(agg)                                   # (B, M, D)
        f_s_pooled = self.agg_mlp(self.agg_norm(f_s_seq.reshape(B, -1)))   # (B, D)
        return f_s_seq, f_s_pooled

    # ── 对外接口 ─────────────────────────────────────────────────────

    def encode(self, source) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        统一入口（pipeline 调用）。

        Args:
            source: (tensor (B,3,H,W_max), mask (B,W_max)) 二元组 / 单张 tensor /
                    路径 / PIL 图 / 上述元素的列表。
        Returns:
            (f_s_seq (B, num_query, feature_dim), f_s_pooled (B, feature_dim))
        """
        x, mask = self._prepare(source)
        return self.forward(x, mask)


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

    def encode(self, images: Union[torch.Tensor, List[torch.Tensor], Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: (B, 3, H, W) 堆叠 tensor、长度 B 的 tensor 列表，
                    或 collate 输出的 (tensor, mask) 二元组（mask 被忽略）。
        Returns:
            f_s_seq:   (B, 1, feature_dim)
            f_s_pooled: (B, feature_dim)
        """
        # 新版 collate 会把风格图打包成 (tensor (B,3,H,W), mask (B,W))；
        # offline 编码器不需要 mask。判据同时看 ndim，避免 batch=2 的 tensor 列表被误判。
        if isinstance(images, (tuple, list)) and len(images) == 2 \
                and torch.is_tensor(images[0]) and torch.is_tensor(images[1]) \
                and images[0].dim() == 4 and images[1].dim() == 2:
            images = images[0]

        if isinstance(images, torch.Tensor):
            batch = [images] if images.dim() == 3 else list(images.unbind(0))
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
    # ── 以下为 ConvNeXt 风格编码器参数 ──
    encoder_type: str = "convnext",
    backbone: str = "convnext_tiny",
    pretrained: bool = True,
    height: int = 64,
    max_width: Optional[int] = None,
    num_query: int = 4,
    fg_threshold: float = 0.0,
    pad_side: str = "right",
    init_ckpt: str = "",
    freeze_backbone: bool = False,
    num_heads: int = 1,
) -> nn.Module:
    """
    创建风格编码器实例。

    Args:
        encoder_type: "convnext"（默认，自训整图编码器）|
                      "clip_tiled"（冻结 CLIP + tiling 基线，对照用）|
                      "offline"（轻量 CNN 冒烟测试）
        init_ckpt:    阶段 A/B 产出的权重路径；空 = ImageNet 初始化 / 随机初始化
        freeze_backbone: True 时冻结 backbone（阶段 C 必须为 False）
    """
    encoder_type = (encoder_type or "convnext").lower()

    if offline_test:
        enc: nn.Module = OfflineStyleEncoder(feature_dim=feature_dim, image_size=image_size)
        return enc.to(device)

    if encoder_type == "convnext":
        enc = ConvNeXtStyleEncoder(
            backbone=backbone,
            pretrained=pretrained,
            feature_dim=feature_dim,
            height=height,
            num_query=num_query,
            fg_threshold=fg_threshold,
            max_width=max_width,
            pad_side=pad_side,
            num_heads=num_heads,
            device=device,
        )
        if init_ckpt:
            print(f"[style_encoder] 加载预训练风格编码器权重: {init_ckpt}")
            enc.load_pretrained(init_ckpt, strict=False)
        if freeze_backbone:
            enc.set_backbone_trainable(False)
        return enc

    if encoder_type == "offline":
        return OfflineStyleEncoder(feature_dim=feature_dim, image_size=image_size).to(device)

    if encoder_type == "clip_tiled":
        return TiledCLIPStyleEncoder(
            model_name=model_name,
            feature_dim=feature_dim,
            tile_size=tile_size,
            stride=stride,
            cache_dir=cache_dir,
            device=device,
        )

    raise ValueError(
        f"unknown style encoder type {encoder_type!r}; "
        "choose from 'convnext' / 'clip_tiled' / 'offline'"
    )
