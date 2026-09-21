"""
风格一致性评估指标。

- 风格检索 Top-1/5：用 CLIP 特征检索最相似的风格类。
- Gram Matrix MSE：风格统计特征差异。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def compute_gram_matrix(feature_map: torch.Tensor) -> torch.Tensor:
    """
    计算 Gram 矩阵。

    Args:
        feature_map: (B, C, H, W)

    Returns:
        gram: (B, C, C)
    """
    B, C, H, W = feature_map.shape
    feat = feature_map.view(B, C, H * W)
    gram = feat @ feat.transpose(1, 2)  # (B, C, C)
    return gram / (C * H * W)


def gram_matrix_mse(
    gen_loader: DataLoader,
    style_loader: DataLoader,
    vgg_layer: str = "relu2_2",
    device: str = "cuda",
) -> float:
    """
    计算生成图像与风格参考图像之间的 Gram Matrix MSE。

    Args:
        gen_loader:   生成图像 dataloader。
        style_loader: 风格参考图像 dataloader。
        vgg_layer:    使用的 VGG 层。
        device:       计算设备。

    Returns:
        平均 Gram MSE。
    """
    try:
        from torchvision.models import vgg16, VGG16_Weights
    except ImportError:
        raise ImportError("torchvision required for Gram matrix computation")

    vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device)
    vgg.eval()

    # 找到 relu2_2 的深度
    layer_map = {"relu1_2": 4, "relu2_2": 9, "relu3_3": 16, "relu4_3": 23}
    end_idx = layer_map.get(vgg_layer, 9)

    @torch.no_grad()
    def gram_loader(loader: DataLoader):
        grams = []
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                imgs = batch[0]
            else:
                imgs = batch
            imgs = (imgs * 0.5 + 0.5).clamp(0, 1).to(device)
            feat = vgg[:end_idx + 1](imgs)
            grams.append(compute_gram_matrix(feat).cpu())
        return torch.cat(grams, dim=0)

    gram_gen = gram_loader(gen_loader)
    gram_style = gram_loader(style_loader)

    return F.mse_loss(gram_gen, gram_style).item()


def style_retrieval_accuracy(
    gen_loader: DataLoader,
    style_loader: DataLoader,
    style_labels: torch.Tensor,  # (N,) 每个风格参考图的类别标签
    device: str = "cuda",
    topk: tuple = (1, 5),
) -> dict:
    """
    风格检索准确率：用 CLIP 特征检索最相似的风格类别。

    Args:
        gen_loader:   生成图像 dataloader。
        style_loader: 风格参考图像 dataloader（按类别组织的）。
        style_labels: 每个风格参考图的类别标签。
        device:       计算设备。
        topk:         评估的 Top-K。

    Returns:
        {"top1": acc1, "top5": acc5}
    """
    try:
        from transformers import CLIPVisionModel, CLIPImageProcessor
    except ImportError:
        raise ImportError("transformers library required")

    clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip.eval()

    @torch.no_grad()
    def extract_clip(loader):
        features = []
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                imgs = batch[0]
            else:
                imgs = batch
            imgs = (imgs * 0.5 + 0.5).clamp(0, 1).to(device)
            imgs = F.interpolate(imgs, size=(224, 224), mode="bicubic", antialias=True)
            feat = clip(pixel_values=imgs).pooler_output
            features.append(feat.cpu())
        return torch.cat(features, dim=0)

    feat_gen = extract_clip(gen_loader)
    feat_style = extract_clip(style_loader)

    # 对于每个生成图像，找到最近邻的风格参考图
    sim = feat_gen @ feat_style.T  # (G, S)
    _, indices = sim.topk(max(topk), dim=-1)  # (G, K)

    results = {}
    for k in topk:
        pred_labels = style_labels[indices[:, :k]]
        correct = pred_labels.eq(style_labels[0].view(1, 1)).any(dim=1).sum()
        results[f"top{k}"] = correct.item() / feat_gen.shape[0]

    return results
