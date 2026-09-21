"""
FID / LPIPS 评估。

- FID: 使用 torchvision 的 InceptionV3 计算 FID。
- LPIPS: 使用预训练 LPIPS 网络。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def compute_fid(
    real_loader: DataLoader,
    gen_loader: DataLoader,
    device: str = "cuda",
) -> float:
    """
    计算两个数据集的 FID。

    使用 torchvision 的 InceptionV3 提取 2048 维特征。

    Args:
        real_loader: 真实图像 dataloader。
        gen_loader:  生成图像 dataloader。
        device:      计算设备。

    Returns:
        FID 分数（越低越好）。
    """
    try:
        from torchvision.models import inception_v3, Inception_V3_Weights
    except ImportError:
        raise ImportError("torchvision required for FID computation")

    inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False).to(device)
    inception.fc = nn.Identity()  # 去掉分类头，输出 pool3 特征
    inception.eval()

    @torch.no_grad()
    def extract_features(loader: DataLoader) -> tuple:
        features = []
        for batch in tqdm(loader, desc="Extracting features"):
            if isinstance(batch, (list, tuple)):
                imgs = batch[0]
            else:
                imgs = batch
            imgs = imgs.to(device)

            # Inception 需要 299×299
            if imgs.shape[-1] != 299:
                imgs = F.interpolate(imgs, size=(299, 299), mode="bilinear", antialias=True)

            feat = inception(imgs)
            features.append(feat.cpu())
        return torch.cat(features, dim=0)

    feat_real = extract_features(real_loader)
    feat_gen = extract_features(gen_loader)

    mu_real = feat_real.mean(dim=0)
    sigma_real = torch.cov(feat_real.T)
    mu_gen = feat_gen.mean(dim=0)
    sigma_gen = torch.cov(feat_gen.T)

    diff = mu_real - mu_gen
    # 数值稳定性
    covmean = torch.linalg.sqrtm(sigma_real @ sigma_gen)
    if torch.is_complex(covmean):
        covmean = covmean.real

    fid = diff @ diff + torch.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid.item()


def compute_lpips(
    real_loader: DataLoader,
    gen_loader: DataLoader,
    device: str = "cuda",
) -> float:
    """
    计算 LPIPS 距离。

    Args:
        real_loader: 真实图像 dataloader。
        gen_loader:  生成图像 dataloader。
        device:      计算设备。

    Returns:
        平均 LPIPS 分数（越低越好）。
    """
    try:
        import lpips
    except ImportError:
        raise ImportError("lpips library required. Install with: pip install lpips")

    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.eval()

    total_lpips = 0.0
    count = 0

    with torch.no_grad():
        for (batch_real, batch_gen) in tqdm(zip(real_loader, gen_loader), desc="LPIPS"):
            if isinstance(batch_real, (list, tuple)):
                real = batch_real[0]
            else:
                real = batch_real
            if isinstance(batch_gen, (list, tuple)):
                gen = batch_gen[0]
            else:
                gen = batch_gen

            real = real.to(device)
            gen = gen.to(device)

            # LPIPS 期望 [-1, 1]
            score = lpips_fn(real, gen)
            total_lpips += score.sum().item()
            count += real.shape[0]

    return total_lpips / count
