
#!/usr/bin/env python
"""阶段 B：对比 + 度量学习微调风格编码器。

用法示例：
    python scripts/train_style_encoder.py \
        --data_root ./data \
        --dino_checkpoint ./checkpoints/style_dino/dino_backbone_final.pt \
        --output_dir ./checkpoints/style_encoder \
        --batch_size 256 \
        --epochs 100 \
        --device cuda
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.style_encoder import ConvNeXtStyleEncoder
from data.style_dataset import ContrastiveStyleDataset
from losses.style_losses import SupConLoss, ArcFaceLoss


class ProjectionHead(nn.Module):
    """对比学习用投影头（训练时用，推理可丢弃）。"""

    def __init__(self, in_dim: int = 768, hidden_dim: int = 2048, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return F.normalize(x, dim=-1, p=2)


def build_model(
    image_height: int = 64,
    max_width: int = 512,
    pretrained: bool = True,
    fg_threshold: float = 0.0,
    num_heads: int = 12,
    proj_dim: int = 128,
    num_classes: int = 20,
    device: str = "cuda",
) -> Tuple[ConvNeXtStyleEncoder, ProjectionHead, ArcFaceLoss]:
    encoder = ConvNeXtStyleEncoder(
        feature_dim=768,
        num_query=4,
        image_height=image_height,
        max_width=max_width,
        pretrained=pretrained,
        fg_threshold=fg_threshold,
        num_heads=num_heads,
        device=device,
    )
    proj = ProjectionHead(in_dim=768, out_dim=proj_dim).to(device)
    arcface = ArcFaceLoss(in_features=768, num_classes=num_classes).to(device)
    return encoder, proj, arcface


def load_dino_backbone(encoder: ConvNeXtStyleEncoder, checkpoint_path: str):
    """加载阶段 A DINO backbone 权重。"""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = ckpt.get("backbone_state_dict", ckpt.get("model_state_dict", ckpt))
    missing, unexpected = encoder.backbone.load_state_dict(state, strict=False)
    print(f"Loaded DINO backbone from {checkpoint_path}")
    if missing:
        print(f"  missing keys: {missing}")
    if unexpected:
        print(f"  unexpected keys: {unexpected}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage B: contrastive fine-tuning of style encoder")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--dino_checkpoint", type=str, default=None, help="Path to DINO backbone checkpoint")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/style_encoder")
    parser.add_argument("--image_height", type=int, default=64)
    parser.add_argument("--max_width", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--backbone_lr", type=float, default=1e-4)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--supcon_temp", type=float, default=0.07)
    parser.add_argument("--arcface_weight", type=float, default=0.5, help="Auxiliary ArcFace loss weight")
    parser.add_argument("--num_classes", type=int, default=20, help="Number of writers for ArcFace head")
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--pretrained", action="store_true", default=True)
    return parser.parse_args()


def build_optimizer(
    encoder: ConvNeXtStyleEncoder,
    proj: ProjectionHead,
    arcface: ArcFaceLoss,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    params = [
        {"params": encoder.parameters(), "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": proj.parameters(), "lr": head_lr, "weight_decay": weight_decay},
        {"params": arcface.parameters(), "lr": head_lr, "weight_decay": weight_decay},
    ]
    return torch.optim.AdamW(params, betas=(0.9, 0.999))


def cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds = ContrastiveStyleDataset(
        data_roots=args.data_root,
        image_height=args.image_height,
        max_width=args.max_width,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=ds.collate_fn,
        drop_last=True,
    )

    encoder, proj, arcface = build_model(
        image_height=args.image_height,
        max_width=args.max_width,
        pretrained=args.pretrained,
        proj_dim=args.proj_dim,
        num_classes=args.num_classes,
        device=device,
    )

    if args.dino_checkpoint is not None:
        load_dino_backbone(encoder, args.dino_checkpoint)

    supcon = SupConLoss(temperature=args.supcon_temp).to(device)
    optimizer = build_optimizer(encoder, proj, arcface, args.backbone_lr, args.head_lr, args.weight_decay)

    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    total_steps = len(loader) * args.epochs
    warmup_steps = len(loader) * args.warmup_epochs
    step = 0

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        proj.train()
        arcface.train()

        epoch_supcon = 0.0
        epoch_arc = 0.0
        epoch_total = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            (anchor_im, anchor_mask), (pos_im, pos_mask), (hard_im, hard_mask), ids = batch
            anchor_im = anchor_im.to(device)
            anchor_mask = anchor_mask.to(device)
            pos_im = pos_im.to(device)
            pos_mask = pos_mask.to(device)
            hard_im = hard_im.to(device)
            hard_mask = hard_mask.to(device)
            anchor_wid = ids["anchor_writer_id"].to(device)
            pos_wid = ids["positive_writer_id"].to(device)
            hard_wid = ids["hard_writer_id"].to(device)

            # anchor 和 positive 同 writer，hard negative 不同 writer
            images = torch.cat([anchor_im, pos_im, hard_im], dim=0)  # (3B, 3, H, W)
            masks = torch.cat([anchor_mask, pos_mask, hard_mask], dim=0)  # (3B, W)
            labels = torch.cat([anchor_wid, pos_wid, hard_wid], dim=0)

            f_s_seq, f_s_pooled = encoder.encode(images, masks)

            z = proj(f_s_pooled)
            loss_supcon = supcon(z, labels)

            loss_arc = arcface(F.normalize(f_s_pooled, dim=-1, p=2), labels)

            loss = loss_supcon + args.arcface_weight * loss_arc

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in encoder.parameters() if p.requires_grad] +
                list(proj.parameters()) + list(arcface.parameters()),
                1.0,
            )
            optimizer.step()

            step += 1
            lr_now = cosine_lr(step, total_steps, warmup_steps, args.backbone_lr, args.min_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr_now

            epoch_supcon += loss_supcon.item()
            epoch_arc += loss_arc.item()
            epoch_total += loss.item()

            if step % args.log_every == 0:
                writer.add_scalar("Train/supcon", loss_supcon.item(), step)
                writer.add_scalar("Train/arcface", loss_arc.item(), step)
                writer.add_scalar("Train/total", loss.item(), step)
                writer.add_scalar("Train/lr", lr_now, step)

            pbar.set_postfix({
                "supcon": f"{loss_supcon.item():.4f}",
                "arc": f"{loss_arc.item():.4f}",
                "total": f"{loss.item():.4f}",
                "lr": f"{lr_now:.2e}",
            })

        n = len(loader)
        print(f"Epoch {epoch}: supcon={epoch_supcon/n:.4f}, arc={epoch_arc/n:.4f}, total={epoch_total/n:.4f}, lr={lr_now:.2e}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_path = Path(args.output_dir) / f"style_encoder_epoch_{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "encoder_state_dict": encoder.state_dict(),
                "proj_state_dict": proj.state_dict(),
                "arcface_state_dict": arcface.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, save_path)
            print(f"Saved checkpoint: {save_path}")

    final_path = Path(args.output_dir) / "style_encoder_final.pt"
    torch.save({"encoder_state_dict": encoder.state_dict()}, final_path)
    print(f"Saved final encoder: {final_path}")

    writer.close()


if __name__ == "__main__":
    main()
