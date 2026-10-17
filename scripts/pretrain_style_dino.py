#!/usr/bin/env python
"""阶段 A：DINO 自监督预训练风格编码器 backbone。

用法示例：
    python scripts/pretrain_style_dino.py \
        --data_root ./data \
        --output_dir ./checkpoints/style_dino \
        --image_height 64 \
        --max_width 512 \
        --batch_size 128 \
        --epochs 100 \
        --lr 5e-4 \
        --device cuda
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models as torchvision_models
from tqdm import tqdm

from data.style_dataset import DINODataset
from losses.style_losses import DINOLoss


class DINOProjector(nn.Module):
    """DINO 投影头：3 层 MLP + bottleneck + L2 norm。"""

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        out_dim: int = 65536,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1.0)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DINOStyleEncoder(nn.Module):
    """DINO 风格编码器：ConvNeXt-T backbone + student/teacher projector。"""

    def __init__(
        self,
        image_height: int = 64,
        pretrained: bool = True,
        out_dim: int = 65536,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
    ):
        super().__init__()
        # ConvNeXt-Tiny with stem stride 2 -> total stride 16
        weights = torchvision_models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        convnext = torchvision_models.convnext_tiny(weights=weights)
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
        self.backbone = convnext.features  # output (B, 768, H/16, W/16)
        self.image_height = image_height

        # Global average pooling over spatial dims, but weighted by foreground mask.
        self.student_proj = DINOProjector(768, hidden_dim, bottleneck_dim, out_dim)
        self.teacher_proj = DINOProjector(768, hidden_dim, bottleneck_dim, out_dim)
        self._init_teacher()

    def _init_teacher(self):
        for p_s, p_t in zip(self.student_proj.parameters(), self.teacher_proj.parameters()):
            p_t.data.copy_(p_s.data)
            p_t.requires_grad = False

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        feat = self.backbone(x)  # (B, 768, H', W')
        # spatial GAP
        feat = feat.flatten(2).mean(dim=2)  # (B, 768)
        return feat

    @torch.no_grad()
    def forward_teacher(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._extract(x)
        out = self.teacher_proj(feat)
        return F.softmax(out / 0.04, dim=-1)

    def forward_student(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._extract(x)
        out = self.student_proj(feat)
        return out

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        for p_s, p_t in zip(self.student_proj.parameters(), self.teacher_proj.parameters()):
            p_t.data = momentum * p_t.data + (1 - momentum) * p_s.data


def build_dino_loader(
    data_root: str,
    image_height: int,
    max_width: int,
    batch_size: int,
    num_workers: int,
    include_unlabeled: bool,
) -> DataLoader:
    ds = DINODataset(
        data_roots=data_root,
        image_height=image_height,
        max_width=max_width,
        include_unlabeled=include_unlabeled,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=ds.collate_fn,
        drop_last=True,
    )


def dino_forward(model: DINOStyleEncoder, crops: List[Tuple[torch.Tensor, torch.Tensor]], is_teacher: bool) -> torch.Tensor:
    """crops: list of (image, valid_mask). 返回 (B, N_crops, out_dim) softmax/logits。"""
    outputs = []
    for images, _ in crops:
        images = images.to(next(model.parameters()).device)
        if is_teacher:
            out = model.forward_teacher(images)
        else:
            out = model.forward_student(images)
        outputs.append(out)
    return torch.stack(outputs, dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage A: DINO pre-training for style encoder")
    parser.add_argument("--data_root", type=str, required=True, help="Path to data root containing style*/ and optionally print/")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/style_dino")
    parser.add_argument("--image_height", type=int, default=64)
    parser.add_argument("--max_width", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--momentum_teacher", type=float, default=0.996)
    parser.add_argument("--momentum_teacher_final", type=float, default=0.9995)
    parser.add_argument("--student_temp", type=float, default=0.1)
    parser.add_argument("--teacher_temp", type=float, default=0.04)
    parser.add_argument("--teacher_temp_final", type=float, default=0.07)
    parser.add_argument("--teacher_temp_warmup_epochs", type=int, default=10)
    parser.add_argument("--out_dim", type=int, default=65536)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--bottleneck_dim", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--include_unlabeled", action="store_true", default=True)
    parser.add_argument("--pretrained", action="store_true", default=True)
    return parser.parse_args()


def cosine_schedule(epoch: int, total_epochs: int, start: float, end: float) -> float:
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * epoch / total_epochs))


def build_optimizer(model: DINOStyleEncoder, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    # no weight decay on biases and teacher
    params_no_wd = []
    params_wd = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "teacher_proj" in name:
            continue  # teacher has no grad
        if len(p.shape) == 1 or name.endswith(".bias"):
            params_no_wd.append(p)
        else:
            params_wd.append(p)
    return torch.optim.AdamW(
        [{"params": params_wd, "weight_decay": weight_decay}, {"params": params_no_wd, "weight_decay": 0.0}],
        lr=lr,
        betas=(0.9, 0.999),
    )


def main():
    args = parse_args()
    import math
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    loader = build_dino_loader(
        args.data_root,
        args.image_height,
        args.max_width,
        args.batch_size,
        args.num_workers,
        args.include_unlabeled,
    )

    model = DINOStyleEncoder(
        image_height=args.image_height,
        pretrained=args.pretrained,
        out_dim=args.out_dim,
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
    ).to(device)

    criterion = DINOLoss(
        out_dim=args.out_dim,
        student_temp=args.student_temp,
        teacher_temp=args.teacher_temp,
        teacher_temp_warmup_epochs=args.teacher_temp_warmup_epochs,
        teacher_temp_final=args.teacher_temp_final,
    ).to(device)

    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    total_steps = len(loader) * args.epochs

    step = 0
    for epoch in range(1, args.epochs + 1):
        momentum = cosine_schedule(epoch - 1, args.epochs, args.momentum_teacher, args.momentum_teacher_final)
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for crops in pbar:
            # crops: list of (image, mask); first is global, rest are local
            teacher_out = dino_forward(model, [crops[0]], is_teacher=True)
            student_out = dino_forward(model, crops, is_teacher=False)
            loss = criterion(student_out, teacher_out, epoch=epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_teacher(momentum)

            epoch_loss += loss.item()
            step += 1
            lr_now = optimizer.param_groups[0]["lr"]

            # linear warmup + cosine decay
            if step < args.warmup_epochs * len(loader):
                lr_scale = step / (args.warmup_epochs * len(loader))
            else:
                progress = (step - args.warmup_epochs * len(loader)) / max(1, total_steps - args.warmup_epochs * len(loader))
                lr_scale = args.min_lr / args.lr + (1 - args.min_lr / args.lr) * (0.5 * (1 + math.cos(math.pi * progress)))
            for g in optimizer.param_groups:
                g["lr"] = args.lr * lr_scale

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr_now:.2e}", "m": f"{momentum:.4f}"})

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}, lr={lr_now:.2e}, momentum={momentum:.4f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_path = Path(args.output_dir) / f"dino_epoch_{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, save_path)
            print(f"Saved checkpoint: {save_path}")

    # Save only backbone for stage B / C initialization
    final_backbone_path = Path(args.output_dir) / "dino_backbone_final.pt"
    torch.save({"backbone_state_dict": model.backbone.state_dict()}, final_backbone_path)
    print(f"Saved final backbone: {final_backbone_path}")


if __name__ == "__main__":
    main()
