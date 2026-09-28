"""
阶段 B：对比 + 度量学习微调（SupCon + 可选 ArcFace）。

目的：让特征空间做到「同 writer 紧致、不同 writer 分离」，且**只编码风格不编码内容**。

核心：本项目的天然网格结构（每条公式在每种风格目录下都有同名文件）让我们可以
严格构造 writers × formulas 的矩形 batch：

                    formula_1  formula_2  ...  formula_M
        writer_1        ✅         ✅              ✅      ← 同行 = 正样本（同笔迹，内容不同）
        writer_2        ✅         ✅              ✅
           ...
        writer_P        ✅         ✅              ✅
                        ↑
                   同列 = **难负样本**（内容完全相同，只有笔迹不同）

难负样本是**压制内容泄露的最强手段**：模型必须忽略符号内容才能把同列的样本分开。
正样本（同行）与难负样本（同列）在同一个 batch 里同时出现。

启动（单卡）：
    python scripts/train_style_encoder.py --data_root ./data \
        --init_ckpt ./style_ckpt/dino/dino_final.pt \
        --height 64 --batch_writers 16 --batch_formulas 16 --epochs 150 \
        --out_dir ./style_ckpt/stage_b

启动（多卡）：
    torchrun --nproc_per_node=2 scripts/train_style_encoder.py ... （同上）

留一法评估配套：加 --holdout_writers 20 会把 20 个 writer 排除在训练之外，
并把名单写进 checkpoint，供 scripts/eval_style_encoder.py 直接读取。
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from typing import List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from data.style_augment import StyleAugment
from data.style_corpus import StyleCorpus
from data.transforms import pad_style_batch, style_transform_pil
from models.style_encoder import ConvNeXtStyleEncoder


# ════════════════════════════════════════════════════════════════════
#  数据：writers × formulas 网格
# ════════════════════════════════════════════════════════════════════

class GridBatchDataset(Dataset):
    """
    每个 item = 一个完整的 writer × formula 网格（P×M 张图）。

    用 index 播种随机数 ⇒ 与 DataLoader worker / epoch / 进程数无关，完全可复现。
    """

    def __init__(
        self,
        corpus: StyleCorpus,
        height: int = 64,
        max_width: Optional[int] = None,
        num_writers: int = 16,
        num_formulas: int = 16,
        length: int = 1000,
        seed: int = 0,
        augment: Optional[StyleAugment] = None,
    ):
        self.corpus = corpus
        self.height = int(height)
        self.max_width = int(max_width) if max_width else None
        self.num_writers = int(num_writers)
        self.num_formulas = int(num_formulas)
        self.length = int(length)
        self.seed = int(seed)
        self.augment = augment

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        rng = random.Random((self.seed * 1000003 + idx) % (2 ** 31))
        paths, wids, formulas, strict = self.corpus.sample_grid(
            self.num_writers, self.num_formulas, rng
        )

        tensors: List[torch.Tensor] = []
        for p in paths:
            try:
                with Image.open(p) as img:
                    t = style_transform_pil(
                        img, height=self.height, max_width=self.max_width, crop="center"
                    )
            except Exception:
                # 坏图：填一张全白图（前景 mask 全 0 ⇒ null token 兜底，不会 NaN）
                t = torch.ones(3, self.height, max(1, self.height))
            # 增广必须在 padding 之前施加（否则平移会把内容推进 padding 区）
            if self.augment is not None:
                t = self.augment(t, rng)
            tensors.append(t)

        return tensors, wids, formulas, strict


def grid_collate(batch):
    """
    batch 内每个 item 是一个网格；把它们摊平成一个 batch。

    Returns:
        (x (N,3,H,W_max), mask (N,W_max), writer_ids (N,) long, formulas list[str])
    """
    tensors: List[torch.Tensor] = []
    wids: List[int] = []
    formulas: List[str] = []
    for t_list, w_list, f_list, _ in batch:
        tensors.extend(t_list)
        wids.extend(w_list)
        formulas.extend(f_list)
    x, mask = pad_style_batch(tensors, pad_value=1.0, pad_side="right")
    return x, mask, torch.tensor(wids, dtype=torch.long), formulas


# ════════════════════════════════════════════════════════════════════
#  损失
# ════════════════════════════════════════════════════════════════════

def supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07):
    """
    Supervised Contrastive Loss（SupCon）。

    Args:
        features: (N, D)，**调用前需 L2 归一化**
        labels:   (N,) 长整型，同标签 = 正样本
        temperature: 温度（默认 0.07）
    """
    N = features.shape[0]
    if N < 2:
        return features.sum() * 0.0

    sim = torch.matmul(features, features.transpose(0, 1)) / temperature   # (N, N)
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()          # 数值稳定

    eye = torch.eye(N, dtype=torch.bool, device=features.device)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (~eye)        # (N, N)

    exp_sim = torch.exp(sim) * (~eye).to(sim.dtype)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
    n_pos = pos_mask.to(log_prob.dtype).sum(dim=1).clamp(min=1.0)
    mean_log_prob_pos = (pos_mask.to(log_prob.dtype) * log_prob).sum(dim=1) / n_pos
    return -mean_log_prob_pos.mean()


class ArcFaceHead(nn.Module):
    """
    ArcFace 分类头（**仅训练期使用，推理时丢弃**）。

    加性角度间隔惩罚：让同类在角度空间更紧致。对 15 类这种少类场景，
    单独用交叉熵容易「记住这 15 个人的特定符号形状」，所以只作辅助损失
    （默认权重 0.5），主损失是 SupCon。
    """

    def __init__(self, in_dim: int, num_classes: int, s: float = 30.0, m: float = 0.5):
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_classes = int(num_classes)
        self.s = float(s)
        self.m = float(m)
        self.weight = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, feat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat:   (N, D) 未归一化特征（内部会归一化）
            labels: (N,) 类 id
        Returns:
            交叉熵损失
        """
        cosine = F.linear(F.normalize(feat), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0.0, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        logits = torch.where(one_hot.bool(), phi, cosine) * self.s
        return F.cross_entropy(logits, labels.long())


# ════════════════════════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="阶段 B：SupCon + ArcFace 度量学习微调风格编码器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # 数据
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--extra_roots", type=str, nargs="*", default=[])
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--max_width", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--holdout_writers", type=int, default=0,
                   help="排除多少个 writer 不参与训练（留一法评估用，名单会写进 ckpt）")
    p.add_argument("--holdout_seed", type=int, default=42)
    # batch 构造
    p.add_argument("--batch_writers", type=int, default=16, help="网格行数 P")
    p.add_argument("--batch_formulas", type=int, default=16, help="网格列数 M（P*M = batch 图片数）")
    p.add_argument("--steps_per_epoch", type=int, default=500, help="每 epoch 多少个网格 batch")
    # 模型
    p.add_argument("--backbone", type=str, default="convnext_tiny")
    p.add_argument("--pretrained", type=int, default=0,
                   help="0=从 --init_ckpt 加载（阶段 A 权重）；1=ImageNet 初始化（无阶段 A 时用）")
    p.add_argument("--feature_dim", type=int, default=768)
    p.add_argument("--num_query", type=int, default=4)
    p.add_argument("--fg_threshold", type=float, default=0.0)
    p.add_argument("--init_ckpt", type=str, default="", help="阶段 A 产出的权重路径")
    # 损失
    p.add_argument("--supcon_temp", type=float, default=0.07)
    p.add_argument("--arcface_weight", type=float, default=0.5, help="0 = 不用 ArcFace")
    p.add_argument("--arcface_s", type=float, default=30.0)
    p.add_argument("--arcface_m", type=float, default=0.5)
    # 优化
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr_backbone", type=float, default=1e-4)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    # 增广
    p.add_argument("--augment", type=int, default=1, help="1=开启笔迹增广")
    p.add_argument("--aug_rotate", type=float, default=3.0)
    p.add_argument("--aug_shear", type=float, default=0.05)
    p.add_argument("--aug_translate", type=float, default=0.02)
    p.add_argument("--aug_morph_prob", type=float, default=0.15)
    # 其他
    p.add_argument("--out_dir", type=str, default="./style_ckpt/stage_b")
    p.add_argument("--save_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args()


def _lr_at(base: float, min_lr: float, step: int, total: int, warm: int) -> float:
    if step < warm:
        return base * (step + 1) / float(max(1, warm))
    prog = min(1.0, (step - warm) / float(max(1, total - warm)))
    return min_lr + (base - min_lr) * 0.5 * (1.0 + math.cos(math.pi * prog))


@torch.no_grad()
def extract(encoder: nn.Module, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """编码器 → f_s_pooled (N, D)。"""
    _, pooled = encoder(x, mask)
    return pooled


def main():
    args = parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    is_main = (local_rank == 0)

    torch.manual_seed(args.seed + local_rank)
    random.seed(args.seed + local_rank)

    # ── 语料 + hold-out 划分 ──
    corpus_all = StyleCorpus([args.data_root] + list(args.extra_roots), verbose=False)
    holdout: List[str] = []
    if args.holdout_writers > 0:
        holdout, seen = StyleCorpus.split_holdout(
            corpus_all.writer_names, args.holdout_writers, seed=args.holdout_seed
        )
        if is_main:
            print(f"[stageB] hold-out {len(holdout)} 个 writer（不参与训练）: {holdout[:5]}"
                  f"{' ...' if len(holdout) > 5 else ''}")
    corpus = StyleCorpus(
        [args.data_root] + list(args.extra_roots),
        exclude_writers=holdout, verbose=is_main,
    ) if holdout else corpus_all

    # ── 数据 ──
    augment = None
    if args.augment:
        augment = StyleAugment(
            translate=args.aug_translate, rotate_deg=args.aug_rotate,
            shear=args.aug_shear, morph_prob=args.aug_morph_prob,
            fg_threshold=args.fg_threshold,
        )
    dataset = GridBatchDataset(
        corpus, height=args.height, max_width=args.max_width,
        num_writers=args.batch_writers, num_formulas=args.batch_formulas,
        length=args.steps_per_epoch, seed=args.seed, augment=augment,
    )
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True) if world_size > 1 else None
    loader = DataLoader(
        dataset, batch_size=1, sampler=sampler, shuffle=(sampler is None),
        num_workers=args.num_workers, pin_memory=True, collate_fn=grid_collate,
        persistent_workers=args.num_workers > 0,
    )

    # ── 模型 ──
    encoder = ConvNeXtStyleEncoder(
        backbone=args.backbone, pretrained=bool(args.pretrained),
        feature_dim=args.feature_dim, height=args.height,
        num_query=args.num_query, fg_threshold=args.fg_threshold,
        max_width=args.max_width, device=str(device),
    )
    if args.init_ckpt:
        encoder.load_pretrained(args.init_ckpt, strict=False)
        if is_main:
            print(f"[stageB] 从阶段 A 权重初始化: {args.init_ckpt}")
    elif is_main:
        print("[stageB][warn] 没有 --init_ckpt，backbone 从 ImageNet/随机初始化起步，"
              "zero-shot 泛化能力会明显下降")

    arcface = None
    if args.arcface_weight > 0:
        arcface = ArcFaceHead(
            in_dim=args.feature_dim, num_classes=corpus.num_writers,
            s=args.arcface_s, m=args.arcface_m,
        ).to(device)
        if is_main:
            print(f"[stageB] ArcFace: {corpus.num_writers} 类")

    # 参数组：backbone 小 lr，聚合头（+ ArcFace）大 lr
    backbone_ids = {id(p) for p in encoder.backbone.parameters()}
    groups = [
        {"params": [p for p in encoder.parameters() if id(p) in backbone_ids],
         "lr": args.lr_backbone},
        {"params": [p for p in encoder.parameters() if id(p) not in backbone_ids],
         "lr": args.lr_head},
    ]
    if arcface is not None:
        groups.append({"params": list(arcface.parameters()), "lr": args.lr_head})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    enc_fwd = encoder
    if world_size > 1:
        enc_fwd = torch.nn.parallel.DistributedDataParallel(
            encoder, device_ids=[local_rank], broadcast_buffers=False
        )
        if arcface is not None:
            arcface = torch.nn.parallel.DistributedDataParallel(
                arcface, device_ids=[local_rank]
            )

    amp_dtype = {"no": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.amp]
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp == "fp16"))

    os.makedirs(args.out_dir, exist_ok=True)
    steps_per_epoch = len(loader)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warm_steps = max(1, steps_per_epoch * args.warmup_epochs)
    global_step = 0
    t0 = time.time()

    for epoch_idx in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch_idx)
        encoder.train()
        if arcface is not None:
            arcface.train()
        running_sup, running_arc, running_std, n = 0.0, 0.0, 0.0, 0

        for x, mask, wids, formulas in loader:
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            wids = wids.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype):
                pooled = extract(enc_fwd, x, mask)                 # (N, D)
                feat = F.normalize(pooled.float(), dim=-1)
                loss_sup = supcon_loss(feat, wids, args.supcon_temp)
                loss = loss_sup
                loss_arc = torch.zeros((), device=x.device)
                if arcface is not None:
                    loss_arc = arcface(pooled.float(), wids)
                    loss = loss + args.arcface_weight * loss_arc

            # 各组按自己的 base lr 同步缩放（backbone 组始终是 head 组的 1/N 量级）
            lr_scale = _lr_at(1.0, args.min_lr / max(1e-12, args.lr_backbone),
                              global_step, total_steps, warm_steps)
            base_lrs = [args.lr_backbone, args.lr_head] \
                + ([args.lr_head] if arcface is not None else [])
            for pg, base in zip(optimizer.param_groups, base_lrs):
                pg["lr"] = base * lr_scale

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters())
                + (list(arcface.parameters()) if arcface is not None else []),
                args.grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()

            running_sup += float(loss_sup.detach().item())
            running_arc += float(loss_arc.detach().item())
            running_std += float(pooled.detach().float().std(dim=0).mean().item())
            n += 1
            global_step += 1

            if is_main and global_step % args.log_every == 0:
                print(f"[stageB] epoch {epoch_idx} step {global_step} "
                      f"sup={running_sup/max(1,n):.4f} arc={running_arc/max(1,n):.4f} "
                      f"f_s_std={running_std/max(1,n):.4f} lr_b={args.lr_backbone*lr_scale:.2e} "
                      f"{(time.time()-t0)/60:.1f}min")
                running_sup, running_arc, running_std, n = 0.0, 0.0, 0.0, 0

        if is_main and ((epoch_idx + 1) % args.save_every == 0 or epoch_idx == args.epochs - 1):
            path = os.path.join(args.out_dir, f"style_encoder_epoch_{epoch_idx+1:04d}.pt")
            torch.save({
                "model": encoder.state_dict(),
                "epoch": epoch_idx + 1,
                "args": vars(args),
                "writers": corpus.writer_names,
                "holdout_writers": holdout,
                "height": args.height,
                "max_width": args.max_width,
                "feature_dim": args.feature_dim,
                "num_query": args.num_query,
                "fg_threshold": args.fg_threshold,
                "backbone": args.backbone,
            }, path)
            print(f"[stageB] saved {path}")

    if is_main:
        final = os.path.join(args.out_dir, "style_encoder_final.pt")
        torch.save({
            "model": encoder.state_dict(),
            "epoch": args.epochs,
            "args": vars(args),
            "writers": corpus.writer_names,
            "holdout_writers": holdout,
            "height": args.height,
            "max_width": args.max_width,
            "feature_dim": args.feature_dim,
            "num_query": args.num_query,
            "fg_threshold": args.fg_threshold,
            "backbone": args.backbone,
        }, final)
        print(f"[stageB] saved {final}")
        print("[stageB] 下一步：python scripts/eval_style_encoder.py "
              f"--data_root {args.data_root} --ckpt {final}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
