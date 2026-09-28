"""
阶段 A：DINO 自监督预训练风格编码器 backbone。

目的：不依赖 writer 标签，让 ConvNeXt 学会「什么是笔画、什么是墨迹纹理」。
**zero-shot 泛化能力主要来自这一阶段**——有监督阶段只有 ≤15 个 writer，
类数太少，单靠分类/对比学不到连续的风格流形。

为什么是 DINO 而不是 MAE：ConvNeXt 是卷积架构，不兼容 MAE 的「遮挡 patch + 像素重建」范式。

裁剪策略（针对性改造）：
  - 公式图长宽比极端（1:1 ~ 1:10+），**裁剪必须保持宽高比，绝不压扁**；
  - 全局裁剪：高度 = H，宽度取原宽的 60%~100%（随机起点）；
  - 局部裁剪：取**水平片段**（左 / 中 / 右 / 随机），宽度 ≈ 原宽 / 4；
  - teacher 只吃全局裁剪，student 吃全局 + 局部（标准 DINO 做法）
    ⇒ 有效的自蒸馏对是「student 局部片段 ↔ teacher 全局图」。

启动（单卡）：
    python scripts/pretrain_style_dino.py --data_root ./data --height 64 \
        --epochs 200 --batch_size 32 --out_dir ./style_ckpt/dino

启动（多卡 DDP，torchrun）：
    torchrun --nproc_per_node=2 scripts/pretrain_style_dino.py \
        --data_root ./data --epochs 200 --batch_size 32 --out_dir ./style_ckpt/dino

混入外部手写数据（IAM / CVL）补多样性（阶段 A 完全不需要标签）：
    python scripts/pretrain_style_dino.py --data_root ./data \
        --extra_roots /data/IAM /data/CVL ...
"""

from __future__ import annotations

import argparse
import copy
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
from torchvision import transforms as T

from data.style_corpus import StyleCorpus
from data.transforms import pad_style_batch, style_resize_to_height
from models.style_encoder import ConvNeXtStyleEncoder

_IMAGENET_MEAN = None   # 项目统一用 [-1, 1]，不做 CLIP/ImageNet 归一化


# ════════════════════════════════════════════════════════════════════
#  数据：多裁剪
# ════════════════════════════════════════════════════════════════════

class DinoCropDataset(Dataset):
    """
    每张风格图 → [num_global 个全局裁剪] + [num_local 个局部裁剪]。

    所有裁剪都在「已等比缩放到高度 H」的图上取水平片段，因此**宽高比不被破坏**。
    每个裁剪的宽度不同 ⇒ batch 内用 pad_style_batch 统一（padding = +1.0 白），
    并由 style_mask 告知编码器哪些列是真实的。
    """

    def __init__(
        self,
        paths: Sequence[str],
        height: int = 64,
        max_width: Optional[int] = None,
        num_global: int = 1,
        num_local: int = 4,
        global_ratio: Tuple[float, float] = (0.6, 1.0),
        local_ratio: float = 0.25,
        seed: int = 0,
    ):
        self.paths = list(paths)
        self.height = int(height)
        self.max_width = int(max_width) if max_width else None
        self.num_global = int(num_global)
        self.num_local = int(num_local)
        self.global_ratio = global_ratio
        self.local_ratio = float(local_ratio)
        self.seed = int(seed)
        self.n_views = self.num_global + self.num_local
        if self.num_global < 1:
            raise ValueError("num_global must be >= 1（teacher 至少要看一个全局视图）")
        if self.num_local < 1:
            raise ValueError("num_local must be >= 1（否则没有可配对的自蒸馏视图）")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        # 按 index 播种 ⇒ 与 DataLoader worker / epoch 无关，完全可复现
        rng = random.Random((self.seed * 1000003 + idx) % (2 ** 31))
        with Image.open(self.paths[idx]) as img:
            base = style_resize_to_height(
                img, height=self.height, max_width=self.max_width, crop="center"
            )
        W0 = base.size[0]

        crops: List[Image.Image] = []
        # 全局裁剪：宽度取 60%~100%，起点随机
        for _ in range(self.num_global):
            w = max(8, int(round(W0 * rng.uniform(*self.global_ratio))))
            w = min(w, W0)
            x0 = rng.randint(0, max(0, W0 - w))
            crops.append(base.crop((x0, 0, x0 + w, self.height)))

        # 局部裁剪：水平片段（左 / 中 / 右 / 随机），宽度 ≈ W0/4
        base_lw = max(8, int(round(W0 * self.local_ratio)))
        for i in range(self.num_local):
            lw = max(8, int(round(base_lw * rng.uniform(0.75, 1.25))))
            lw = min(lw, W0)
            if i == 0:
                x0 = 0
            elif i == 1:
                x0 = max(0, (W0 - lw) // 2)
            elif i == 2:
                x0 = max(0, W0 - lw)
            else:
                x0 = rng.randint(0, max(0, W0 - lw))
            crops.append(base.crop((x0, 0, x0 + lw, self.height)))

        # PIL → tensor → [-1, 1]
        return [T.ToTensor()(c) * 2.0 - 1.0 for c in crops]


def dino_collate(batch):
    """batch: B 个样本 × n_views 个 crop → (B*n_views, 3, H, W_max) + mask。"""
    flat = [t for sample in batch for t in sample]
    return pad_style_batch(flat, pad_value=1.0, pad_side="right")


# ════════════════════════════════════════════════════════════════════
#  DINO head
# ════════════════════════════════════════════════════════════════════

class DINOHead(nn.Module):
    """
    DINO 投影头：MLP → L2 归一化 → prototype 层（out_dim = K 个原型）。

    最后一层用「forward 时归一化权重」代替 nn.utils.weight_norm（后者在新版 torch
    已废弃），数值等价（g 恒为 1）。
    """

    def __init__(
        self,
        in_dim: int = 768,
        out_dim: int = 8192,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.last_layer.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        w = F.normalize(self.last_layer.weight, dim=-1, p=2)   # 等价 weight_norm(g=1)
        return F.linear(x, w, None)


# ════════════════════════════════════════════════════════════════════
#  训练
# ════════════════════════════════════════════════════════════════════

class DINOTrainer:
    def __init__(self, student: nn.Module, head: nn.Module, args, device: torch.device):
        self.args = args
        self.device = device
        self.ddp = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.ddp else 0

        self.student_raw = student          # 裸模块（EMA 用）
        self.head_raw = head                # 裸模块（EMA / freeze_last_layer 用）
        self.student_fwd = student          # 前向用（DDP 包装后会被替换）
        self.head_fwd = head                # 前向用（DDP 包装后会被替换）
        self.teacher = copy.deepcopy(student).to(device)
        self.teacher_head = copy.deepcopy(head).to(device)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        for p in self.teacher_head.parameters():
            p.requires_grad_(False)
        self.teacher.eval()
        self.teacher_head.eval()

        self.center = torch.zeros(1, args.proto_k, device=device)
        self.center_momentum = args.center_momentum

        params = list(student.parameters()) + list(head.parameters())
        self.optimizer = torch.optim.AdamW(
            params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999)
        )
        self.grad_params = params

    # ── schedule ────────────────────────────────────────────────────

    def _momentum(self, epoch: float) -> float:
        """teacher EMA momentum：cosine 从 m_min 到 m_max。"""
        a, b = self.args.momentum_min, self.args.momentum_max
        prog = min(1.0, max(0.0, epoch / max(1, self.args.epochs)))
        return b - (b - a) * (math.cos(math.pi * prog) + 1.0) / 2.0

    def _teacher_temp(self, epoch: int) -> float:
        """teacher 温度 warmup：0.04 → 0.07。"""
        lo, hi = self.args.teacher_temp_min, self.args.teacher_temp_max
        n = max(1, self.args.warmup_teacher_temp_epochs)
        if epoch >= n:
            return hi
        return lo + (hi - lo) * (epoch / float(n))

    def _lr_at(self, step: int, steps_per_epoch: int) -> float:
        """warmup（前 warmup_epochs 个 epoch）+ cosine 衰减。"""
        total = max(1, steps_per_epoch * self.args.epochs)
        warm = max(1, steps_per_epoch * self.args.warmup_epochs)
        if step < warm:
            return self.args.lr * (step + 1) / float(warm)
        prog = min(1.0, (step - warm) / float(max(1, total - warm)))
        return self.args.min_lr + (self.args.lr - self.args.min_lr) * \
            0.5 * (1.0 + math.cos(math.pi * prog))

    # ── EMA / center ────────────────────────────────────────────────

    @torch.no_grad()
    def _ema_update(self, m: float):
        for ps, pt in zip(self.student_raw.parameters(), self.teacher.parameters()):
            pt.data.mul_(m).add_(ps.detach().data, alpha=1.0 - m)
        for bs, bt in zip(self.student_raw.buffers(), self.teacher.buffers()):
            bt.data.copy_(bs.detach().data)
        for ps, pt in zip(self.head_raw.parameters(), self.teacher_head.parameters()):
            pt.data.mul_(m).add_(ps.detach().data, alpha=1.0 - m)

    @torch.no_grad()
    def _update_center(self, teacher_logits: torch.Tensor):
        batch_center = teacher_logits.mean(dim=0, keepdim=True)
        if self.ddp:
            dist.all_reduce(batch_center)
            batch_center /= float(dist.get_world_size())
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1.0 - self.center_momentum)

    @torch.no_grad()
    def _cancel_gradients_last_layer(self, epoch: int):
        """前 freeze_last_layer 个 epoch 冻结 prototype 层（防止原型塌缩到少数几个）。"""
        if epoch >= self.args.freeze_last_layer_epochs:
            return
        for p in self.head_raw.last_layer.parameters():
            p.grad = None

    # ── 一步 ────────────────────────────────────────────────────────

    def train_step(self, x, mask, epoch: float, epoch_idx: int) -> torch.Tensor:
        B_total = x.shape[0]
        n_views = self.args.num_global + self.args.num_local
        B = B_total // n_views

        # student：全部视图
        _, s_pooled = self.student_fwd(x, mask)              # (B*n_views, D)
        s_logits = self.head_fwd(s_pooled)                   # (B*n_views, K)

        # teacher：只吃全局视图（第 0 ~ num_global-1 个）
        with torch.no_grad():
            g = x[: B * self.args.num_global]
            g_mask = mask[: B * self.args.num_global]
            _, t_pooled = self.teacher(g, g_mask)            # (B*num_global, D)
            t_logits = self.teacher_head(t_pooled)           # (B*num_global, K)
            self._update_center(t_logits)
            t_probs = F.softmax((t_logits - self.center) / self._teacher_temp(epoch_idx), dim=-1)

        # 配对：student 的**局部视图** ↔ teacher 的全局视图（同图）
        # student 的全局视图与 teacher 输入完全相同，跳过（自蒸馏退化为恒等映射）
        s_logits = s_logits.view(B, n_views, -1)
        s_local = s_logits[:, self.args.num_global:, :]      # (B, n_local, K)
        n_local = s_local.shape[1]
        if self.args.num_global == 1:
            t_probs = t_probs.view(B, 1, -1).expand(B, n_local, -1)
        else:
            t_probs = t_probs.view(B, self.args.num_global, -1).mean(dim=1, keepdim=True) \
                             .expand(B, n_local, -1)
        loss = -(t_probs.reshape(-1, t_probs.shape[-1])
                 * F.log_softmax(s_local.reshape(-1, s_local.shape[-1]) / self.args.student_temp,
                                 dim=-1)).sum(dim=-1).mean()
        return loss

    def step(self, x, mask, epoch: float, epoch_idx: int, step: int, steps_per_epoch: int):
        """完整一步：lr schedule + 反传 + 裁剪 + EMA。AMP 由调用方在外面开。"""
        lr = self._lr_at(step, steps_per_epoch)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.train_step(x, mask, epoch, epoch_idx)
        loss.backward()
        self._cancel_gradients_last_layer(epoch_idx)
        torch.nn.utils.clip_grad_norm_(self.grad_params, self.args.grad_clip)
        self.optimizer.step()
        self._ema_update(self._momentum(epoch))
        return float(loss.detach().item()), lr


# ════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="阶段 A：DINO 自监督预训练风格编码器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # 数据
    p.add_argument("--data_root", type=str, default="./data", help="数据根目录（style*/）")
    p.add_argument("--extra_roots", type=str, nargs="*", default=[], help="外部手写数据根目录")
    p.add_argument("--height", type=int, default=64, help="风格图基准高度 H")
    p.add_argument("--max_width", type=int, default=512, help="风格图宽度上限")
    p.add_argument("--num_workers", type=int, default=8)
    # 模型
    p.add_argument("--backbone", type=str, default="convnext_tiny")
    p.add_argument("--pretrained", type=int, default=1, help="1=用 ImageNet-1K 权重初始化")
    p.add_argument("--feature_dim", type=int, default=768)
    p.add_argument("--num_query", type=int, default=4)
    p.add_argument("--fg_threshold", type=float, default=0.0)
    p.add_argument("--init_ckpt", type=str, default="", help="断点续训的风格编码器权重")
    # DINO
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32, help="每卡的**图片数**（非 crop 数）")
    p.add_argument("--num_global", type=int, default=1, help="全局裁剪数（teacher 只吃这个）")
    p.add_argument("--num_local", type=int, default=4, help="局部裁剪数（水平片段）")
    p.add_argument("--global_ratio", type=float, nargs=2, default=[0.6, 1.0])
    p.add_argument("--local_ratio", type=float, default=0.25)
    p.add_argument("--proto_k", type=int, default=8192, help="原型个数 K")
    p.add_argument("--student_temp", type=float, default=0.1)
    p.add_argument("--teacher_temp_min", type=float, default=0.04)
    p.add_argument("--teacher_temp_max", type=float, default=0.07)
    p.add_argument("--warmup_teacher_temp_epochs", type=int, default=30)
    p.add_argument("--momentum_min", type=float, default=0.996)
    p.add_argument("--momentum_max", type=float, default=0.9995)
    p.add_argument("--center_momentum", type=float, default=0.9)
    p.add_argument("--freeze_last_layer_epochs", type=int, default=1)
    # 优化
    p.add_argument("--lr", type=float, default=5e-4, help="基准 lr（会按 batch/256 缩放）")
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.04)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    # 其他
    p.add_argument("--out_dir", type=str, default="./style_ckpt/dino")
    p.add_argument("--save_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()

    # ── DDP（torchrun 启动）──
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

    # ── 数据 ──
    corpus = StyleCorpus([args.data_root] + list(args.extra_roots), verbose=is_main)
    paths = [p for _, _, p in corpus.all_images()]
    if is_main:
        print(f"[dino] {len(paths)} 张风格图, {corpus.num_writers} 个 writer "
              f"（阶段 A 不用 writer 标签）")

    dataset = DinoCropDataset(
        paths, height=args.height, max_width=args.max_width,
        num_global=args.num_global, num_local=args.num_local,
        global_ratio=tuple(args.global_ratio), local_ratio=args.local_ratio,
        seed=args.seed,
    )
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True) if world_size > 1 else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        shuffle=(sampler is None), drop_last=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=dino_collate, persistent_workers=args.num_workers > 0,
    )

    # ── 模型 ──
    student = ConvNeXtStyleEncoder(
        backbone=args.backbone, pretrained=bool(args.pretrained),
        feature_dim=args.feature_dim, height=args.height,
        num_query=args.num_query, fg_threshold=args.fg_threshold,
        max_width=args.max_width, device=str(device),
    )
    if args.init_ckpt:
        student.load_pretrained(args.init_ckpt, strict=False)
        if is_main:
            print(f"[dino] 从 {args.init_ckpt} 续训")

    head = DINOHead(in_dim=args.feature_dim, out_dim=args.proto_k).to(device)

    trainer = DINOTrainer(student, head, args, device)

    if world_size > 1:
        # 前向必须用 DDP 包装后的模块（梯度 all-reduce）；EMA 仍基于裸模块参数
        trainer.student_fwd = torch.nn.parallel.DistributedDataParallel(
            student, device_ids=[local_rank], broadcast_buffers=False
        )
        trainer.head_fwd = torch.nn.parallel.DistributedDataParallel(
            head, device_ids=[local_rank]
        )

    # lr 按 batch 缩放（DINO 惯例：lr = base * batch_size / 256）
    eff_batch = args.batch_size * world_size
    args.lr = args.lr * eff_batch / 256.0
    trainer.args.lr = args.lr
    if is_main:
        print(f"[dino] world_size={world_size}, 每卡 batch={args.batch_size}, "
              f"有效 batch={eff_batch}, lr={args.lr:.2e}")

    amp_dtype = {"no": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.amp]
    # bf16 与 fp32 同指数位，不需要 GradScaler；fp16 需要
    use_scaler = (args.amp == "fp16")
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    os.makedirs(args.out_dir, exist_ok=True)
    steps_per_epoch = len(loader)
    global_step = 0
    t0 = time.time()

    for epoch_idx in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch_idx)
        trainer.student_fwd.train()
        trainer.head_fwd.train()
        running = 0.0
        n = 0

        for it, (x, mask) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype):
                loss = trainer.train_step(x, mask, float(epoch_idx) + it / max(1, steps_per_epoch),
                                          epoch_idx)
            # lr schedule + 反传（AMP 在外面开，所以在这里手动走一遍 step 流程）
            lr = trainer._lr_at(global_step, steps_per_epoch)
            for pg in trainer.optimizer.param_groups:
                pg["lr"] = lr
            trainer.optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(trainer.optimizer)
            trainer._cancel_gradients_last_layer(epoch_idx)
            torch.nn.utils.clip_grad_norm_(trainer.grad_params, args.grad_clip)
            scaler.step(trainer.optimizer)
            scaler.update()
            trainer._ema_update(
                trainer._momentum(float(epoch_idx) + it / max(1, steps_per_epoch))
            )

            running += float(loss.detach().item())
            n += 1
            global_step += 1

            if is_main and global_step % args.log_every == 0:
                print(f"[dino] epoch {epoch_idx} step {global_step} "
                      f"loss={running/max(1,n):.4f} lr={lr:.2e} "
                      f"temp_t={trainer._teacher_temp(epoch_idx):.3f} "
                      f"{(time.time()-t0)/60:.1f}min")
                running, n = 0.0, 0

        if is_main and ((epoch_idx + 1) % args.save_every == 0 or epoch_idx == args.epochs - 1):
            path = os.path.join(args.out_dir, f"dino_epoch_{epoch_idx+1:04d}.pt")
            torch.save({
                "model": student.state_dict(),
                "head": head.state_dict(),
                "epoch": epoch_idx + 1,
                "args": vars(args),
            }, path)
            print(f"[dino] saved {path}")

    if is_main:
        final = os.path.join(args.out_dir, "dino_final.pt")
        torch.save({
            "model": student.state_dict(),
            "head": head.state_dict(),
            "epoch": args.epochs,
            "args": vars(args),
        }, final)
        print(f"[dino] saved {final}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
