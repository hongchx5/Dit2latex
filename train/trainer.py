"""
训练器：完整的训练循环（动态分桶 + 可选多卡 DDP）。

- 每个桶维护独立的 DataLoader（batch 内同桶、尺寸统一）。
- "一个桶训练若干步再换桶"：桶切换序列由共享随机种子确定性生成，
  所有 DDP 进程在同一 global_step 切到同一桶，无需通信、天然同步。
- 多卡（DDP）：trainer 内部用 DistributedDataParallel 包装 pipeline；
  EMA / checkpoint 始终操作裸 pipeline（无 module. 前缀）；
  TensorBoard、tqdm、验证、保存只由 rank 0 执行。
"""

from __future__ import annotations

import os
import random
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from config.config_loader import Config
from models.pipeline import DiTtolatexPipeline
from train.optimizer import build_optimizer, build_scheduler
from train.checkpoint import save_checkpoint, load_checkpoint, EMA


class Trainer:
    def __init__(
        self,
        config: Config,
        pipeline: DiTtolatexPipeline,
        rank: int = 0,
        world_size: int = 1,
    ):
        """
        Args:
            config: 全局配置。
            pipeline: 未包装的裸管线（EMA/checkpoint 操作的对象）。
            rank: 当前进程 rank（单卡时为 0）。
            world_size: 总进程数（单卡时为 1）。
        """
        self.config = config
        self.pipeline = pipeline
        self.rank = rank
        self.world_size = world_size
        self.is_main = rank == 0
        self.device = config.mode.device

        # DDP 包装：world_size > 1 时启用（nccl 需要 device_ids）
        if world_size > 1:
            self.model = DistributedDataParallel(pipeline, device_ids=[rank])
        else:
            self.model = pipeline

        self.optimizer = build_optimizer(
            self.model,
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            total_steps=config.training.total_steps,
            warmup_steps=config.training.warmup_steps,
            min_lr=config.training.min_lr,
        )

        # EMA 始终基于裸 pipeline
        self.ema = EMA(pipeline, decay=config.training.ema_decay)

        self.global_step = 0
        self.best_loss = float("inf")

        # NaN 监控（P0）：累计 NaN 步数 / 连续 NaN 步数 / 本日志窗口被 scaler 跳过的步数
        self.nan_steps = 0
        self.consecutive_nan = 0
        self.window_skipped = 0

        # 混合精度（P1）：按配置显式指定 dtype。
        # - fp16：需要 GradScaler（上限 65504，易溢出）
        # - bf16：与 fp32 同指数位（上限 3.4e38），不需要 GradScaler
        mp = config.training.mixed_precision
        self.amp_enabled = mp in ("fp16", "bf16")
        self.amp_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }.get(mp, None)
        if mp not in ("no", "fp16", "bf16"):
            raise ValueError(
                f"training.mixed_precision must be one of 'no'/'fp16'/'bf16', got {mp!r}"
            )
        self.use_scaler = (mp == "fp16")

        if self.is_main:
            os.makedirs(config.training.checkpoint_dir, exist_ok=True)
            os.makedirs(config.training.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=config.training.log_dir)
        else:
            self.writer = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_scaler)

    # ── 桶循环辅助 ───────────────────────────────────────────────────

    def _build_bucket_sequence(
        self,
        bucket_ids: list,
        total_steps: int,
        bucket_steps: int,
    ) -> list:
        """
        用共享随机种子生成确定性桶切换序列，保证所有 DDP 进程同步换桶。

        序列长度 = 换桶次数 + 2（覆盖全程，末尾截断）。相邻桶不重复。
        """
        rng = random.Random(self.config.mode.seed)
        seq = []
        prev = None
        n_switches = total_steps // max(1, bucket_steps) + 2
        for _ in range(n_switches):
            pool = [b for b in bucket_ids if b != prev] if len(bucket_ids) > 1 else bucket_ids
            b = rng.choice(pool)
            seq.append(b)
            prev = b
        return seq

    def _get_iter(self, bucket, bucket_loaders, bucket_samplers):
        """重建指定桶的 DataLoader 迭代器（DistributedSampler 需 set_epoch 保证 shuffle 变化）。"""
        sampler = bucket_samplers.get(bucket)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(self.global_step)
        return iter(bucket_loaders[bucket])

    # ── 主训练循环 ───────────────────────────────────────────────────

    def train(
        self,
        bucket_loaders: Dict[Tuple[int, int], DataLoader],
        bucket_samplers: Optional[Dict[Tuple[int, int], Optional[DistributedSampler]]] = None,
        val_loader: Optional[DataLoader] = None,
    ):
        """
        Args:
            bucket_loaders: {bucket (H, W): DataLoader}，每个 loader 的 batch 内同桶。
            bucket_samplers: {bucket: DistributedSampler 或 None}，用于各进程数据分片与 shuffle。
            val_loader: 验证集 loader（可选，仅 rank 0 使用）。
        """
        cfg = self.config.training
        total_steps = cfg.total_steps
        log_every = cfg.log_every
        val_every = cfg.val_every
        save_every = cfg.save_every
        bucket_steps = cfg.bucket_steps
        bucket_samplers = bucket_samplers or {}

        if not bucket_loaders:
            raise ValueError("bucket_loaders must not be empty")

        self.pipeline.train()

        # 确定性桶序列（所有进程一致）
        bucket_ids = list(bucket_loaders.keys())
        bucket_sequence = self._build_bucket_sequence(bucket_ids, total_steps, bucket_steps)
        seq_idx = 0
        current_bucket = bucket_sequence[0]
        data_iter = self._get_iter(current_bucket, bucket_loaders, bucket_samplers)
        steps_in_bucket = 0

        pbar = None
        if self.is_main:
            pbar = tqdm(total=total_steps, desc=f"Training [bucket {current_bucket}]",
                        initial=self.global_step)

        running_diff_loss = 0.0
        running_percep_loss = 0.0
        running_steps = 0

        while self.global_step < total_steps:
            # 一桶训练 bucket_steps 步后换桶（序列驱动，所有进程同步）
            if steps_in_bucket >= bucket_steps:
                seq_idx += 1
                current_bucket = bucket_sequence[min(seq_idx, len(bucket_sequence) - 1)]
                data_iter = self._get_iter(current_bucket, bucket_loaders, bucket_samplers)
                steps_in_bucket = 0
                if pbar is not None:
                    pbar.set_description(f"Training [bucket {current_bucket}]")

            try:
                I_p, I_s, I_t, buckets, caption_ids, caption_mask = next(data_iter)
            except StopIteration:
                data_iter = self._get_iter(current_bucket, bucket_loaders, bucket_samplers)
                I_p, I_s, I_t, buckets, caption_ids, caption_mask = next(data_iter)

            I_p = I_p.to(self.device)
            I_t = I_t.to(self.device)
            caption_ids = caption_ids.to(self.device)
            caption_mask = caption_mask.to(self.device)
            # I_s：路径列表（正常模式）或 tensor 列表（offline 模式），无需搬运

            with torch.cuda.amp.autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
                outputs = self.model(I_p, I_s, I_t, caption_ids, caption_mask)

            loss = outputs["loss"]
            diff_loss = outputs["diff_loss"].item()
            percep_loss = outputs["percep_loss"].item()

            # ── P0：非有限损失保护 ──────────────────────────────────
            # 前向一旦产出 inf/NaN 就不能反传：权重会停在触发溢出的状态，
            # 之后每个 batch 都溢出（本次崩溃即空转 75750 步的根因）。
            loss_finite = bool(torch.isfinite(loss))
            if self.world_size > 1 and dist.is_available() and dist.is_initialized():
                # DDP：各 rank 拿到的是不同样本，是否 NaN 必须全体一致，
                # 否则有的 rank 反传、有的不反传，backward 的 all-reduce 会挂死。
                flag = torch.tensor([1.0 if loss_finite else 0.0], device=loss.device)
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                loss_finite = float(flag.item()) > 0.5

            if not loss_finite:
                self.optimizer.zero_grad(set_to_none=True)
                self.nan_steps += 1
                self.consecutive_nan += 1
                if self.is_main:
                    self.writer.add_scalar("Train/nan_steps", self.nan_steps, self.global_step)
                    print(f"[NaN] step {self.global_step}: loss={loss.item()} "
                          f"(nan_steps={self.nan_steps}, consecutive={self.consecutive_nan}), skipped")
                if self.consecutive_nan >= cfg.nan_abort_steps:
                    raise RuntimeError(
                        f"Loss has been non-finite for {self.consecutive_nan} consecutive steps "
                        f"(total NaN steps: {self.nan_steps}); aborting at step {self.global_step}. "
                        f"Check Train/scale and Train/grad_norm: a collapsing scale with a "
                        f"stable grad_norm means dtype overflow."
                    )
                continue
            self.consecutive_nan = 0

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # 梯度裁剪前记录梯度范数
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                cfg.gradient_clip,
            )

            # 记录 scaler 是否跳过了这一步（梯度含 inf/NaN 时 scale 会减半）
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scaler.get_scale() < scale_before:
                self.window_skipped += 1

            self.scheduler.step()
            self.ema.update(self.pipeline)

            self.global_step += 1
            steps_in_bucket += 1

            running_diff_loss += diff_loss
            running_percep_loss += percep_loss
            running_steps += 1

            # TensorBoard 记录（每 log_every 步汇总，仅 rank 0）
            if self.is_main and self.global_step % log_every == 0:
                avg_diff = running_diff_loss / running_steps
                avg_percep = running_percep_loss / running_steps
                lr = self.scheduler.get_last_lr()[0]

                self.writer.add_scalar("Loss/diffusion", avg_diff, self.global_step)
                self.writer.add_scalar("Loss/perceptual", avg_percep, self.global_step)
                # P5-B3：total 必须乘感知损失权重，否则日志量级与实际训练损失不符
                self.writer.add_scalar(
                    "Loss/total",
                    avg_diff + cfg.perceptual_loss_weight * avg_percep,
                    self.global_step,
                )
                self.writer.add_scalar("Train/lr", lr, self.global_step)
                self.writer.add_scalar("Train/grad_norm", grad_norm.item(), self.global_step)
                self.writer.add_scalar("Train/scale", self.scaler.get_scale(), self.global_step)
                self.writer.add_scalar("Train/skipped", self.window_skipped, self.global_step)
                self.writer.add_scalar("Train/bucket_h", current_bucket[0], self.global_step)
                self.writer.add_scalar("Train/bucket_w", current_bucket[1], self.global_step)
                running_diff_loss = 0.0
                running_percep_loss = 0.0
                running_steps = 0
                self.window_skipped = 0

            # 终端显示：每步实时更新（percep 仅在开启感知损失时显示）
            if pbar is not None:
                postfix = {
                    "diff": f"{diff_loss:.4f}",
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                    "grad": f"{grad_norm.item():.2f}",
                    "bucket": f"{current_bucket[0]}x{current_bucket[1]}",
                }
                if cfg.perceptual_loss_weight > 0:
                    postfix["percep"] = f"{percep_loss:.4f}"
                pbar.set_postfix(postfix)
                pbar.update(1)

            if val_loader is not None and self.global_step % val_every == 0:
                self._validate(val_loader)

            if self.global_step % save_every == 0:
                self._save_checkpoint()

            if self.global_step >= total_steps:
                break

        if pbar is not None:
            pbar.close()
        self._save_checkpoint(is_final=True)
        if self.writer is not None:
            self.writer.close()
        if self.is_main:
            print(f"Training complete. Final checkpoint at step {self.global_step}.")

    def _validate(self, val_loader: DataLoader):
        """在验证集上计算平均损失（用 EMA 参数，仅 rank 0 执行）。"""
        if not self.is_main:
            return
        # P5-C1：先用 EMA 权重覆盖，验证结束后必须还原训练权重，
        # 否则每验证一次训练权重就被 EMA 权重替换一次。
        backup = {k: v.detach().clone() for k, v in self.pipeline.state_dict().items()}
        self.ema.apply_to(self.pipeline)
        self.pipeline.eval()

        total_val_loss = 0.0
        total_val_diff = 0.0
        total_val_percep = 0.0
        num_batches = 0
        with torch.no_grad():
            for I_p, I_s, I_t, buckets, caption_ids, caption_mask in val_loader:
                I_p = I_p.to(self.device)
                I_t = I_t.to(self.device)
                caption_ids = caption_ids.to(self.device)
                caption_mask = caption_mask.to(self.device)
                with torch.cuda.amp.autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
                    outputs = self.pipeline(I_p, I_s, I_t, caption_ids, caption_mask)
                total_val_loss += outputs["loss"].item()
                total_val_diff += outputs["diff_loss"].item()
                total_val_percep += outputs["percep_loss"].item()
                num_batches += 1
                if num_batches >= 20:
                    break

        avg_loss = total_val_loss / max(1, num_batches)
        avg_diff = total_val_diff / max(1, num_batches)
        avg_percep = total_val_percep / max(1, num_batches)

        self.writer.add_scalar("Val/loss", avg_loss, self.global_step)
        self.writer.add_scalar("Val/diffusion", avg_diff, self.global_step)
        self.writer.add_scalar("Val/perceptual", avg_percep, self.global_step)

        print(f"\n[Step {self.global_step}] Val loss: {avg_loss:.6f} (diff: {avg_diff:.4f}, percep: {avg_percep:.4f})")
        self.pipeline.load_state_dict(backup)   # P5-C1：还原训练权重
        self.pipeline.train()

    def _save_checkpoint(self, is_final: bool = False):
        """保存完整训练状态（仅 rank 0；保存裸 pipeline，无 module. 前缀）。"""
        if not self.is_main:
            return
        suffix = "final" if is_final else f"step_{self.global_step:07d}"
        path = os.path.join(self.config.training.checkpoint_dir, f"checkpoint_{suffix}.pt")
        save_checkpoint(
            path=path,
            model=self.pipeline,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ema_state=self.ema.state_dict(),
            step=self.global_step,
        )
        print(f"Checkpoint saved: {path}")

        # P5-C4：只保留最近 save_max 个周期性 checkpoint（final 不参与清理）
        if not is_final:
            self._prune_checkpoints(getattr(self.config.training, "save_max", 0))

    def _prune_checkpoints(self, save_max: int):
        """删除旧的 checkpoint_step_*.pt，只保留最近 save_max 个（save_max<=0 表示不限制）。"""
        if save_max is None or save_max <= 0:
            return
        ckpt_dir = self.config.training.checkpoint_dir
        if not os.path.isdir(ckpt_dir):
            return
        step_files = []
        for name in os.listdir(ckpt_dir):
            if name.startswith("checkpoint_step_") and name.endswith(".pt"):
                stem = name[len("checkpoint_step_"):-len(".pt")]
                if stem.isdigit():
                    step_files.append((int(stem), os.path.join(ckpt_dir, name)))
        if len(step_files) <= save_max:
            return
        step_files.sort(key=lambda x: x[0])
        for _, path in step_files[: len(step_files) - save_max]:
            try:
                os.remove(path)
                print(f"Checkpoint removed (keep last {save_max}): {path}")
            except OSError as e:
                print(f"Failed to remove checkpoint {path}: {e}")

    def resume(
        self,
        path: str,
        bucket_loaders: Dict[Tuple[int, int], DataLoader],
        bucket_samplers: Optional[Dict[Tuple[int, int], Optional[DistributedSampler]]] = None,
        val_loader: Optional[DataLoader] = None,
    ):
        result = load_checkpoint(
            path=path,
            model=self.pipeline,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )
        self.global_step = result.get("step", 0)
        if result.get("ema_state"):
            self.ema.load_state_dict(result["ema_state"])
        if self.is_main:
            print(f"Resumed from {path} at step {self.global_step}")
        self.train(bucket_loaders, bucket_samplers, val_loader)
