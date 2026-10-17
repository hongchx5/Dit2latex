"""训练器：FiT 可变分辨率训练循环。

- DataLoader 使用 FitDistributedSampler：每个 epoch 动态生成 batch 列表，
  同 batch 内尺寸一致，不同 batch 尺寸可变。
- 多卡（DDP）：trainer 内部用 DistributedDataParallel 包装 pipeline。
- EMA / checkpoint 始终操作裸 pipeline。
- TensorBoard、tqdm、验证、保存仅由 rank 0 执行。
"""

from __future__ import annotations

import os
import random
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
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
        self.config = config
        self.pipeline = pipeline
        self.rank = rank
        self.world_size = world_size
        self.is_main = rank == 0
        self.device = config.mode.device

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

        self.ema = EMA(pipeline, decay=config.training.ema_decay)

        self.global_step = 0
        self.best_loss = float("inf")
        self.nan_steps = 0
        self.consecutive_nan = 0
        self.window_skipped = 0

        mp = config.training.mixed_precision
        self.amp_enabled = mp in ("fp16", "bf16")
        self.amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(mp, None)
        if mp not in ("no", "fp16", "bf16"):
            raise ValueError(f"training.mixed_precision must be 'no'/'fp16'/'bf16', got {mp!r}")
        self.use_scaler = (mp == "fp16")

        if self.is_main:
            os.makedirs(config.training.checkpoint_dir, exist_ok=True)
            os.makedirs(config.training.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=config.training.log_dir)
        else:
            self.writer = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_scaler)

    def train(
        self,
        train_loader: DataLoader,
        train_sampler=None,
        val_loader: Optional[DataLoader] = None,
    ):
        cfg = self.config.training
        total_steps = cfg.total_steps
        log_every = cfg.log_every
        val_every = cfg.val_every
        save_every = cfg.save_every

        self.pipeline.train()

        pbar = None
        if self.is_main:
            pbar = tqdm(total=total_steps, desc="Training", initial=self.global_step)

        running_diff_loss = 0.0
        running_percep_loss = 0.0
        running_steps = 0

        while self.global_step < total_steps:
            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(self.global_step)

            for batch in train_loader:
                I_p, I_s, I_t, buckets, caption_ids, caption_mask = batch

                I_p = I_p.to(self.device)
                I_t = I_t.to(self.device)
                caption_ids = caption_ids.to(self.device)
                caption_mask = caption_mask.to(self.device)

                with torch.cuda.amp.autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
                    outputs = self.model(I_p, I_s, I_t, caption_ids, caption_mask)

                loss = outputs["loss"]
                diff_loss = outputs["diff_loss"].item()
                percep_loss = outputs["percep_loss"].item()

                loss_finite = bool(torch.isfinite(loss))
                if self.world_size > 1 and dist.is_available() and dist.is_initialized():
                    flag = torch.tensor([1.0 if loss_finite else 0.0], device=loss.device)
                    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                    loss_finite = float(flag.item()) > 0.5

                if not loss_finite:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.nan_steps += 1
                    self.consecutive_nan += 1
                    if self.is_main:
                        self.writer.add_scalar("Train/nan_steps", self.nan_steps, self.global_step)
                        print(f"[NaN] step {self.global_step}: loss={loss.item()} skipped")
                    if self.consecutive_nan >= cfg.nan_abort_steps:
                        raise RuntimeError(
                            f"Loss non-finite for {self.consecutive_nan} consecutive steps at step {self.global_step}."
                        )
                    continue
                self.consecutive_nan = 0

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    cfg.gradient_clip,
                )

                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scaler.get_scale() < scale_before:
                    self.window_skipped += 1

                self.scheduler.step()
                self.ema.update(self.pipeline)

                self.global_step += 1

                running_diff_loss += diff_loss
                running_percep_loss += percep_loss
                running_steps += 1

                if self.is_main and self.global_step % log_every == 0:
                    avg_diff = running_diff_loss / running_steps
                    avg_percep = running_percep_loss / running_steps
                    lr = self.scheduler.get_last_lr()[0]

                    self.writer.add_scalar("Loss/diffusion", avg_diff, self.global_step)
                    self.writer.add_scalar("Loss/perceptual", avg_percep, self.global_step)
                    self.writer.add_scalar(
                        "Loss/total",
                        avg_diff + cfg.perceptual_loss_weight * avg_percep,
                        self.global_step,
                    )
                    self.writer.add_scalar("Train/lr", lr, self.global_step)
                    self.writer.add_scalar("Train/grad_norm", grad_norm.item(), self.global_step)
                    self.writer.add_scalar("Train/scale", self.scaler.get_scale(), self.global_step)
                    self.writer.add_scalar("Train/skipped", self.window_skipped, self.global_step)

                    running_diff_loss = 0.0
                    running_percep_loss = 0.0
                    running_steps = 0
                    self.window_skipped = 0

                if pbar is not None:
                    postfix = {
                        "diff": f"{diff_loss:.4f}",
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                        "grad": f"{grad_norm.item():.2f}",
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
        if not self.is_main:
            return
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
        self.pipeline.load_state_dict(backup)
        self.pipeline.train()

    def _save_checkpoint(self, is_final: bool = False):
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

        if not is_final:
            self._prune_checkpoints(getattr(self.config.training, "save_max", 0))

    def _prune_checkpoints(self, save_max: int):
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
        train_loader: DataLoader,
        train_sampler=None,
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
        self.train(train_loader, train_sampler, val_loader)
