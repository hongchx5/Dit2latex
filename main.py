"""DiTtolatex CLI 入口（FiT 可变分辨率版）。"""

from __future__ import annotations

import argparse
import random
import os
import re
import shutil
import socket
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from config.config_loader import load_config, set_config, Config
from models.vae import build_vae
from models.style_encoder import build_style_encoder
from models.caption_encoder import CaptionEncoder, build_vocab
from models.dit.dit import DiT
from models.pipeline import DiTtolatexPipeline
from diffusion.noise_schedule import NoiseSchedule
from losses.perceptual_loss import build_perceptual_loss
from data.dataset import HandwrittenFormulaDataset, bucket_collate, FitDistributedSampler
from train.trainer import Trainer
from inference.inferencer import Inferencer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_pipeline(config: Config) -> DiTtolatexPipeline:
    device = config.mode.device

    vae = build_vae(
        offline_test=config.mode.offline_test,
        pretrained_path=config.model.vae.pretrained_path,
        latent_dim=config.model.vae.latent_dim,
        f=config.model.vae.f,
        device=device,
    )

    style_enc = build_style_encoder(
        offline_test=config.mode.offline_test,
        model_name=config.model.style_encoder.model_name,
        feature_dim=config.model.style_encoder.feature_dim,
        image_size=config.data.image_size,
        tile_size=config.data.tile_size,
        stride=config.data.tile_stride,
        cache_dir=config.data.style_cache_dir,
        device=device,
        use_convnext=config.model.style_encoder.use_convnext,
        image_height=config.model.style_encoder.image_height,
        max_width=config.model.style_encoder.max_width,
        pretrained=config.model.style_encoder.pretrained,
        fg_threshold=config.model.style_encoder.fg_threshold,
        num_heads=config.model.style_encoder.num_heads,
        dropout=config.model.style_encoder.dropout,
    )

    dit = DiT(
        hidden_dim=config.model.dit.hidden_dim,
        num_heads=config.model.dit.num_heads,
        depth=config.model.dit.depth,
        mlp_ratio=config.model.dit.mlp_ratio,
        patch_size=config.model.dit.patch_size,
        in_channels=config.model.dit.in_channels,
        out_channels=config.model.dit.out_channels,
        latent_dim=config.model.dit.latent_dim,
        context_dim=config.model.style_encoder.feature_dim,
        use_caption=config.model.caption.enabled,
    ).to(device)

    caption_enc = None
    if config.model.caption.enabled:
        _, id2token = build_vocab(config.data.dictionary_path)
        vocab_size = len(id2token)
        caption_enc = CaptionEncoder(
            vocab_size=vocab_size,
            hidden_dim=config.model.style_encoder.feature_dim,
            max_len=config.model.caption.max_len,
            num_layers=config.model.caption.num_layers,
        ).to(device)

    noise_schedule = NoiseSchedule(
        num_timesteps=config.diffusion.num_timesteps,
        beta_schedule=config.diffusion.beta_schedule,
        device=device,
    )

    perceptual_loss = (
        build_perceptual_loss(device=device)
        if config.training.perceptual_loss_weight > 0 else None
    )

    pipeline = DiTtolatexPipeline(
        vae=vae,
        style_encoder=style_enc,
        caption_encoder=caption_enc,
        dit=dit,
        noise_schedule=noise_schedule,
        perceptual_loss=perceptual_loss,
        perceptual_loss_weight=config.training.perceptual_loss_weight,
        cfg_dropout_rate=config.training.cfg_dropout_rate,
        device=device,
    )

    return pipeline


def _make_exp_dir(base_log_dir: str, config_path: Optional[str] = None) -> str:
    os.makedirs(base_log_dir, exist_ok=True)
    ids = [
        int(m.group(1))
        for d in os.listdir(base_log_dir)
        if (m := re.fullmatch(r"exp_(d+)", d)) and os.path.isdir(os.path.join(base_log_dir, d))
    ]
    exp_id = max(ids, default=0) + 1
    exp_dir = os.path.join(base_log_dir, f"exp_{exp_id:04d}")
    os.makedirs(exp_dir, exist_ok=True)

    if config_path and os.path.exists(config_path):
        try:
            shutil.copy(config_path, os.path.join(exp_dir, "config.yaml"))
        except OSError as e:
            print(f"[warn] failed to copy config to {exp_dir}: {e}")
    return exp_dir


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_train_dataset(config: Config) -> HandwrittenFormulaDataset:
    fit = config.data.fit
    return HandwrittenFormulaDataset(
        data_root=config.data.data_root,
        max_patch=fit.max_patch,
        patch_size=fit.patch_size,
        multiple=fit.multiple,
        min_height=fit.min_height,
        batch_size=fit.batch_size,
        drop_last=fit.drop_last,
        seed=fit.seed,
        repeats_per_image=config.data.repeats_per_image,
        styles_per_repeat=config.data.styles_per_repeat,
        style_as_tensor=config.mode.offline_test,
        tile_size=config.data.tile_size,
        tile_stride=config.data.tile_stride,
        vae_f=config.model.vae.f,
        caption_path=config.data.caption_path,
        dictionary_path=config.data.dictionary_path,
        use_caption=config.model.caption.enabled,
        style_image_height=config.data.style_image_height,
        style_max_width=config.data.style_max_width,
        style_random_crop=config.data.style_random_crop,
    )


def _build_fit_loader(
    config: Config,
    train_dataset: HandwrittenFormulaDataset,
    rank: int = 0,
    world_size: int = 1,
) -> tuple:
    sampler = FitDistributedSampler(
        dataset=train_dataset,
        batch_size=config.data.fit.batch_size,
        num_replicas=world_size,
        rank=rank,
        seed=config.data.fit.seed,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=None,  # sampler yields complete batches
        sampler=sampler,
        num_workers=config.data.num_workers,
        pin_memory=True,
        collate_fn=bucket_collate,
    )
    return loader, sampler


def _ddp_worker(local_rank: int, config_path: str, resume: str, world_size: int, port: int, exp_dir: str):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        backend="nccl", init_method="env://",
        rank=local_rank, world_size=world_size,
    )
    torch.cuda.set_device(local_rank)

    config = load_config(config_path)
    config.mode.device = config.distributed.gpus[local_rank]
    config.training.log_dir = exp_dir
    set_config(config)
    set_seed(config.mode.seed + local_rank)

    print(f"[rank {local_rank}] building pipeline on {config.mode.device} ...")
    pipeline = build_pipeline(config)

    print(f"[rank {local_rank}] loading data from {config.data.data_root} ...")
    train_dataset = _build_train_dataset(config)
    train_loader, train_sampler = _build_fit_loader(
        config, train_dataset, rank=local_rank, world_size=world_size,
    )

    if local_rank == 0:
        print(f"Dataset size: {len(train_dataset)}")
        print(f"Pipeline params: {sum(p.numel() for p in pipeline.parameters() if p.requires_grad):,}")

    trainer = Trainer(config, pipeline, rank=local_rank, world_size=world_size)

    if resume:
        if local_rank == 0:
            print(f"Resuming from {resume} ...")
        trainer.resume(resume, train_loader, train_sampler)
    else:
        trainer.train(train_loader, train_sampler)

    dist.destroy_process_group()


def cmd_train(args):
    config = load_config(args.config)

    if config.distributed.enabled:
        gpus = config.distributed.gpus
        world_size = len(gpus)
        if world_size < 2:
            print("distributed.enabled=true but gpus has fewer than 2 devices; falling back to single-GPU.")
            config.distributed.enabled = False
        else:
            if world_size > torch.cuda.device_count():
                raise RuntimeError(
                    f"Requested {world_size} GPUs but only {torch.cuda.device_count()} available: {gpus}"
                )
            port = _find_free_port()
            exp_dir = _make_exp_dir(config.training.log_dir, args.config)
            print(f"Starting DDP training on {world_size} GPUs {gpus} (master port {port}) ...")
            print(f"TensorBoard log dir: {exp_dir}")
            set_config(config)
            mp.spawn(
                _ddp_worker,
                args=(args.config, args.resume, world_size, port, exp_dir),
                nprocs=world_size,
                join=True,
            )
            return

    config.training.log_dir = _make_exp_dir(config.training.log_dir, args.config)
    print(f"TensorBoard log dir: {config.training.log_dir}")
    set_config(config)
    set_seed(config.mode.seed)

    print(f"Building pipeline... (offline_test={config.mode.offline_test})")
    pipeline = build_pipeline(config)

    print(f"Loading data from {config.data.data_root} ...")
    train_dataset = _build_train_dataset(config)
    train_loader, train_sampler = _build_fit_loader(config, train_dataset)

    print(f"Dataset size: {len(train_dataset)}")
    print(f"Pipeline params: {sum(p.numel() for p in pipeline.parameters() if p.requires_grad):,}")

    trainer = Trainer(config, pipeline)

    if args.resume:
        print(f"Resuming from {resume} ...")
        trainer.resume(args.resume, train_loader, train_sampler)
    else:
        trainer.train(train_loader, train_sampler)


def cmd_infer(args):
    config = load_config(args.config)
    set_config(config)
    set_seed(config.mode.seed)

    print(f"Building pipeline... (offline_test={config.mode.offline_test})")
    pipeline = build_pipeline(config)

    inferencer = Inferencer(config, pipeline)

    if args.checkpoint:
        inferencer.load_checkpoint(args.checkpoint)

    img = inferencer.generate(
        print_image=args.print_path,
        style_image=args.style_path,
        output_path=args.output,
        caption=args.caption,
    )
    print(f"Generated image size: {img.size}")


def cmd_eval(args):
    config = load_config(args.config)
    set_config(config)
    set_seed(config.mode.seed)
    print("Evaluation mode — to be implemented with specific dataloaders.")
    print(f"Real dir: {args.real_dir}, Generated dir: {args.generated_dir}")


def main():
    parser = argparse.ArgumentParser(description="DiTtolatex: Style-Conditioned DiT for Handwritten Formula Generation")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config YAML")
    train_parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config YAML")
    infer_parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    infer_parser.add_argument("--print_path", type=str, required=True, help="Print formula image path")
    infer_parser.add_argument("--style_path", type=str, required=True, help="Style reference image path")
    infer_parser.add_argument("--output", type=str, default="output.png", help="Output image path")
    infer_parser.add_argument("--caption", type=str, default=None,
                              help="Optional latex formula string (space-split tokens)")
    infer_parser.add_argument("--restore_size", action="store_true",
                              help="Restore output to original print image size")

    eval_parser = subparsers.add_parser("eval", help="Evaluate model")
    eval_parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config YAML")
    eval_parser.add_argument("--real_dir", type=str, required=True, help="Directory of real images")
    eval_parser.add_argument("--generated_dir", type=str, required=True, help="Directory of generated images")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "infer":
        cmd_infer(args)
    elif args.command == "eval":
        cmd_eval(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


