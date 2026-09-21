"""
双卡 DDP 冒烟验证：复用生产路径（main._build_bucket_loaders / Trainer / pipeline），
在 2 张 GPU 上真实跑几步分桶训练，验证：

1. 多进程 init + DDP 包装 + 桶循环训练跑通；
2. 各进程数据分片（DistributedSampler）与桶切换同步（确定性序列）；
3. 梯度同步：训练后两个进程的模型参数应完全一致；
4. rank 0 保存 checkpoint。

用法：python scripts/smoke_test_ddp.py
"""

from __future__ import annotations

import os
import shutil
import socket
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from PIL import Image, ImageDraw

import main as M
from config.config_loader import Config
from models.vae import OfflineVAE
from models.style_encoder import OfflineStyleEncoder
from models.dit.dit import DiT
from models.pipeline import DiTtolatexPipeline
from diffusion.noise_schedule import NoiseSchedule
from data.dataset import HandwrittenFormulaDataset
from train.trainer import Trainer


def make_image(path, w, h):
    img = Image.new("RGB", (w, h), (255, 255, 255))
    ImageDraw.Draw(img).text((w // 10, h // 3), "abc", fill=(0, 0, 0))
    img.save(path)


def build_fake_data(root):
    os.makedirs(os.path.join(root, "print"), exist_ok=True)
    # 10 张不同宽高比的图，保证每桶都有样本且够 2 进程分片
    sizes = [(256, 64), (512, 128), (384, 128), (256, 256), (512, 96),
             (300, 64), (400, 128), (256, 128), (512, 256), (300, 300)]
    for i, (w, h) in enumerate(sizes):
        make_image(os.path.join(root, "print", f"p{i}.png"), w, h)
    for s in ["style1", "style2"]:
        os.makedirs(os.path.join(root, s), exist_ok=True)
        for i, (w, h) in enumerate([(224, 224), (600, 224)]):
            make_image(os.path.join(root, s, f"s{i}.png"), w, h)


class FakePerceptual(torch.nn.Module):
    def forward(self, x, y):
        return F.mse_loss(x, y)


def make_offline_config(data_root: str, ckpt_dir: str, log_dir: str) -> Config:
    """构造离线小模型配置（不读 yaml，直接构造 dataclass）。"""
    cfg = Config()
    cfg.mode.offline_test = True
    cfg.mode.device = "cuda:0"          # 会被 worker 覆盖
    cfg.mode.seed = 42
    cfg.data.data_root = data_root
    cfg.data.num_workers = 0            # spawn 下避免 DataLoader worker 问题
    cfg.data.buckets = [[64, 512], [128, 512], [256, 256]]  # 3 个桶
    cfg.training.total_steps = 6
    cfg.training.batch_size = 2
    cfg.training.bucket_steps = 2       # 触发多次换桶
    cfg.training.log_every = 2
    cfg.training.save_every = 3         # 触发 rank0 保存
    cfg.training.checkpoint_dir = ckpt_dir
    cfg.training.log_dir = log_dir
    cfg.training.mixed_precision = "no"
    cfg.training.perceptual_loss_weight = 0.0
    cfg.model.dit.hidden_dim = 192
    cfg.model.dit.num_heads = 3
    cfg.model.dit.depth = 2
    cfg.model.style_encoder.feature_dim = 192
    cfg.model.vae.f = 8
    cfg.model.vae.latent_dim = 4
    cfg.diffusion.num_timesteps = 100
    cfg.distributed.enabled = True
    cfg.distributed.gpus = ["cuda:0", "cuda:1"]
    return cfg


def _build_pipeline(cfg: Config):
    dev = cfg.mode.device
    vae = OfflineVAE(latent_dim=cfg.model.vae.latent_dim, f=cfg.model.vae.f).to(dev)
    style_enc = OfflineStyleEncoder(feature_dim=cfg.model.style_encoder.feature_dim).to(dev)
    dit = DiT(
        hidden_dim=cfg.model.dit.hidden_dim,
        num_heads=cfg.model.dit.num_heads,
        depth=cfg.model.dit.depth,
        patch_size=2, in_channels=8, out_channels=8,
        latent_dim=cfg.model.vae.latent_dim,
        context_dim=cfg.model.style_encoder.feature_dim,
    ).to(dev)
    ns = NoiseSchedule(num_timesteps=cfg.diffusion.num_timesteps,
                       beta_schedule="cosine", device=dev)
    pipe = DiTtolatexPipeline(
        vae=vae, style_encoder=style_enc, dit=dit, noise_schedule=ns,
        perceptual_loss=FakePerceptual(),
        perceptual_loss_weight=cfg.training.perceptual_loss_weight,
        cfg_dropout_rate=cfg.training.cfg_dropout_rate,
        device=dev,
    ).to(dev)
    return pipe


def worker(local_rank: int, tmp: str, port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="nccl", init_method="env://",
                            rank=local_rank, world_size=2)
    torch.cuda.set_device(local_rank)

    data_root = os.path.join(tmp, "data")
    ckpt_dir = os.path.join(tmp, "ckpt")
    log_dir = os.path.join(tmp, "logs")
    cfg = make_offline_config(data_root, ckpt_dir, log_dir)
    cfg.mode.device = cfg.distributed.gpus[local_rank]

    pipe = _build_pipeline(cfg)
    ds = HandwrittenFormulaDataset(
        data_root=data_root,
        buckets=[tuple(b) for b in cfg.data.buckets],
        repeats_per_image=2, styles_per_repeat=2,
        style_as_tensor=True, vae_f=cfg.model.vae.f,
    )
    loaders, samplers = M._build_bucket_loaders(cfg, ds, rank=local_rank, world_size=2)
    print(f"[rank {local_rank}] buckets with distributed sampler: "
          f"{[b for b, s in samplers.items() if s is not None]}", flush=True)

    trainer = Trainer(cfg, pipe, rank=local_rank, world_size=2)
    trainer.train(loaders, samplers)

    # 训练后 dump 参数（父进程比较两卡是否一致）
    torch.save({k: v.detach().clone() for k, v in pipe.state_dict().items()},
               os.path.join(tmp, f"params_rank{local_rank}.pt"))
    dist.destroy_process_group()
    print(f"[rank {local_rank}] worker done", flush=True)


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    if torch.cuda.device_count() < 2:
        print("SKIP: need >= 2 GPUs")
        return
    tmp = tempfile.mkdtemp(prefix="ddp_smoke_")
    try:
        build_fake_data(os.path.join(tmp, "data"))
        port = _find_free_port()
        print(f"spawning 2 workers on port {port} ...")
        mp.spawn(worker, args=(tmp, port), nprocs=2, join=True)

        # 校验：两卡参数一致
        p0 = torch.load(os.path.join(tmp, "params_rank0.pt"), map_location="cpu", weights_only=True)
        p1 = torch.load(os.path.join(tmp, "params_rank1.pt"), map_location="cpu", weights_only=True)
        assert set(p0.keys()) == set(p1.keys()), "param keys differ"
        diffs = [(k, (p0[k] - p1[k]).abs().max().item()) for k in p0 if p0[k].dtype.is_floating_point]
        bad = [d for d in diffs if d[1] > 1e-5]
        assert not bad, f"params diverge across ranks: {bad[:5]}"
        print(f"param sync OK: {len(diffs)} floating params all within 1e-5")

        # 校验：rank0 checkpoint 已保存
        ckpts = os.listdir(os.path.join(tmp, "ckpt"))
        assert any("checkpoint" in c for c in ckpts), f"checkpoint missing: {ckpts}"
        print("checkpoint saved by rank0 OK:", sorted(ckpts))
        print("DDP SMOKE TEST PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
