"""配置加载器：从 YAML 解析为结构化 dataclass，提供单例访问。"""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, get_type_hints


# ── 子配置 ──────────────────────────────────────────────────────────

@dataclass
class DiTConfig:
    type: str = "DiT-B/2"
    hidden_dim: int = 768
    num_heads: int = 12
    depth: int = 12
    mlp_ratio: float = 4.0
    patch_size: int = 2
    in_channels: int = 8
    out_channels: int = 8
    latent_dim: int = 4
    use_grid_cond: bool = False


@dataclass
class VAEConfig:
    pretrained_path: str = "stabilityai/sd-vae-ft-ema"
    latent_dim: int = 4
    f: int = 8


@dataclass
class StyleEncoderConfig:
    model_name: str = "openai/clip-vit-base-patch32"
    feature_dim: int = 768
    num_style_images: int = 3


@dataclass
class CaptionConfig:
    enabled: bool = True
    num_layers: int = 2
    max_len: int = 256


@dataclass
class ModelConfig:
    dit: DiTConfig = field(default_factory=DiTConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    style_encoder: StyleEncoderConfig = field(default_factory=StyleEncoderConfig)
    caption: CaptionConfig = field(default_factory=CaptionConfig)


@dataclass
class FitConfig:
    max_patch: int = 192
    patch_size: int = 16
    latent_patch_size: int = 2
    multiple: int = 8
    min_height: int = 64
    batch_size: int = 4
    drop_last: bool = True
    seed: int = 42


@dataclass
class TrainingConfig:
    total_steps: int = 450000
    batch_size: int = 4
    lr: float = 2.0e-5
    weight_decay: float = 1.0e-4
    warmup_steps: int = 2000
    min_lr: float = 1.0e-6
    ema_decay: float = 0.9999
    cfg_dropout_rate: float = 0.1
    perceptual_loss_weight: float = 0.0
    bucket_steps: int = 500
    val_every: int = 1000
    log_every: int = 50
    save_every: int = 10000
    save_max: int = 5
    nan_abort_steps: int = 50
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    gradient_clip: float = 1.0
    mixed_precision: str = "bf16"


@dataclass
class DiffusionConfig:
    num_timesteps: int = 1000
    beta_schedule: str = "cosine"
    ddim_steps: int = 60
    ddim_eta: float = 0.0
    cfg_scale: float = 7.0


@dataclass
class DataConfig:
    data_root: str = "./data"
    image_size: int = 256
    fit: FitConfig = field(default_factory=FitConfig)
    all_size: list = field(default_factory=list)
    train_size: list = field(default_factory=list)
    style_cache_dir: str = "./style_cache"
    tile_size: int = 224
    tile_stride: int = 168
    caption_path: str = "data/caption.txt"
    dictionary_path: str = "data/dictionary.txt"
    repeats_per_image: int = 3
    styles_per_repeat: int = 3
    num_workers: int = 4


@dataclass
class ModeConfig:
    offline_test: bool = False
    seed: int = 42
    device: str = "cuda"


@dataclass
class DistributedConfig:
    enabled: bool = False
    gpus: list = field(default_factory=lambda: ["cuda:0", "cuda:1"])
    backend: str = "nccl"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    mode: ModeConfig = field(default_factory=ModeConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)


# ── 加载函数 ────────────────────────────────────────────────────────

def _dict_to_dataclass(cls, d: dict):
    """递归地将字典转为 dataclass 实例。"""
    if d is None:
        return cls()
    field_types = get_type_hints(cls)
    kwargs = {}
    for key, val in d.items():
        if key in field_types and hasattr(field_types[key], "__dataclass_fields__"):
            kwargs[key] = _dict_to_dataclass(field_types[key], val)
        else:
            kwargs[key] = val
    return cls(**kwargs)


def load_config(path: str) -> Config:
    """从 YAML 文件加载配置。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _dict_to_dataclass(Config, raw)


_cfg: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置单例（需要先调用 load_config）。"""
    global _cfg
    if _cfg is None:
        raise RuntimeError("Config not loaded. Call load_config() first.")
    return _cfg


def set_config(cfg: Config) -> None:
    """设置全局配置单例。"""
    global _cfg
    _cfg = cfg
