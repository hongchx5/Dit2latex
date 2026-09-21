"""
推理器：加载模型，对单张图像执行风格条件生成（动态分桶 + 平铺切分）。

流程：
1. 内容图（print）按宽高比选桶 → 等比缩放 + 白色居中填充；
2. 风格图走平铺切分（tiling）→ CLIP → 4-query 聚合（命中缓存则直接用）；
3. DiT 在桶尺寸的 latent 上做 DDIM 采样；
4. 解码后按填充信息裁剪，还原原始宽高比。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union, Optional, List

import torch
import torch.nn as nn
from PIL import Image

from config.config_loader import Config
from models.pipeline import DiTtolatexPipeline
from models.caption_encoder import build_vocab, encode_caption_string
from diffusion.ddim import ddim_sample
from data.transforms import bucket_transform_pil, tensor_to_pil


class Inferencer:
    def __init__(self, config: Config, pipeline: DiTtolatexPipeline):
        self.config = config
        self.pipeline = pipeline
        self.device = config.mode.device
        self.vae_f = config.model.vae.f
        # 静态分桶：输出尺寸固定为 train_size（与训练一致）
        self.train_size = tuple(config.data.train_size)
        self.token2id = {}
        if config.data.dictionary_path and os.path.exists(config.data.dictionary_path):
            self.token2id, _ = build_vocab(config.data.dictionary_path)
        self.pipeline.eval()

    def load_checkpoint(self, path: str):
        """加载 DiT/可训练参数的权重。"""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        # 仅加载可训练参数（跳过冻结模块如 VAE、CLIP）
        state = ckpt.get("model_state_dict", ckpt)
        self.pipeline.load_state_dict(state, strict=False)
        print(f"Loaded weights from {path}")

    def _load_content(self, print_image: Union[str, Image.Image, torch.Tensor]):
        """
        内容图预处理：等比缩放到 train_size + 白色居中填充（静态分桶，固定输出尺寸）。

        Returns:
            (I_p (1,3,th,tw) tensor, (train_h, train_w), None)
        """
        train_h, train_w = self.train_size
        if isinstance(print_image, torch.Tensor):
            # 调用方已按 train_size 变换：直接使用
            tensor = print_image.to(self.device)
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            _, _, bh, bw = tensor.shape
            return tensor, (bh, bw), None

        if isinstance(print_image, str):
            img = Image.open(print_image)
        else:
            img = print_image
        tensor, _ = bucket_transform_pil(img, train_h, train_w)
        return tensor.to(self.device).unsqueeze(0), (train_h, train_w), None

    def _make_caption(self, caption: Optional[str]):
        """
        将 latex 公式字符串转为 caption 序列特征 + padding mask。

        caption 关闭（caption_encoder 为 None）时返回 (None, None)；
        caption 为空时返回零序列 + 全 False mask（等价无条件分支，避免全 mask 的 softmax NaN）。

        Returns:
            (caption_seq (1, L, D) 或 None, caption_mask (1, L) bool 或 None)
        """
        if self.pipeline.caption_encoder is None:
            return None, None
        D = self.config.model.style_encoder.feature_dim
        if caption is None or caption.strip() == "":
            return (torch.zeros(1, 1, D, device=self.device),
                    torch.zeros(1, 1, dtype=torch.bool, device=self.device))
        ids = encode_caption_string(caption.strip(), self.token2id)
        ids_t = torch.tensor([ids], dtype=torch.long, device=self.device)
        mask_t = torch.zeros(1, len(ids), dtype=torch.bool, device=self.device)
        seq = self.pipeline.encode_caption(ids_t, mask_t)
        return seq, mask_t

    @torch.no_grad()
    def generate(
        self,
        print_image: Union[str, Image.Image, torch.Tensor],
        style_image: Union[str, Image.Image],
        output_path: Optional[str] = None,
        caption: Optional[str] = None,
    ) -> Image.Image:
        """
        风格条件生成单张图像（输出固定 train_size 尺寸，与训练一致）。

        Args:
            print_image: 打印体公式图像（str 路径 / PIL / 已变换 tensor）。
            style_image: 手写风格参考图像（str 路径 / PIL）。
            output_path: 可选，保存路径。
            caption:     可选，latex 公式字符串（空格分隔 token），缺省则无 caption 条件。

        Returns:
            PIL Image（train_h × train_w，等比缩放 + 白填充，不做裁剪）
        """
        # 1. 内容图：选桶 + 缩放填充
        I_p, bucket, info = self._load_content(print_image)   # (1,3,bh,bw)
        bucket_h, bucket_w = bucket
        print(f"[Diag] print -> bucket {bucket_h}x{bucket_w}")

        # 2. 风格图：tiling + CLIP + 4-query 聚合（缓存优先）
        f_s_seq, f_s_pooled = self.pipeline.encode_style(style_image)

        # 3. caption 条件
        caption_seq, caption_mask = self._make_caption(caption)

        # 4. 初始噪声（桶尺寸的 latent）
        latent_h = bucket_h // self.vae_f
        latent_w = bucket_w // self.vae_f
        z_T = torch.randn(
            1, self.config.model.vae.latent_dim, latent_h, latent_w,
            device=self.device,
        )

        # 5. 内容条件 latent
        z_p = self.pipeline.encode_content(I_p)  # (1, 4, latent_h, latent_w)

        # 6. DDIM 采样
        z_0 = ddim_sample(
            pipeline=self.pipeline,
            noise_schedule=self.pipeline.noise_schedule,
            z_t=z_T,
            z_p=z_p,
            f_s_pooled=f_s_pooled,
            f_s_seq=f_s_seq,
            caption_seq=caption_seq,
            caption_mask=caption_mask,
            num_steps=self.config.diffusion.ddim_steps,
            eta=self.config.diffusion.ddim_eta,
            cfg_scale=self.config.diffusion.cfg_scale,
        )

        # 7. 解码
        I_g = self.pipeline.decode_latent(z_0)  # (1, 3, bh, bw)
        print(f"[Diag] z_0 range: [{z_0.min().item():.4f}, {z_0.max().item():.4f}]")
        print(f"[Diag] I_g  range: [{I_g.min().item():.4f}, {I_g.max().item():.4f}]")

        I_g = I_g.squeeze(0).cpu()
        pil_img = tensor_to_pil(I_g)  # (train_h, train_w)，与训练一致的固定输出尺寸

        if output_path is not None:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            pil_img.save(output_path)
            print(f"Saved to {output_path}")

        return pil_img
