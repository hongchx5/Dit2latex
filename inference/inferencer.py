"""推理器：FiT 可变分辨率风格条件生成。

流程：
1. 内容图按 FiT patch 预算动态缩放；
2. 风格图走平铺切分 → CLIP → 4-query 聚合；
3. DiT 在动态 latent 上做 DDIM 采样；
4. 可选 --restore_size 还原到原始像素尺寸。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union, Optional

import torch
import torch.nn.functional as F
from PIL import Image

from config.config_loader import Config
from models.pipeline import DiTtolatexPipeline
from models.caption_encoder import build_vocab, encode_caption_string
from diffusion.ddim import ddim_sample
from data.buckets import fit_size_to_patch_budget, bucket_transform_pil, bucket_transform_tensor
from data.transforms import tensor_to_pil


class Inferencer:
    def __init__(self, config: Config, pipeline: DiTtolatexPipeline):
        self.config = config
        self.pipeline = pipeline
        self.device = config.mode.device
        self.vae_f = config.model.vae.f
        self.fit = config.data.fit
        self.token2id = {}
        if config.data.dictionary_path and os.path.exists(config.data.dictionary_path):
            self.token2id, _ = build_vocab(config.data.dictionary_path)
        self.pipeline.eval()

    def load_checkpoint(self, path: str):
        """加载 DiT/可训练参数的权重。"""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        self.pipeline.load_state_dict(state, strict=False)
        print(f"Loaded weights from {path}")

    def _load_content(self, print_image: Union[str, Image.Image, torch.Tensor]):
        """
        内容图预处理：按 FiT patch 预算等比缩放。

        Returns:
            (I_p (1,3,H,W), original_size (H,W), fitted_size (H,W))
        """
        if isinstance(print_image, torch.Tensor):
            tensor = print_image.to(self.device)
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            _, _, h, w = tensor.shape
            fitted = fit_size_to_patch_budget(
                h, w, self.fit.max_patch, self.fit.patch_size, self.fit.multiple
            )
            if fitted is None or fitted[0] < self.fit.min_height:
                raise ValueError(f"Input tensor {h}x{w} cannot fit FiT budget")
            fh, fw = fitted
            tensor, _ = bucket_transform_tensor(tensor.squeeze(0), fh, fw)
            return tensor.unsqueeze(0).to(self.device), (h, w), (fh, fw)

        if isinstance(print_image, str):
            img = Image.open(print_image)
        else:
            img = print_image
        w, h = img.size
        fitted = fit_size_to_patch_budget(
            h, w, self.fit.max_patch, self.fit.patch_size, self.fit.multiple
        )
        if fitted is None or fitted[0] < self.fit.min_height:
            raise ValueError(f"Input image {h}x{w} cannot fit FiT budget")
        fh, fw = fitted
        tensor, _ = bucket_transform_pil(img, fh, fw)
        return tensor.unsqueeze(0).to(self.device), (h, w), (fh, fw)

    def _make_caption(self, caption: Optional[str]):
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
        restore_size: bool = False,
    ) -> Image.Image:
        """
        风格条件生成单张图像。

        Args:
            print_image: 打印体公式图像。
            style_image: 手写风格参考图像。
            output_path: 可选保存路径。
            caption: 可选 latex 公式字符串。
            restore_size: 是否将输出还原到原始 print 图像尺寸。

        Returns:
            PIL Image。
        """
        # 1. 内容图缩放
        I_p, original_size, fitted_size = self._load_content(print_image)
        fh, fw = fitted_size
        print(f"[Diag] print {original_size[0]}x{original_size[1]} -> {fh}x{fw}")

        # 2. 风格编码
        f_s_seq, f_s_pooled = self.pipeline.encode_style(style_image)

        # 3. caption
        caption_seq, caption_mask = self._make_caption(caption)

        # 4. 初始噪声
        latent_h = fh // self.vae_f
        latent_w = fw // self.vae_f
        z_T = torch.randn(
            1, self.config.model.vae.latent_dim, latent_h, latent_w,
            device=self.device,
        )

        # 5. 内容条件 latent
        z_p = self.pipeline.encode_content(I_p)

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
        I_g = self.pipeline.decode_latent(z_0)
        I_g = I_g.squeeze(0).cpu()
        pil_img = tensor_to_pil(I_g)  # (fh, fw)

        # 8. 可选还原尺寸
        if restore_size:
            orig_h, orig_w = original_size
            pil_img = pil_img.resize((orig_w, orig_h), Image.BICUBIC)

        if output_path is not None:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            pil_img.save(output_path)
            print(f"Saved to {output_path}")

        return pil_img
