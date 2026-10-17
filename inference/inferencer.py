"""推理器：FiT 可变分辨率风格条件生成。"""

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
from data.transforms import preprocess_style_image, tensor_to_pil


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
        """风格条件生成单张图像。"""
        I_p, original_size, fitted_size = self._load_content(print_image)
        fh, fw = fitted_size
        print(f"[Diag] print {original_size[0]}x{original_size[1]} -> {fh}x{fw}")

        # 新版风格编码：整图输入 + valid_mask
        if isinstance(style_image, str):
            style_pil = Image.open(style_image)
        else:
            style_pil = style_image
        style_tensor, _ = preprocess_style_image(
            style_pil.convert("RGB"),
            image_height=self.config.data.style_image_height,
            max_width=self.config.data.style_max_width,
            random_crop=False,
        )
        style_tensor = style_tensor.unsqueeze(0).to(self.device)
        style_mask = torch.ones(1, style_tensor.shape[-1], dtype=torch.bool, device=self.device)
        f_s_seq, f_s_pooled = self.pipeline.encode_style((style_tensor, style_mask))

        caption_seq, caption_mask = self._make_caption(caption)

        latent_h = fh // self.vae_f
        latent_w = fw // self.vae_f
        z_T = torch.randn(
            1, self.config.model.vae.latent_dim, latent_h, latent_w,
            device=self.device,
        )

        z_p = self.pipeline.encode_content(I_p)

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

        I_g = self.pipeline.decode_latent(z_0)
        I_g = I_g.squeeze(0).cpu()
        pil_img = tensor_to_pil(I_g)

        if restore_size:
            orig_h, orig_w = original_size
            pil_img = pil_img.resize((orig_w, orig_h), Image.BICUBIC)

        if output_path is not None:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            pil_img.save(output_path)
            print(f"Saved to {output_path}")

        return pil_img
