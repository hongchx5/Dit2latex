"""
推理器：加载模型，对单张图像执行风格条件生成（可变分辨率 + 平铺切分）。

流程：
1. 内容图（print）按 token 预算等比缩放（保留原始宽高比，不做固定尺寸填充）；
2. 风格图走平铺切分（tiling）→ CLIP → 4-query 聚合（任意分辨率参考图都先按
   height=224 等比缩放再切块，因此分辨率不受限制）；
3. DiT 在该图自己的 token 网格上做 DDIM 采样（无 padding）；
4. 解码后按需还原到原始像素尺寸（--restore_size）。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union, Optional, List, Tuple

import torch
import torch.nn as nn
from PIL import Image

from config.config_loader import Config
from models.pipeline import DiTtolatexPipeline
from models.caption_encoder import build_vocab, encode_caption_string
from diffusion.ddim import ddim_sample
from data.buckets import fit_size_to_token_budget, token_unit
from data.transforms import fit_transform_pil, tensor_to_pil


class Inferencer:
    def __init__(self, config: Config, pipeline: DiTtolatexPipeline):
        self.config = config
        self.pipeline = pipeline
        self.device = config.mode.device
        self.vae_f = config.model.vae.f
        self.patch_size = config.model.dit.patch_size
        self.unit = token_unit(self.vae_f, self.patch_size)
        # 可变分辨率：每张图按 token 预算缩放，不再有固定的 train_size
        self.max_tokens = int(getattr(config.data, "max_tokens", 256))
        self.min_grid_h = int(getattr(config.data, "min_grid_h", 5))
        self.only_downscale = bool(getattr(config.data, "only_downscale", True))
        self.token2id = {}
        if config.data.dictionary_path and os.path.exists(config.data.dictionary_path):
            self.token2id, _ = build_vocab(config.data.dictionary_path)
        self.pipeline.eval()

    def load_checkpoint(self, path: str):
        """加载 DiT/可训练参数的权重。"""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        # 仅加载可训练参数（跳过冻结模块如 VAE、CLIP）
        state = ckpt.get("model_state_dict", ckpt)
        result = self.pipeline.load_state_dict(state, strict=False)
        print(f"Loaded weights from {path}")
        # 可变分辨率改过 cond_dim（grid 尺度条件）/ 新增模块时，旧 checkpoint 会有缺键，
        # 静默跳过只会让权重随机初始化，这里显式报出来。
        if result.missing_keys:
            print(f"  [warn] {len(result.missing_keys)} keys missing (随机初始化): "
                  f"{result.missing_keys[:5]}{' ...' if len(result.missing_keys) > 5 else ''}")
        if result.unexpected_keys:
            print(f"  [warn] {len(result.unexpected_keys)} keys unused: "
                  f"{result.unexpected_keys[:5]}{' ...' if len(result.unexpected_keys) > 5 else ''}")

    def fit_size(self, w: int, h: int) -> Tuple[int, int]:
        """按 token 预算算出这张图的生成尺寸 (fit_h, fit_w)（unit 的整数倍）。"""
        return fit_size_to_token_budget(
            w, h,
            max_tokens=self.max_tokens,
            vae_f=self.vae_f,
            patch_size=self.patch_size,
            only_downscale=self.only_downscale,
        )

    def _load_content(self, print_image: Union[str, Image.Image, torch.Tensor]):
        """
        内容图预处理：按 token 预算等比缩放（保留原始宽高比，无填充）。

        Returns:
            (I_p (1,3,fit_h,fit_w) tensor, (fit_h, fit_w), (orig_w, orig_h) 或 None)
        """
        if isinstance(print_image, torch.Tensor):
            # 调用方已按自己的尺寸变换好：直接使用（尺寸必须是 unit 的整数倍）
            tensor = print_image.to(self.device)
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            _, _, bh, bw = tensor.shape
            assert bh % self.unit == 0 and bw % self.unit == 0, \
                f"输入 tensor 尺寸 {bh}x{bw} 必须是 {self.unit} 的整数倍"
            return tensor, (bh, bw), None

        if isinstance(print_image, str):
            img = Image.open(print_image)
        else:
            img = print_image
        w, h = img.size
        fit_h, fit_w = self.fit_size(w, h)
        # 推理时 canvas = fit，没有 padding
        tensor, _, _ = fit_transform_pil(
            img, fit_h, fit_w,
            vae_f=self.vae_f, patch_size=self.patch_size,
        )
        return tensor.to(self.device).unsqueeze(0), (fit_h, fit_w), (w, h)

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
        restore_original_size: bool = False,
        rope_scale: Optional[tuple] = None,
    ) -> Image.Image:
        """
        风格条件生成单张图像（输出尺寸由该图的宽高比 + token 预算决定）。

        Args:
            print_image: 打印体公式图像（str 路径 / PIL / 已变换 tensor）。
            style_image: 手写风格参考图像（str 路径 / PIL），任意分辨率均可。
            output_path: 可选，保存路径。
            caption:     可选，latex 公式字符串（空格分隔 token），缺省则无 caption 条件。
            restore_original_size: True 时把生成结果放大/缩小回输入图的原始像素尺寸。
            rope_scale:  可选 (s_h, s_w)，推理 token 数超出训练预算时的 RoPE 外推缩放。

        Returns:
            PIL Image（fit_h × fit_w，或还原后的原始尺寸）
        """
        # 1. 内容图：按 token 预算缩放（保留宽高比）
        I_p, bucket, orig_size = self._load_content(print_image)   # (1,3,fh,fw)
        fit_h, fit_w = bucket
        grid_h, grid_w = fit_h // self.unit, fit_w // self.unit
        print(f"[Diag] print -> {fit_h}x{fit_w} ({grid_h}x{grid_w} = {grid_h*grid_w} tokens)")
        if grid_h < self.min_grid_h:
            print(f"[Diag][warn] grid_h={grid_h} < min_grid_h={self.min_grid_h}："
                  f"该图过于狭长，字符高度只剩 {grid_h*self.unit}px，生成质量可能下降")

        # 2. 风格图：tiling + CLIP + 4-query 聚合（缓存优先）
        #    tiling 内部会先把参考图等比缩放到 height=224 再切块，
        #    所以任意分辨率/宽高比的参考图都能直接喂进来。
        f_s_seq, f_s_pooled = self.pipeline.encode_style(style_image)

        # 3. caption 条件
        caption_seq, caption_mask = self._make_caption(caption)

        # 4. 初始噪声（该图自己的 latent 尺寸）
        latent_h = fit_h // self.vae_f
        latent_w = fit_w // self.vae_f
        z_T = torch.randn(
            1, self.config.model.vae.latent_dim, latent_h, latent_w,
            device=self.device,
        )

        # 5. 内容条件 latent
        z_p = self.pipeline.encode_content(I_p)  # (1, 4, latent_h, latent_w)

        # 6. DDIM 采样（推理无 padding，attn_mask=None）
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
            rope_scale=rope_scale,
        )

        # 7. 解码
        I_g = self.pipeline.decode_latent(z_0)  # (1, 3, fit_h, fit_w)
        print(f"[Diag] z_0 range: [{z_0.min().item():.4f}, {z_0.max().item():.4f}]")
        print(f"[Diag] I_g  range: [{I_g.min().item():.4f}, {I_g.max().item():.4f}]")

        I_g = I_g.squeeze(0).cpu()
        pil_img = tensor_to_pil(I_g)  # (fit_h, fit_w)

        # 8. 可选：还原到输入图的原始像素尺寸
        if restore_original_size and orig_size is not None:
            pil_img = pil_img.resize(orig_size, Image.BICUBIC)

        if output_path is not None:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            pil_img.save(output_path)
            print(f"Saved to {output_path}")

        return pil_img
