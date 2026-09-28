"""
训练数据集（FiT 式可变分辨率 / token 预算分桶）：从 print/style 目录加载
三元组 (I_p, I_s, I_t)，并附 latex caption（公式）的 token id 序列。

与旧版「静态分桶 + train_size 过滤」的区别：
  - 不再把所有图压到同一个固定尺寸（如 128×384）。每张图按自身面积等比缩放，
    使 patch token 数 grid_h × grid_w ≤ max_tokens，宽高比被完整保留；
  - 再按宽高比自动分箱，每箱一个 canvas（同样满足 token 预算），batch 内同 canvas，
    因此 tensor 形状一致；样本自己的内容区之外的 token 用 token_mask 屏蔽；
  - min_grid_h 只用于统计「缩放后 grid_h 过小」的样本占比，不做任何过滤/约束。

caption（可选，use_caption=True 时加载）：
  - caption.txt：每行 "<文件名>\t<空格 split 的 latex token>"；
  - dictionary.txt：每行一个 latex token（词表）。

caption 定位：用 print 图文件名（去掉扩展名）作为 key 查 caption.txt。
"""

from __future__ import annotations

import os
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset
from PIL import Image

from data.buckets import (
    build_aspect_bins,
    fit_size_to_token_budget,
    size_to_grid,
    token_unit,
    ContentRect,
)
from data.transforms import fit_transform_pil, pad_style_batch, style_transform_pil
from models.caption_encoder import build_vocab, encode_caption_string, PAD_ID


def _seed_from_key(key: str) -> int:
    """将字符串 key 映射为确定性整数种子。"""
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**31)


def _stem(name: str) -> str:
    """去除文件扩展名（如 '23_em_52.png' -> '23_em_52'）。"""
    return Path(name).stem


class HandwrittenFormulaDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        max_tokens: int = 256,
        min_grid_h: int = 5,
        num_aspect_bins: int = 8,
        only_downscale: bool = True,
        repeats_per_image: int = 3,
        styles_per_repeat: int = 3,
        style_as_tensor: bool = True,
        style_height: int = 64,
        style_max_width: Optional[int] = None,
        style_crop: str = "random",
        tile_size: int = 224,
        tile_stride: int = 168,
        vae_f: int = 8,
        patch_size: int = 2,
        caption_path: Optional[str] = None,
        dictionary_path: Optional[str] = None,
        use_caption: bool = True,
        verbose: bool = True,
    ):
        """
        Args:
            data_root: 数据根目录，结构为 data_root/print/ 和 data_root/style*/
            max_tokens: patch token 预算（grid_h × grid_w 上界；1 token = vae_f×patch_size 像素）。
            min_grid_h: 仅用于统计：缩放后 grid_h < min_grid_h 的样本占比（不做约束）。
            num_aspect_bins: 宽高比自动分箱数量（决定有多少个 canvas / DataLoader）。
            only_downscale: True 时小图不放大。
            repeats_per_image: 每张 print 图重复几轮风格采样。
            styles_per_repeat: 每轮从同风格中选几张 I_s。
            style_as_tensor: True（默认）= 风格图已缩放到 style_height 的 tensor；
                             False = 只返回路径 str（clip_tiled 基线：tiling 在编码器内做）。
            style_height:   风格图基准高度 H（等比缩放，保留宽高比，绝不压扁）。
            style_max_width: 风格图宽度上限；超过时按 style_crop 裁一段（防 OOM）。
            style_crop:     "random"（训练）/ "center"（验证、推理）。
            tile_size / tile_stride: 风格图平铺切分参数（仅 clip_tiled 基线使用）。
            vae_f / patch_size: 用于 token 网格换算。
            caption_path / dictionary_path / use_caption: caption 相关。
            verbose: 是否打印分桶与统计信息。
        """
        super().__init__()
        self.data_root = Path(data_root)
        self.max_tokens = int(max_tokens)
        self.min_grid_h = int(min_grid_h)
        self.num_aspect_bins = int(num_aspect_bins)
        self.only_downscale = bool(only_downscale)
        self.vae_f = int(vae_f)
        self.patch_size = int(patch_size)
        self.unit = token_unit(self.vae_f, self.patch_size)
        self.use_caption = use_caption
        self.repeats_per_image = repeats_per_image
        self.styles_per_repeat = styles_per_repeat
        self.style_as_tensor = style_as_tensor
        self.style_height = int(style_height)
        self.style_max_width = int(style_max_width) if style_max_width else None
        self.style_crop = style_crop
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.verbose = verbose

        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        if self.num_aspect_bins < 1:
            raise ValueError(f"num_aspect_bins must be >= 1, got {num_aspect_bins}")

        self.print_dir = self.data_root / "print"
        if not self.print_dir.exists():
            raise FileNotFoundError(f"print directory not found: {self.print_dir}")

        all_print_images = sorted(
            f.name for f in self.print_dir.iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
        if not all_print_images:
            raise RuntimeError(f"No print images found in {self.print_dir}")

        # 扫描所有 style 目录
        self.style_dirs: List[Path] = []
        self.style_images: Dict[Path, List[str]] = {}
        for entry in sorted(self.data_root.iterdir()):
            if entry.is_dir() and entry.name.lower().startswith("style") and entry.name != "print":
                self.style_dirs.append(entry)
                self.style_images[entry] = sorted(
                    f.name for f in entry.iterdir()
                    if f.suffix.lower() in (".png", ".jpg", ".jpeg")
                )
        if not self.style_dirs:
            raise RuntimeError(f"No style directories found in {self.data_root}")

        # ── 1. 每张图按 token 预算拟合尺寸 ──────────────────────────
        self.orig_size: Dict[str, Tuple[int, int]] = {}
        self.print_fit: Dict[str, Tuple[int, int]] = {}
        self.print_grid: Dict[str, Tuple[int, int]] = {}
        items: List[Tuple[str, int, int]] = []
        for name in all_print_images:
            with Image.open(self.print_dir / name) as im:
                w, h = im.size
            self.orig_size[name] = (w, h)
            fit_h, fit_w = fit_size_to_token_budget(
                w, h, self.max_tokens, self.vae_f, self.patch_size, self.only_downscale
            )
            self.print_fit[name] = (fit_h, fit_w)
            self.print_grid[name] = size_to_grid(fit_h, fit_w, self.vae_f, self.patch_size)
            items.append((name, w, h))

        # ── 2. 按宽高比自动分箱 → 每箱一个 canvas ──────────────────
        self.bins = build_aspect_bins(
            items,
            max_tokens=self.max_tokens,
            num_bins=self.num_aspect_bins,
            vae_f=self.vae_f,
            patch_size=self.patch_size,
            only_downscale=self.only_downscale,
        )
        self.print_canvas: Dict[str, Tuple[int, int]] = {}
        for canvas_h, canvas_w, names in self.bins:
            for name in names:
                self.print_canvas[name] = (canvas_h, canvas_w)

        # 全部图片都参与训练（不再按 train_size 过滤）
        self.print_images: List[str] = [name for name, _, _ in items]

        # ── 3. 统计（min_grid_h 只统计、不约束）────────────────────
        self.samples_per_image = repeats_per_image * styles_per_repeat
        self._length = len(self.print_images) * self.samples_per_image

        self.bucket_of_sample: List[Tuple[int, int]] = [
            self.print_canvas[self.print_images[idx // self.samples_per_image]]
            for idx in range(self._length)
        ]

        # canvas → 样本索引列表（供 sampler 使用）
        self.bucket_indices: Dict[Tuple[int, int], List[int]] = {}
        for idx, b in enumerate(self.bucket_of_sample):
            self.bucket_indices.setdefault(b, []).append(idx)

        # ── caption 词表与公式映射 ─────────────────────────────────
        self.token2id: Dict[str, int] = {}
        self.caption_dict: Dict[str, List[int]] = {}
        if use_caption and dictionary_path is not None and os.path.exists(dictionary_path):
            self.token2id, _ = build_vocab(dictionary_path)
        if use_caption and caption_path is not None and os.path.exists(caption_path):
            self._load_captions(caption_path)

        if self.verbose:
            self._report()

    # ── 统计报告 ──────────────────────────────────────────────────

    def _report(self):
        """打印 token 预算拟合结果：分箱情况 + grid_h 过小样本占比。"""
        total = len(self.print_images)
        if total == 0:
            return

        below = sum(1 for n in self.print_images if self.print_grid[n][0] < self.min_grid_h)
        ratios = [
            float(self.orig_size[n][0]) / max(1.0, float(self.orig_size[n][1]))
            for n in self.print_images
        ]
        rmin, rmax = min(ratios), max(ratios)
        tokens = [gh * gw for gh, gw in (self.print_grid[n] for n in self.print_images)]

        print(
            f"[dataset][fit] images={total}, max_tokens={self.max_tokens}, "
            f"unit={self.unit}px, bins={len(self.bins)}"
        )
        print(
            f"[dataset][fit] aspect ratio: {rmin:.2f} ~ {rmax:.2f}; "
            f"tokens: {min(tokens)} ~ {max(tokens)} (avg {sum(tokens)/total:.1f})"
        )
        # min_grid_h 只做统计，不做任何过滤/约束
        print(
            f"[dataset][fit] grid_h < min_grid_h({self.min_grid_h}): "
            f"{below}/{total} = {below / total * 100:.2f}%  (仅统计，无约束效果)"
        )
        for canvas_h, canvas_w, names in self.bins:
            gh, gw = size_to_grid(canvas_h, canvas_w, self.vae_f, self.patch_size)
            print(f"[dataset][fit]   canvas {canvas_h}x{canvas_w} "
                  f"({gh}x{gw} = {gh*gw} tokens): {len(names)} imgs")

    def _load_captions(self, caption_path: str):
        """加载 caption.txt → {文件名(无扩展名): token id 列表}。"""
        matched = 0
        with open(caption_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n").strip()
                if not line:
                    continue
                if "\t" in line:
                    name, formula = line.split("\t", 1)
                else:
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    name, formula = parts[0], " ".join(parts[1:])
                self.caption_dict[name] = encode_caption_string(formula, self.token2id)
                matched += 1

        missing = sum(1 for n in self.print_images if _stem(n) not in self.caption_dict)
        if missing:
            print(f"[dataset] caption loaded {matched}; missing caption for {missing}/{len(self.print_images)} prints")

    def __len__(self) -> int:
        return self._length

    def indices_for_bucket(self, bucket: Tuple[int, int]) -> List[int]:
        """返回属于指定 canvas 桶的全部样本索引。"""
        return self.bucket_indices.get(bucket, [])

    def buckets_with_data(self) -> List[Tuple[int, int]]:
        """返回有数据的 canvas 桶（按样本数降序）。"""
        return sorted(self.bucket_indices, key=lambda b: -len(self.bucket_indices[b]))

    def _load_content_image(
        self,
        path: Path,
        canvas: Tuple[int, int],
        fit: Tuple[int, int],
    ) -> Tuple[torch.Tensor, ContentRect]:
        """
        加载内容图（I_p / I_t）：缩放到 fit 尺寸 + 居中贴进 canvas。

        Returns:
            (tensor (3, canvas_h, canvas_w) in [-1, 1], rect)
        """
        with Image.open(path) as img:
            tensor, _, rect = fit_transform_pil(
                img, fit[0], fit[1], canvas[0], canvas[1],
                vae_f=self.vae_f, patch_size=self.patch_size,
            )
        return tensor, rect

    def _load_style_image(self, path: Path) -> Union[str, torch.Tensor]:
        """
        加载风格图。

        - style_as_tensor=True（默认，ConvNeXt 风格编码器）：
          打开 → RGB → 等比缩放到高度 style_height → ToTensor → [-1, 1]，
          返回 (3, H, W_i) tensor。缩放在 worker 进程内完成（主进程只做 padding）。
        - style_as_tensor=False（clip_tiled 基线）：返回路径 str，tiling 在编码器内做。
        """
        if self.style_as_tensor:
            with Image.open(path) as img:
                return style_transform_pil(
                    img,
                    height=self.style_height,
                    max_width=self.style_max_width,
                    crop=self.style_crop,
                )
        return str(path)

    def __getitem__(self, idx: int):
        """
        Returns:
            I_p:        (3, canvas_h, canvas_w) in [-1, 1]
            I_s:        风格图 tensor (3, style_height, W_i) in [-1,1]，或路径 str（clip_tiled 基线）
            I_t:        (3, canvas_h, canvas_w) in [-1, 1]
            bucket:     (canvas_h, canvas_w) 元组
            caption_ids: list[int]，latex 公式 token id 序列（use_caption=False 时恒空）
            rect:       (canvas_gh, canvas_gw, row0, col0, content_gh, content_gw)
        """
        print_idx = idx // self.samples_per_image
        group_idx = idx % self.samples_per_image

        img_name = self.print_images[print_idx]
        canvas = self.print_canvas[img_name]
        fit = self.print_fit[img_name]
        round_idx = group_idx // self.styles_per_repeat
        ref_idx = group_idx % self.styles_per_repeat

        # 确定性选择风格
        rng_style = random.Random(_seed_from_key(f"{img_name}:round:{round_idx}"))
        style_dir = rng_style.choice(self.style_dirs)

        # 加载 I_p
        I_p, rect = self._load_content_image(self.print_dir / img_name, canvas, fit)

        # 加载 I_t（同名手写图，与 I_p 共用 canvas / fit 保证对齐）
        tgt_path = style_dir / img_name
        if tgt_path.exists():
            I_t, _ = self._load_content_image(tgt_path, canvas, fit)
        else:
            I_t = I_p.clone()  # fallback

        # 加载 I_s
        rng_refs = random.Random(_seed_from_key(f"{img_name}:round:{round_idx}:refs"))
        candidates = [f for f in self.style_images[style_dir] if f != img_name]
        if not candidates:
            candidates = self.style_images[style_dir]
        rng_refs.shuffle(candidates)
        I_s = self._load_style_image(style_dir / candidates[ref_idx % len(candidates)])

        # caption
        caption_ids = self.caption_dict.get(_stem(img_name), []) if self.use_caption else []

        return I_p, I_s, I_t, canvas, caption_ids, rect


def bucket_collate(batch):
    """
    自定义 collate：I_p / I_t 同 canvas 堆叠，I_s 打包成 (tensor, mask)，bucket 信息堆叠，
    caption 序列 padding 到 batch 内最大长度，并按内容区生成 token padding mask。

    I_s 的两种形态（由 dataset.style_as_tensor 决定）：
      - True（默认，ConvNeXt 风格编码器）：二元组 (style_tensor, style_mask)
            style_tensor : (B, 3, H, W_max) float，[-1,1]，padding 用 **+1.0（背景色白）**
            style_mask   : (B, W_max) bool，True = 真实区域，False = padding
      - False（clip_tiled 基线）：保持 List[str] 路径列表

    两种形态下**元组个数都保持不变**（本分支仍为 7 元组）。

    Returns:
        I_p:          (B, 3, canvas_h, canvas_w)
        I_s:          见上（二元组 或 路径列表）
        I_t:          (B, 3, canvas_h, canvas_w)
        buckets:      (B, 2) long
        caption_ids:  (B, L) long
        caption_mask: (B, L) bool，True = padding
        token_mask:   (B, N) bool，True = padding（内容区之外的 patch token）
    """
    I_p = torch.stack([b[0] for b in batch], dim=0)
    I_t = torch.stack([b[2] for b in batch], dim=0)
    buckets = torch.tensor([list(b[3]) for b in batch], dtype=torch.long)

    # ── 风格图：tensor 模式 → batch 内 padding 到 W_max + valid_mask ──
    # padding 在主进程做（dataset 侧已完成缩放）；填充值必须是 +1.0（背景色）。
    I_s = [b[1] for b in batch]
    if torch.is_tensor(I_s[0]):
        I_s = pad_style_batch(I_s, pad_value=1.0, pad_side="right")

    # ── token padding mask（内容区之外的 patch token）─────────────
    rects: List[ContentRect] = [b[5] for b in batch]
    canvas_gh, canvas_gw = rects[0][0], rects[0][1]
    N = canvas_gh * canvas_gw
    token_mask = torch.ones((len(batch), N), dtype=torch.bool)  # True = padding
    for i, (cgh, cgw, row0, col0, gh, gw) in enumerate(rects):
        assert (cgh, cgw) == (canvas_gh, canvas_gw), \
            f"batch 内 canvas 网格不一致: {(cgh, cgw)} != {(canvas_gh, canvas_gw)}"
        grid = torch.ones((cgh, cgw), dtype=torch.bool)
        grid[row0: row0 + gh, col0: col0 + gw] = False
        token_mask[i] = grid.reshape(-1)      # 行主序，与 DiT 的 coords 顺序一致

    # ── caption padding ───────────────────────────────────────────
    caption_ids = [b[4] for b in batch]
    max_len = max((len(c) for c in caption_ids), default=0)
    max_len = max(max_len, 1)  # 至少 1 列

    padded = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    mask = torch.ones((len(batch), max_len), dtype=torch.bool)  # True = padding
    for i, c in enumerate(caption_ids):
        L = len(c)
        if L > 0:
            padded[i, :L] = torch.tensor(c[:L], dtype=torch.long)
            mask[i, :L] = False
        else:
            # P3：空 caption（该图不在 caption.txt 中）不能整行都是 padding。
            # 全 True mask → TransformerEncoder / CrossAttn 里 masked_fill(-inf)
            # 后 softmax 全 -inf → 确定性 NaN。这里留 1 个非 padding 位（PAD id，
            # 其 embedding 恒为 0），与推理端 inferencer._make_caption 的
            # "零序列 + 全 False mask" 处理方式一致。
            mask[i, 0] = False

    return I_p, I_s, I_t, buckets, padded, mask, token_mask
