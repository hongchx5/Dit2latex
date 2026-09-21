"""
训练数据集（静态分桶）：按用户定义的采样策略，从 print/style 目录加载三元组 (I_p, I_s, I_t)，
并附 latex caption（公式）的 token id 序列。

静态分桶：
  - all_size（buckets 参数）是全部桶集合，用于按宽高比对图片分类（pick_bucket）；
  - train_size 是单个训练尺寸（必须属于 all_size），只加载分类到该尺寸的图片；
  - 其余尺寸的图片被过滤，不参与训练。

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

from data.buckets import DEFAULT_BUCKETS, pick_bucket
from data.transforms import bucket_transform_pil
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
        buckets: Sequence[Tuple[int, int]] = DEFAULT_BUCKETS,
        train_size: Optional[Tuple[int, int]] = None,
        repeats_per_image: int = 3,
        styles_per_repeat: int = 3,
        style_as_tensor: bool = False,
        tile_size: int = 224,
        tile_stride: int = 168,
        vae_f: int = 8,
        caption_path: Optional[str] = None,
        dictionary_path: Optional[str] = None,
        use_caption: bool = True,
    ):
        """
        Args:
            data_root: 数据根目录，结构为 data_root/print/ 和 data_root/style*/
            buckets: 全部桶集合 (H, W)（all_size），用于图片分类。
            train_size: 训练尺寸（单个，必须属于 buckets）；None 表示不过滤（保留全部）。
            repeats_per_image: 每张 print 图重复几轮风格采样。
            styles_per_repeat: 每轮从同风格中选几张 I_s。
            style_as_tensor: True 时风格图以原始尺寸 tensor 返回（offline 测试模式）。
            tile_size: 风格图平铺切分边长。
            tile_stride: 风格图平铺切分步长。
            vae_f: VAE 下采样倍率（用于桶校验）。
            caption_path: caption.txt 路径（可选）。
            dictionary_path: dictionary.txt 路径（caption 词表）。
            use_caption: 是否加载 caption（False 时 caption_ids 恒空）。
        """
        super().__init__()
        self.data_root = Path(data_root)
        self.buckets = list(buckets)          # all_size，分类用
        self.train_size = tuple(train_size) if train_size is not None else None
        self.use_caption = use_caption
        self.repeats_per_image = repeats_per_image
        self.styles_per_repeat = styles_per_repeat
        self.style_as_tensor = style_as_tensor
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.vae_f = vae_f

        # 桶尺寸校验
        for bh, bw in self.buckets:
            assert bh % vae_f == 0 and bw % vae_f == 0, \
                f"bucket ({bh}, {bw}) not divisible by vae_f={vae_f}"
            assert (bh // vae_f) % 2 == 0 and (bw // vae_f) % 2 == 0, \
                f"bucket ({bh}, {bw}) latent size not divisible by 2 (patch_size)"
        if self.train_size is not None:
            assert tuple(self.train_size) in [tuple(b) for b in self.buckets], \
                f"train_size {self.train_size} not in buckets {self.buckets}"

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

        # ── 按 all_size 分类 + train_size 过滤 ──
        self.print_buckets: Dict[str, Tuple[int, int]] = {}
        self.print_images: List[str] = []
        for name in all_print_images:
            with Image.open(self.print_dir / name) as im:
                w, h = im.size
            b = pick_bucket(w, h, self.buckets)
            if self.train_size is None or b == self.train_size:
                self.print_buckets[name] = b
                self.print_images.append(name)

        if not self.print_images:
            raise RuntimeError(
                f"No print images classified to train_size={self.train_size} "
                f"(total {len(all_print_images)} prints, buckets={self.buckets})"
            )

        # ── caption 词表与公式映射 ──
        self.token2id: Dict[str, int] = {}
        self.caption_dict: Dict[str, List[int]] = {}
        if use_caption and dictionary_path is not None and os.path.exists(dictionary_path):
            self.token2id, _ = build_vocab(dictionary_path)
        if use_caption and caption_path is not None and os.path.exists(caption_path):
            self._load_captions(caption_path)

        # 样本索引 → 桶 的映射（基于过滤后的 print_images，静态单桶）
        self.samples_per_image = repeats_per_image * styles_per_repeat
        self._length = len(self.print_images) * self.samples_per_image

        self.bucket_of_sample: List[Tuple[int, int]] = [
            self.print_buckets[self.print_images[idx // self.samples_per_image]]
            for idx in range(self._length)
        ]

        # 桶 → 样本索引列表（供 sampler 使用；静态单桶时只有一个桶）
        self.bucket_indices: Dict[Tuple[int, int], List[int]] = {}
        for idx, b in enumerate(self.bucket_of_sample):
            self.bucket_indices.setdefault(b, []).append(idx)

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
        """返回属于指定桶的全部样本索引。"""
        return self.bucket_indices.get(bucket, [])

    def buckets_with_data(self) -> List[Tuple[int, int]]:
        """返回有数据的桶（静态分桶时只有 train_size 一个）。"""
        return sorted(self.bucket_indices, key=lambda b: -len(self.bucket_indices[b]))

    def _load_content_image(self, path: Path, bucket: Tuple[int, int]) -> torch.Tensor:
        """加载内容图（I_p / I_t）并按桶变换。"""
        with Image.open(path) as img:
            tensor, _ = bucket_transform_pil(img, bucket[0], bucket[1])
        return tensor

    def _load_style_image(self, path: Path) -> Union[str, torch.Tensor]:
        """加载风格图：默认返回路径（tiling 在 style_encoder 内做），offline 模式返回 tensor。"""
        if self.style_as_tensor:
            with Image.open(path) as img:
                img = img.convert("RGB")
            import torchvision.transforms as T
            tensor = T.ToTensor()(img) * 2.0 - 1.0  # [-1, 1]，保持原始尺寸
            return tensor
        return str(path)

    def __getitem__(self, idx: int):
        """
        Returns:
            I_p:        (3, bucket_h, bucket_w) in [-1, 1]
            I_s:        风格图路径 str，或 (3, H, W) tensor（offline 模式）
            I_t:        (3, bucket_h, bucket_w) in [-1, 1]
            bucket:     (bucket_h, bucket_w) 元组
            caption_ids: list[int]，latex 公式 token id 序列（use_caption=False 时恒空）
        """
        print_idx = idx // self.samples_per_image
        group_idx = idx % self.samples_per_image

        img_name = self.print_images[print_idx]
        bucket = self.print_buckets[img_name]
        round_idx = group_idx // self.styles_per_repeat
        ref_idx = group_idx % self.styles_per_repeat

        # 确定性选择风格
        rng_style = random.Random(_seed_from_key(f"{img_name}:round:{round_idx}"))
        style_dir = rng_style.choice(self.style_dirs)

        # 加载 I_p
        I_p = self._load_content_image(self.print_dir / img_name, bucket)

        # 加载 I_t（同名手写图）
        tgt_path = style_dir / img_name
        if tgt_path.exists():
            I_t = self._load_content_image(tgt_path, bucket)
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

        return I_p, I_s, I_t, bucket, caption_ids


def bucket_collate(batch):
    """
    自定义 collate：I_p / I_t 同桶堆叠，I_s 保持路径列表，bucket 信息堆叠，
    caption 序列 padding 到 batch 内最大长度。

    Returns:
        (I_p, I_s, I_t, buckets, caption_ids (B,L) long, caption_mask (B,L) bool True=padding)
    """
    I_p = torch.stack([b[0] for b in batch], dim=0)
    I_t = torch.stack([b[2] for b in batch], dim=0)
    buckets = torch.tensor([list(b[3]) for b in batch], dtype=torch.long)

    I_s = [b[1] for b in batch]

    # caption padding
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

    return I_p, I_s, I_t, buckets, padded, mask
