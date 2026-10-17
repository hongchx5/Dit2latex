"""训练数据集（FiT 动态分桶）：按 patch 预算动态缩放、按尺寸分组 batch。

每张 print 图先通过 fit_size_to_patch_budget 计算目标尺寸 (H', W')；
满足 min_height 的图按精确尺寸分桶；样本不足 batch_size 的桶合并到最接近的已有桶；
每个 epoch 重新生成 batch 列表并打乱。

返回三元组 (I_p, I_s, I_t) 和目标尺寸，同 batch 内尺寸严格一致。
"""

from __future__ import annotations

import os
import math
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset
from PIL import Image

from data.buckets import fit_size_to_patch_budget, fit_transform_pil
from data.transforms import preprocess_style_image, batch_style_images
from models.caption_encoder import build_vocab, encode_caption_string, PAD_ID


def _seed_from_key(key: str) -> int:
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**31)


def _stem(name: str) -> str:
    return Path(name).stem


def _bucket_distance(b1: Tuple[int, int], b2: Tuple[int, int]) -> float:
    h1, w1 = b1
    h2, w2 = b2
    return abs(math.log(h1 / max(h2, 1))) + abs(math.log(w1 / max(w2, 1)))


class HandwrittenFormulaDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        max_patch: int = 192,
        patch_size: int = 16,
        multiple: int = 8,
        min_height: int = 64,
        batch_size: int = 4,
        drop_last: bool = True,
        seed: int = 42,
        repeats_per_image: int = 3,
        styles_per_repeat: int = 3,
        style_as_tensor: bool = False,
        tile_size: int = 224,
        tile_stride: int = 168,
        vae_f: int = 8,
        caption_path: Optional[str] = None,
        dictionary_path: Optional[str] = None,
        use_caption: bool = True,
        # ---- 新版风格编码器参数 ----
        style_image_height: int = 64,
        style_max_width: int = 512,
        style_random_crop: bool = True,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.max_patch = max_patch
        self.patch_size = patch_size
        self.multiple = multiple
        self.min_height = min_height
        self.batch_size = max(1, batch_size)
        self.drop_last = drop_last
        self.seed = seed
        self.use_caption = use_caption
        self.repeats_per_image = repeats_per_image
        self.styles_per_repeat = styles_per_repeat
        self.style_as_tensor = style_as_tensor
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.vae_f = vae_f

        # 新版风格图参数
        self.style_image_height = style_image_height
        self.style_max_width = style_max_width
        self.style_random_crop = style_random_crop

        self.print_dir = self.data_root / "print"
        if not self.print_dir.exists():
            raise FileNotFoundError(f"print directory not found: {self.print_dir}")

        all_print_images = sorted(
            f.name for f in self.print_dir.iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
        if not all_print_images:
            raise RuntimeError(f"No print images found in {self.print_dir}")

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

        # FiT sizing and filtering
        self.print_buckets: Dict[str, Tuple[int, int]] = {}
        self.print_images: List[str] = []
        discarded = 0
        for name in all_print_images:
            with Image.open(self.print_dir / name) as im:
                w, h = im.size
            fitted = fit_size_to_patch_budget(h, w, max_patch, patch_size, multiple)
            if fitted is None:
                print(f"[FiT discard] {name}: cannot fit ({h}x{w})")
                discarded += 1
                continue
            fh, fw = fitted
            if fh < min_height:
                print(f"[FiT discard] {name}: {h}x{w} -> {fh}x{fw} (height < {min_height})")
                discarded += 1
                continue
            self.print_buckets[name] = (fh, fw)
            self.print_images.append(name)

        print(f"[FiT dataset] {len(self.print_images)}/{len(all_print_images)} prints kept, "
              f"{discarded} discarded (min_height={min_height})")

        if not self.print_images:
            raise RuntimeError("No print images left after FiT filtering")

        self.token2id: Dict[str, int] = {}
        self.caption_dict: Dict[str, List[int]] = {}
        if use_caption and dictionary_path is not None and os.path.exists(dictionary_path):
            self.token2id, _ = build_vocab(dictionary_path)
        if use_caption and caption_path is not None and os.path.exists(caption_path):
            self._load_captions(caption_path)

        self.samples_per_image = repeats_per_image * styles_per_repeat
        self._length = len(self.print_images) * self.samples_per_image

        self.bucket_of_sample: List[Tuple[int, int]] = [
            self.print_buckets[self.print_images[idx // self.samples_per_image]]
            for idx in range(self._length)
        ]

    def _load_captions(self, caption_path: str):
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

    def _build_merged_size_buckets(self) -> Dict[Tuple[int, int], List[int]]:
        """按精确尺寸分组后，将小桶合并到距离最近的大桶。"""
        size_to_indices: Dict[Tuple[int, int], List[int]] = {}
        for idx, bucket in enumerate(self.bucket_of_sample):
            size_to_indices.setdefault(bucket, []).append(idx)

        big_buckets: Dict[Tuple[int, int], List[int]] = {}
        small_buckets: List[Tuple[Tuple[int, int], List[int]]] = []
        for size, indices in size_to_indices.items():
            if len(indices) >= self.batch_size:
                big_buckets[size] = indices
            else:
                small_buckets.append((size, indices))

        final_buckets: Dict[Tuple[int, int], List[int]] = {k: list(v) for k, v in big_buckets.items()}
        for small_size, indices in small_buckets:
            if not final_buckets:
                break
            target = min(final_buckets.keys(), key=lambda b: _bucket_distance(small_size, b))
            final_buckets[target].extend(indices)

        if not final_buckets:
            final_buckets = {k: list(v) for k, v in size_to_indices.items()}

        return final_buckets

    def build_batches(self, epoch: int = 0) -> List[List[int]]:
        """生成当前 epoch 的所有 batch（用于自定义 Sampler 或验证）。"""
        final_buckets = self._build_merged_size_buckets()
        rng = random.Random(self.seed + epoch)
        batches: List[List[int]] = []
        effective_bs = self.batch_size
        if max(len(v) for v in final_buckets.values()) < self.batch_size:
            effective_bs = 1

        for indices in final_buckets.values():
            order = list(indices)
            rng.shuffle(order)
            for i in range(0, len(order), effective_bs):
                chunk = order[i:i + effective_bs]
                if len(chunk) < effective_bs and self.drop_last and effective_bs > 1:
                    continue
                batches.append(chunk)

        rng.shuffle(batches)
        return batches

    def buckets_with_data(self) -> List[Tuple[int, int]]:
        """返回所有有数据的尺寸桶。"""
        return sorted(set(self.bucket_of_sample))

    def indices_for_bucket(self, bucket: Tuple[int, int]) -> List[int]:
        """返回属于指定尺寸桶的全部样本索引。"""
        return [idx for idx, b in enumerate(self.bucket_of_sample) if b == bucket]

    def _load_content_image(self, path: Path, bucket: Tuple[int, int]) -> torch.Tensor:
        with Image.open(path) as img:
            tensor = fit_transform_pil(img, bucket[0], bucket[1])
        return tensor

    def _load_style_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as img:
            tensor, _ = preprocess_style_image(
                img,
                image_height=self.style_image_height,
                max_width=self.style_max_width,
                random_crop=self.style_random_crop,
            )
        return tensor

    def _get_single(self, idx: int):
        """获取单个样本。"""
        print_idx = idx // self.samples_per_image
        group_idx = idx % self.samples_per_image

        img_name = self.print_images[print_idx]
        bucket = self.print_buckets[img_name]
        round_idx = group_idx // self.styles_per_repeat
        ref_idx = group_idx % self.styles_per_repeat

        rng_style = random.Random(_seed_from_key(f"{img_name}:round:{round_idx}"))
        style_dir = rng_style.choice(self.style_dirs)

        I_p = self._load_content_image(self.print_dir / img_name, bucket)

        tgt_path = style_dir / img_name
        if tgt_path.exists():
            I_t = self._load_content_image(tgt_path, bucket)
        else:
            I_t = I_p.clone()

        rng_refs = random.Random(_seed_from_key(f"{img_name}:round:{round_idx}:refs"))
        candidates = [f for f in self.style_images[style_dir] if f != img_name]
        if not candidates:
            candidates = self.style_images[style_dir]
        rng_refs.shuffle(candidates)
        I_s = self._load_style_image(style_dir / candidates[ref_idx % len(candidates)])

        caption_ids = self.caption_dict.get(_stem(img_name), []) if self.use_caption else []

        return I_p, I_s, I_t, bucket, caption_ids

    def __getitem__(self, idx):
        """
        支持 DataLoader 两种方式：
        - batch_size=... + Sampler  yielding int -> idx 为 int
        - batch_size=None + BatchSampler yielding list[int] -> idx 为 list
        """
        if isinstance(idx, int):
            return self._get_single(idx)
        # idx is a list of sample indices (one batch)
        return [self._get_single(i) for i in idx]


def bucket_collate(batch):
    """同 batch 内 I_p/I_t 尺寸相同，I_s 需要按 W 方向 padding。"""
    I_p = torch.stack([b[0] for b in batch], dim=0)
    I_t = torch.stack([b[2] for b in batch], dim=0)
    buckets = torch.tensor([list(b[3]) for b in batch], dtype=torch.long)

    # 新版风格图：按高度 H 一致但宽度不同，右侧 padding + valid_mask
    style_tensors = [b[1] for b in batch]
    I_s, I_s_mask = batch_style_images(style_tensors, fill_value=1.0)

    caption_ids = [b[4] for b in batch]
    max_len = max((len(c) for c in caption_ids), default=0)
    max_len = max(max_len, 1)

    padded = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    mask = torch.ones((len(batch), max_len), dtype=torch.bool)
    for i, c in enumerate(caption_ids):
        L = len(c)
        if L > 0:
            padded[i, :L] = torch.tensor(c[:L], dtype=torch.long)
            mask[i, :L] = False
        else:
            mask[i, 0] = False

    return I_p, (I_s, I_s_mask), I_t, buckets, padded, mask


# ── 动态 batch Sampler ──────────────────────────────────────────────

class FitDistributedSampler:
    """
    FiT 动态 batch 采样器。

    每个 epoch 调用 dataset.build_batches(epoch) 生成 batch 列表，
    然后按 rank 对 batch 列表分片。
    """
    def __init__(
        self,
        dataset: HandwrittenFormulaDataset,
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.num_samples = 0
        self.total_size = 0

    def __iter__(self):
        batches = self.dataset.build_batches(self.epoch)
        if not batches:
            raise RuntimeError(
                f"FitDistributedSampler rank={self.rank}: no batches generated. "
                f"Check data.fit.batch_size / min_height / dataset size."
            )
        valid_batches = batches[self.rank::self.num_replicas]
        if not valid_batches:
            raise RuntimeError(
                f"FitDistributedSampler rank={self.rank}: no batch left after DDP split. "
                f"Try reducing data.fit.batch_size or increasing data."
            )
        self.num_samples = len(valid_batches)
        self.total_size = len(valid_batches)
        self.epoch += 1
        return iter(valid_batches)

    def __len__(self):
        return max(1, self.num_samples)

    def set_epoch(self, epoch: int):
        self.epoch = epoch
