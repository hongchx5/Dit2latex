"""风格编码器训练用数据集（阶段 A DINO、阶段 B 对比学习）。

目录结构约定（与主项目一致）：
    data_root/print/         打印体公式图
    data_root/style*/        各 writer 手写公式图（同名文件 = 同一公式）
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from data.transforms import preprocess_style_image, batch_style_images


def _safe_listdir(dir_path: Path, extensions=(".png", ".jpg", ".jpeg", ".bmp")) -> List[str]:
    if not dir_path.exists():
        return []
    return sorted(
        f.name for f in dir_path.iterdir()
        if f.suffix.lower() in extensions
    )


class StyleImageDataset(Dataset):
    """
    手写风格图数据集：每个样本是一张风格图 + writer 标签。

    适用于阶段 A（忽略标签）和阶段 B（使用 writer 标签）。
    """

    def __init__(
        self,
        data_roots: Union[str, List[str]],
        image_height: int = 64,
        max_width: int = 512,
        random_crop: bool = True,
        require_writer: bool = True,
    ):
        super().__init__()
        if isinstance(data_roots, str):
            data_roots = [data_roots]
        self.data_roots = [Path(r) for r in data_roots]
        self.image_height = image_height
        self.max_width = max_width
        self.random_crop = random_crop
        self.require_writer = require_writer

        self.samples: List[Tuple[Path, int]] = []
        self.writer_names: List[str] = []

        for root in self.data_roots:
            for entry in sorted(root.iterdir()):
                if not entry.is_dir():
                    continue
                # 有标签：style* 目录
                if entry.name.lower().startswith("style"):
                    writer_name = entry.name
                    if writer_name not in self.writer_names:
                        self.writer_names.append(writer_name)
                    wid = self.writer_names.index(writer_name)
                    for name in _safe_listdir(entry):
                        self.samples.append((entry / name, wid))
                elif not require_writer and entry.name != "print":
                    # 无标签目录（用于阶段 A）
                    for name in _safe_listdir(entry):
                        self.samples.append((entry / name, -1))

        if len(self.samples) == 0:
            raise RuntimeError(f"No style images found in {self.data_roots}")

        print(f"[StyleImageDataset] {len(self.samples)} images, {len(self.writer_names)} writers from {self.data_roots}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path, writer_id = self.samples[idx]
        with Image.open(path) as img:
            tensor, _ = preprocess_style_image(
                img,
                image_height=self.image_height,
                max_width=self.max_width,
                random_crop=self.random_crop,
            )
        return tensor, writer_id, path.name

    def collate_fn(self, batch: List[Tuple[torch.Tensor, int, str]]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        tensors, writer_ids, names = zip(*batch)
        images, valid_mask = batch_style_images(list(tensors), fill_value=1.0)
        return images, torch.tensor(writer_ids, dtype=torch.long), list(names)


def build_writer_index(data_roots: Union[str, List[str]]) -> Tuple[StyleImageDataset, Dict[int, List[int]], Dict[str, List[int]]]:
    """返回 (dataset, writer_id->indices, formula_stem->indices)。"""
    from collections import defaultdict
    ds = StyleImageDataset(
        data_roots,
        image_height=64,
        max_width=512,
        random_crop=False,
        require_writer=True,
    )
    writer_to_indices: Dict[int, List[int]] = defaultdict(list)
    content_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, (path, writer_id, _) in enumerate(ds.samples):
        writer_to_indices[writer_id].append(idx)
        content_to_indices[Path(path).stem].append(idx)
    return ds, dict(writer_to_indices), dict(content_to_indices)


class ContrastiveStyleDataset(Dataset):
    """
    阶段 B 对比学习数据集。

    每个 anchor 必须存在：
        - 同一 writer 的另一张图（positive）
        - 不同 writer 写的同名公式图（hard negative）
    """

    def __init__(
        self,
        data_roots: Union[str, List[str]],
        image_height: int = 64,
        max_width: int = 512,
    ):
        super().__init__()
        self.base, self.writer_to_indices, self.content_to_indices = build_writer_index(data_roots)
        self.image_height = image_height
        self.max_width = max_width

        valid_indices = []
        for idx in range(len(self.base)):
            path, writer_id, _ = self.base.samples[idx]
            stem = Path(path).stem
            same_content = self.content_to_indices.get(stem, [])
            hard_negs = [i for i in same_content if self.base.samples[i][1] != writer_id]
            if len(hard_negs) > 0:
                valid_indices.append(idx)

        if len(valid_indices) == 0:
            raise RuntimeError("No hard negatives found: each formula must appear in multiple style dirs.")
        self.valid_indices = valid_indices
        print(f"[ContrastiveStyleDataset] {len(valid_indices)}/{len(self.base)} anchors with hard negatives")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def _load(self, idx: int) -> torch.Tensor:
        path, _, _ = self.base.samples[idx]
        with Image.open(path) as img:
            tensor, _ = preprocess_style_image(
                img,
                image_height=self.image_height,
                max_width=self.max_width,
                random_crop=True,
            )
        return tensor

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int]]:
        anchor_idx = self.valid_indices[idx]
        path, writer_id, _ = self.base.samples[anchor_idx]
        stem = Path(path).stem

        same_writer = [i for i in self.writer_to_indices[writer_id] if i != anchor_idx]
        pos_idx = random.choice(same_writer) if same_writer else anchor_idx
        pos_writer_id = self.base.samples[pos_idx][1]

        same_content = self.content_to_indices.get(stem, [])
        hard_neg_candidates = [i for i in same_content if self.base.samples[i][1] != writer_id]
        hard_neg_idx = random.choice(hard_neg_candidates)
        hard_writer_id = self.base.samples[hard_neg_idx][1]

        return {
            "anchor": self._load(anchor_idx),
            "positive": self._load(pos_idx),
            "negative_hard": self._load(hard_neg_idx),
            "anchor_writer_id": writer_id,
            "positive_writer_id": pos_writer_id,
            "hard_writer_id": hard_writer_id,
        }

    def collate_fn(self, batch: List[Dict[str, Union[torch.Tensor, int]]]) -> Dict[str, torch.Tensor]:
        anchors = [b["anchor"] for b in batch]
        positives = [b["positive"] for b in batch]
        hard_negs = [b["negative_hard"] for b in batch]
        anchor_wids = torch.tensor([b["anchor_writer_id"] for b in batch], dtype=torch.long)
        pos_wids = torch.tensor([b["positive_writer_id"] for b in batch], dtype=torch.long)
        hard_wids = torch.tensor([b["hard_writer_id"] for b in batch], dtype=torch.long)

        anchor_im, anchor_mask = batch_style_images(anchors, fill_value=1.0)
        pos_im, pos_mask = batch_style_images(positives, fill_value=1.0)
        hard_im, hard_mask = batch_style_images(hard_negs, fill_value=1.0)

        return {
            "anchor": (anchor_im, anchor_mask),
            "positive": (pos_im, pos_mask),
            "negative_hard": (hard_im, hard_mask),
            "anchor_writer_id": anchor_wids,
            "positive_writer_id": pos_wids,
            "hard_writer_id": hard_wids,
        }


class DINOCropTransform:
    """DINO 风格的多裁剪：一个全局裁剪 + 多个水平片段局部裁剪，保持宽高比不压扁。"""

    def __init__(
        self,
        image_height: int = 64,
        max_width: int = 512,
        global_crop_min_ratio: float = 0.6,
        local_crops_number: int = 4,
        local_crop_min_ratio: float = 0.2,
        local_crop_max_ratio: float = 0.4,
    ):
        self.image_height = image_height
        self.max_width = max_width
        self.global_min = global_crop_min_ratio
        self.local_n = local_crops_number
        self.local_min = local_crop_min_ratio
        self.local_max = local_crop_max_ratio

    def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img.convert("RGB"), dtype=np.float32)  # (H, W, 3)
        tensor = torch.from_numpy(arr).permute(2, 0, 1) / 127.5 - 1.0  # (3, H, W)
        return tensor

    def _random_horizontal_crop(
        self,
        img: Image.Image,
        min_ratio: float,
        max_ratio: float,
    ) -> torch.Tensor:
        w, h = img.size
        ratio = random.uniform(min_ratio, max_ratio)
        crop_w = max(self.image_height, int(round(w * ratio)))  # 至少一个像素高宽比
        crop_w = min(crop_w, w, self.max_width)
        if crop_w >= w:
            start = 0
        else:
            start = random.randint(0, w - crop_w)
        cropped = img.crop((start, 0, start + crop_w, h))

        scale = self.image_height / h
        new_w = max(1, int(round(crop_w * scale)))
        resized = cropped.resize((new_w, self.image_height), Image.BICUBIC)
        return self._pil_to_tensor(resized)

    def __call__(self, img: Image.Image) -> List[torch.Tensor]:
        global_crop = self._random_horizontal_crop(img, self.global_min, 1.0)
        crops = [global_crop]
        for _ in range(self.local_n):
            crops.append(self._random_horizontal_crop(img, self.local_min, self.local_max))
        return crops


class DINODataset(Dataset):
    """阶段 A DINO 预训练数据集：每个样本返回 1 个全局裁剪 + N 个局部裁剪。"""

    def __init__(
        self,
        data_roots: Union[str, List[str]],
        image_height: int = 64,
        max_width: int = 512,
        global_crop_min_ratio: float = 0.6,
        local_crops_number: int = 4,
        local_crop_min_ratio: float = 0.2,
        local_crop_max_ratio: float = 0.4,
        include_unlabeled: bool = True,
    ):
        super().__init__()
        self.base = StyleImageDataset(
            data_roots,
            image_height=image_height,
            max_width=max_width,
            random_crop=False,
            require_writer=not include_unlabeled,
        )
        self.transform = DINOCropTransform(
            image_height=image_height,
            max_width=max_width,
            global_crop_min_ratio=global_crop_min_ratio,
            local_crops_number=local_crops_number,
            local_crop_min_ratio=local_crop_min_ratio,
            local_crop_max_ratio=local_crop_max_ratio,
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> List[torch.Tensor]:
        path, _, _ = self.base.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            crops = self.transform(img)
        return crops

    def collate_fn(self, batch: List[List[torch.Tensor]]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """返回 list[(B,3,H,W), (B,W)]，每个元素对应一个 crop 的全局 batch。"""
        num_crops = len(batch[0])
        result = []
        for c in range(num_crops):
            tensors = [batch[i][c] for i in range(len(batch))]
            images, masks = batch_style_images(tensors, fill_value=1.0)
            result.append((images, masks))
        return result
