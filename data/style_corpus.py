"""
风格图语料索引：把风格图组织成 (writer, formula) → path 的二维表。

三个训练/评估脚本共用：
  - scripts/pretrain_style_dino.py  阶段 A（只需要全部图片路径）
  - scripts/train_style_encoder.py  阶段 B（需要「同 writer 不同公式」正样本 +
                                          「同公式不同 writer」难负样本）
  - scripts/eval_style_encoder.py   验收（需要按 writer 划分 seen / unseen）

数据结构事实（本项目自有数据）：
  - data_root/style0、style1 ... 每个目录 = 一种手写风格（= writer 标签，天然可得）
  - 每条公式在每种风格目录下都有同名文件 ⇒ **同一公式被所有 writer 各写了一遍**
    ⇒ 可以严格构造 writers × formulas 的矩形网格：
        同一行（同 writer，不同公式）  → 正样本
        同一列（同公式，不同 writer）  → **难负样本**（内容完全相同、只有笔迹不同）
        其余                            → 普通负样本

外部手写数据（IAM / CVL）没有「同名公式」这一结构，网格采样会自动退化：
按「公式被尽可能多的 writer 覆盖」来拼，缺失的格子留空（SupCon 仍然成立，
只是难负样本变少）。脚本会打印退化提示。
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def _is_style_dir(name: str) -> bool:
    return name.lower().startswith("style")


class WriterGroup:
    """一个「来源」下的全部 writer（自有数据 = 1 个 group；外部数据每个 root 一个 group）。"""

    def __init__(self, name: str):
        self.name = name
        self.writer_formulas: Dict[str, Dict[str, str]] = {}   # writer -> {formula: path}
        self._formulas: Optional[List[str]] = None

    @property
    def writers(self) -> List[str]:
        return sorted(self.writer_formulas)

    @property
    def n_images(self) -> int:
        return sum(len(v) for v in self.writer_formulas.values())

    def add(self, writer: str, formula: str, path: str) -> None:
        self.writer_formulas.setdefault(writer, {})[formula] = path
        self._formulas = None

    def formulas_of(self, writer: str) -> List[str]:
        return sorted(self.writer_formulas.get(writer, {}))

    @property
    def formulas(self) -> List[str]:
        """全部公式名（去重、排序）。"""
        if self._formulas is None:
            s = set()
            for m in self.writer_formulas.values():
                s.update(m.keys())
            self._formulas = sorted(s)
        return self._formulas


class StyleCorpus:
    """
    (writer, formula) → path 的索引，支持多个来源 root。

    Args:
        roots: 数据根目录列表。每个 root 下找 style*/ 子目录（每个 = 一个 writer）；
               没有 style* 时把每个一级子目录当作一个 writer（外部数据集布局）。
        exclude_writers: 需要排除的 writer 名（留一法评估的 hold-out）。
        verbose: 打印扫描结果。
    """

    def __init__(
        self,
        roots: Sequence[str],
        exclude_writers: Optional[Sequence[str]] = None,
        verbose: bool = True,
    ):
        self.roots = list(roots)
        self.exclude = set(exclude_writers or [])
        self.groups: List[WriterGroup] = []

        for root in self.roots:
            self.groups.append(self._scan_root(root, verbose))

        self.groups = [g for g in self.groups if g.writer_formulas]
        if not self.groups:
            raise RuntimeError(f"没有扫到任何风格图: {self.roots}")

        # 全局 writer 名 → 全局 id（跨 group 唯一）
        self.writer_names: List[str] = []
        for g in self.groups:
            for w in g.writers:
                self.writer_names.append(f"{g.name}::{w}" if len(self.groups) > 1 else w)
        self.writer_to_id: Dict[str, int] = {w: i for i, w in enumerate(self.writer_names)}
        self.num_writers = len(self.writer_names)

        if verbose:
            total = sum(g.n_images for g in self.groups)
            print(f"[corpus] {len(self.groups)} 个来源, {self.num_writers} 个 writer, "
                  f"{total} 张风格图"
                  + (f"（已排除 {len(self.exclude)} 个 writer）" if self.exclude else ""))
            for g in self.groups:
                print(f"[corpus]   {g.name}: {len(g.writer_formulas)} writers, "
                      f"{g.n_images} imgs, {len(g.formulas)} formulas")

    # ── 扫描 ────────────────────────────────────────────────────────

    def _scan_root(self, root: str, verbose: bool) -> WriterGroup:
        root_p = Path(root)
        if not root_p.is_dir():
            raise FileNotFoundError(f"目录不存在: {root}")

        subdirs = sorted(d for d in root_p.iterdir() if d.is_dir())
        style_dirs = [d for d in subdirs if _is_style_dir(d.name)]
        writer_dirs = style_dirs if style_dirs else (subdirs if subdirs else [root_p])
        if verbose and not style_dirs and subdirs:
            print(f"[corpus][warn] {root} 下没有 style* 目录，"
                  f"把每个一级子目录当作一个 writer（{len(subdirs)} 个）")

        group = WriterGroup(str(root))
        for d in writer_dirs:
            writer = d.name
            if writer in self.exclude or f"{root}::{writer}" in self.exclude:
                continue
            files = sorted(p for p in d.iterdir()
                           if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
            for p in files:
                group.add(writer, p.stem, str(p))
        return group

    # ── 查询 ────────────────────────────────────────────────────────

    def all_images(self) -> List[Tuple[str, str, str]]:
        """[(writer_name, formula, path), ...]（全量）。"""
        out: List[Tuple[str, str, str]] = []
        multi = len(self.groups) > 1
        for g in self.groups:
            for w in g.writers:
                gname = f"{g.name}::{w}" if multi else w
                for f, p in sorted(g.writer_formulas[w].items()):
                    out.append((gname, f, p))
        return out

    def images_of_writer(self, writer_global_name: str, limit: int = 0) -> List[Tuple[str, str]]:
        """某个 writer 的 [(formula, path), ...]；limit>0 时最多取 limit 条。"""
        multi = len(self.groups) > 1
        for g in self.groups:
            for w in g.writers:
                gname = f"{g.name}::{w}" if multi else w
                if gname != writer_global_name:
                    continue
                items = sorted(g.writer_formulas[w].items())
                if limit and limit > 0 and len(items) > limit:
                    step = len(items) / float(limit)
                    items = [items[int(i * step)] for i in range(limit)]
                return items
        return []

    # ── 网格采样（阶段 B 的核心）────────────────────────────────────

    def sample_grid(
        self,
        num_writers: int,
        num_formulas: int,
        rng: random.Random,
        max_tries: int = 30,
    ) -> Tuple[List[str], List[int], List[str], bool]:
        """
        采样一个 writer × formula 的矩形网格。

        网格的语义（**阶段 B 的关键设计**）：
          - 同一行 = 同一 writer 写的不同公式  → 正样本（只共享笔迹）
          - 同一列 = 不同 writer 写的同一公式  → **难负样本**（内容完全相同，只差笔迹）
          - 其余                                → 普通负样本

        Args:
            num_writers:  P，网格行数
            num_formulas: M，网格列数（P*M = 一个 batch 的图片数）
            rng: 随机数发生器（由调用方按 index 播种，保证可复现）
            max_tries: 严格网格采样失败多少次后退化

        Returns:
            (paths, writer_ids, formulas, is_strict)
            is_strict=False 表示退化成了「按公式聚合」的松散网格（外部数据常见）。
        """
        # 按图片数加权挑一个 group
        weights = [max(1, g.n_images) for g in self.groups]
        g = rng.choices(self.groups, weights=weights, k=1)[0]
        prefix = f"{g.name}::" if len(self.groups) > 1 else ""

        writers = g.writers
        P = min(num_writers, len(writers))

        # ── 严格网格：P 个 writer 的公式集合取交集 ──
        for _ in range(max_tries):
            sel_writers = rng.sample(writers, P) if P < len(writers) else list(writers)
            common = None
            for w in sel_writers:
                fs = set(g.writer_formulas[w].keys())
                common = fs if common is None else (common & fs)
            common = common or set()
            M = min(num_formulas, len(common))
            if M >= 1 and P >= 2:
                sel_formulas = rng.sample(sorted(common), M)
                paths, wids, formulas = self._emit(g, prefix, sel_writers, sel_formulas)
                return paths, wids, formulas, True

        # ── 退化：按公式聚合（外部数据没有「同名公式」结构）──
        # 挑覆盖 writer 数最多的公式，每个公式取尽可能多的 writer。
        cover: Dict[str, List[str]] = defaultdict(list)
        for w in writers:
            for f in g.writer_formulas[w]:
                cover[f].append(w)
        ranked = sorted(cover.items(), key=lambda kv: -len(kv[1]))
        sel_formulas = [f for f, _ in ranked[:max(1, num_formulas)]]

        paths: List[str] = []
        wids: List[int] = []
        formulas: List[str] = []
        for f in sel_formulas:
            ws = cover[f]
            take = rng.sample(ws, min(P, len(ws))) if P < len(ws) else list(ws)
            for w in take:
                paths.append(g.writer_formulas[w][f])
                wids.append(self.writer_to_id[prefix + w])
                formulas.append(f)
        return paths, wids, formulas, False

    def _emit(
        self,
        g: WriterGroup,
        prefix: str,
        sel_writers: List[str],
        sel_formulas: List[str],
    ) -> Tuple[List[str], List[int], List[str]]:
        paths: List[str] = []
        wids: List[int] = []
        formulas: List[str] = []
        # 行主序：writer 在外层（方便同 writer 的样本在内存里连续）
        for w in sel_writers:
            for f in sel_formulas:
                paths.append(g.writer_formulas[w][f])
                wids.append(self.writer_to_id[prefix + w])
                formulas.append(f)
        return paths, wids, formulas

    # ── hold-out 划分（留一法评估）───────────────────────────────────

    @staticmethod
    def split_holdout(
        writer_names: Sequence[str],
        n_holdout: int,
        seed: int = 42,
    ) -> Tuple[List[str], List[str]]:
        """
        确定性划分 hold-out writer（排序后按种子随机取），保证 train / eval 脚本一致。

        Returns:
            (holdout_writers, seen_writers)
        """
        names = sorted(writer_names)
        n = min(n_holdout, max(0, len(names) - 1))
        rng = random.Random(seed)
        holdout = set(rng.sample(names, n)) if n > 0 else set()
        return sorted(holdout), [w for w in names if w not in holdout]
