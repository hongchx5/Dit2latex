"""
统计风格图的高度分布，给出 style_encoder.height（基准高度 H）与 max_width 的推荐值。

背景：ConvNeXt 风格编码器把风格图**等比缩放到固定高度 H**（保留宽高比，绝不压扁），
H 太小 → 笔迹细节（粗细、连笔）被平均掉；H 太大 → 显存/算力吃紧（宽度按宽高比同步放大）。
本脚本直接扫盘量出真实分布，避免拍脑袋定 H。

只读取图片头部（PIL 惰性解析，不解码像素），15 万张图也能在几分钟内跑完。

用法：
    python scripts/stat_style_height.py --data_root ./data
    python scripts/stat_style_height.py --data_root ./data --extra_roots /data/IAM /data/CVL
    python scripts/stat_style_height.py --data_root ./data --limit_per_dir 2000 --workers 32

输出示例：
    [heights]  n=150000  min=28  p5=48  p25=56  median=64  p75=72  p95=96  max=180
    [ratios]   W/H: p5=1.8  median=3.6  p95=9.4  max=21.0
    [推荐] H = 64
    [推荐] max_width = 512  （H=64 时宽度的 p99 = 604，按 8*H=512 截断会裁掉 0.7% 的图）
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from PIL import Image

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

# 候选基准高度：H 必须是 16 的倍数吗？不是硬性要求，但 backbone 总 stride = 16，
# H 取 16 的倍数时特征图高度 H/16 是整数，位置编码与 mask 对齐最干净。
CANDIDATE_HEIGHTS = (32, 48, 64, 80, 96, 128)


# ── 扫描 ────────────────────────────────────────────────────────────

def is_style_dir(name: str) -> bool:
    return name.lower().startswith("style")


def find_style_dirs(root: str) -> List[Path]:
    """
    在 root 下找风格目录。

    - 自有数据：root/style0、style1 ...（每个目录 = 一种手写风格）
    - 外部数据（IAM / CVL 之类）：没有 style* 前缀，则把 root 下每个一级子目录
      当作一个 writer 目录。
    - 都没有：把 root 自己当成一个 writer 目录。
    """
    root_p = Path(root)
    if not root_p.is_dir():
        raise SystemExit(f"[error] 目录不存在: {root}")

    subdirs = sorted(d for d in root_p.iterdir() if d.is_dir())
    style_dirs = [d for d in subdirs if is_style_dir(d.name)]
    if style_dirs:
        return style_dirs
    if subdirs:
        print(f"[warn] {root} 下没有 style* 目录，改为把每个一级子目录当作一个 writer 目录 "
              f"（共 {len(subdirs)} 个）")
        return subdirs
    return [root_p]


def list_images(d: Path, limit: int = 0) -> List[Path]:
    files = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if limit and limit > 0 and len(files) > limit:
        # 均匀抽样，保持目录内分布（不是只取前 N 个）
        step = len(files) / float(limit)
        files = [files[int(i * step)] for i in range(limit)]
    return files


def _read_size(p: Path) -> Optional[Tuple[int, int]]:
    try:
        with Image.open(p) as im:
            return im.size   # (W, H)；只解析头部，不解码像素
    except Exception:
        return None


def scan_sizes(dirs: Sequence[Path], files_per_dir: Dict[Path, List[Path]],
               workers: int) -> List[Tuple[int, int]]:
    all_files = [f for d in dirs for f in files_per_dir.get(d, [])]
    if not all_files:
        return []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        sizes = list(ex.map(_read_size, all_files))
    return [s for s in sizes if s is not None]


# ── 统计 ────────────────────────────────────────────────────────────

def percentile(sorted_vals: Sequence[float], q: float) -> float:
    """q ∈ [0, 100]。"""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo]) * (1 - frac) + float(sorted_vals[hi]) * frac


def describe(vals: Sequence[float], name: str) -> Dict[str, float]:
    s = sorted(vals)
    out = {
        "n": float(len(s)), "min": float(s[0]), "max": float(s[-1]),
        "mean": sum(s) / len(s),
        "p5": percentile(s, 5), "p25": percentile(s, 25), "median": percentile(s, 50),
        "p75": percentile(s, 75), "p95": percentile(s, 95), "p99": percentile(s, 99),
    }
    print(f"[{name:8s}] n={int(out['n'])}  min={out['min']:.0f}  p5={out['p5']:.0f}  "
          f"p25={out['p25']:.0f}  median={out['median']:.0f}  p75={out['p75']:.0f}  "
          f"p95={out['p95']:.0f}  p99={out['p99']:.0f}  max={out['max']:.0f}  "
          f"mean={out['mean']:.1f}")
    return out


def pick_height(median_h: float) -> int:
    """从候选高度里挑最接近中位数的那个（候选都是 16 的倍数）。"""
    return min(CANDIDATE_HEIGHTS, key=lambda h: abs(h - median_h))


def main():
    ap = argparse.ArgumentParser(
        description="统计风格图高度 / 宽高比分布，推荐 style_encoder.height 与 max_width",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data_root", type=str, default="./data",
                    help="数据根目录（下面有 style*/ 目录）")
    ap.add_argument("--extra_roots", type=str, nargs="*", default=[],
                    help="额外的手写数据根目录（IAM / CVL 等），用于看外部数据的分布")
    ap.add_argument("--limit_per_dir", type=int, default=0,
                    help="每个风格目录最多采样多少张（0 = 全部；扫全量大目录慢时可设 2000）")
    ap.add_argument("--workers", type=int, default=16, help="读图片头部的并发线程数")
    args = ap.parse_args()

    roots = [args.data_root] + list(args.extra_roots)
    all_sizes: List[Tuple[int, int]] = []
    per_root_sizes: Dict[str, List[Tuple[int, int]]] = {}

    for root in roots:
        dirs = find_style_dirs(root)
        files_per_dir = {d: list_images(d, args.limit_per_dir) for d in dirs}
        n_files = sum(len(v) for v in files_per_dir.values())
        print(f"\n[scan] {root}: {len(dirs)} 个 writer 目录, {n_files} 张图")
        if n_files == 0:
            print(f"[warn] {root} 下没找到图片（扩展名白名单 {IMAGE_EXTS}）")
            continue
        sizes = scan_sizes(dirs, files_per_dir, args.workers)
        if len(sizes) < n_files:
            print(f"[warn] {n_files - len(sizes)} 张图读取失败，已跳过")
        per_root_sizes[root] = sizes
        all_sizes.extend(sizes)

    if not all_sizes:
        raise SystemExit("[error] 没有读到任何图片，检查 --data_root")

    heights = [h for _, h in all_sizes]
    ratios = [w / max(1.0, float(h)) for w, h in all_sizes]

    print("\n=== 全部风格图 ===")
    h_stat = describe(heights, "height")
    r_stat = describe(ratios, "W/H")

    for root, sizes in per_root_sizes.items():
        if len(per_root_sizes) > 1 and sizes:
            print(f"\n--- {root} ---")
            describe([h for _, h in sizes], "height")
            describe([w / max(1.0, float(h)) for w, h in sizes], "W/H")

    # ── 推荐 ────────────────────────────────────────────────────────
    H = pick_height(h_stat["median"])
    print("\n=== 推荐 ===")
    print(f"[推荐] model.style_encoder.height = {H}   （高度中位数 {h_stat['median']:.0f}，"
          f"候选 {list(CANDIDATE_HEIGHTS)} 中最接近；取 16 的倍数让特征图高度 H/16 为整数）")
    print(f"[推荐] data.style_height = {H}   （两处必须一致）")

    # max_width：先按 8*H 的默认值算，再报告会被裁掉的比例
    default_max_w = 8 * H
    widths_at_H = [max(1, int(round(w * float(H) / float(h)))) for w, h in all_sizes]
    over = sum(1 for w in widths_at_H if w > default_max_w) / len(widths_at_H)
    w_sorted = sorted(widths_at_H)
    p99_w = percentile(w_sorted, 99)
    p999_w = percentile(w_sorted, 99.9)

    print(f"[推荐] model.style_encoder.max_width = {default_max_w}   （= 8 * H）")
    print(f"       该高度下宽度分布：median={percentile(w_sorted,50):.0f}  "
          f"p95={percentile(w_sorted,95):.0f}  p99={p99_w:.0f}  p99.9={p999_w:.0f}  "
          f"max={w_sorted[-1]}")
    print(f"       按 {default_max_w} 截断会裁掉 {over*100:.2f}% 的图"
          f"（训练随机裁 / 推理居中裁，会丢失被裁掉那段的笔迹）")
    if over > 0.02:
        suggested = int(min(p999_w, max(default_max_w, p99_w * 1.5)) // 16 * 16)
        print(f"       [提示] 超过 2%，可考虑把 max_width 提到 {suggested}"
              f"（代价：显存峰值上升，极端宽图的 batch 会变慢）")

    # 显存/算力粗估
    print(f"\n[参考] H={H} 时 ConvNeXt-T（总 stride 16）的特征图：")
    print(f"       高度 H' = {H // 16}；宽度 W' = W/16"
          f"（median {percentile(w_sorted,50)/16:.0f}，p95 {percentile(w_sorted,95)/16:.0f}）")
    print(f"       token 数 median ≈ {H//16 * percentile(w_sorted,50)/16:.0f}，"
          f"p95 ≈ {H//16 * percentile(w_sorted,95)/16:.0f}")


if __name__ == "__main__":
    main()
