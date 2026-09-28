"""
批量推理：一组内容图 × 多组风格 → 分别输出到多个文件夹。

单张推理等价于（本脚本内部调用的就是同一个 Inferencer.generate）：
    python main.py infer --checkpoint <ckpt> --print_path <content_img> \
        --style_path <style_img> --output <out_dir>/<img_name> --caption "<formula>"

与逐张调用 `main.py infer` 的区别：**模型只加载一次**（VAE / CLIP / DiT / checkpoint），
然后在进程内循环生成，避免每张图都重新加载一遍权重（几百张图能省掉大量时间）。

输入：
  1. --checkpoint      模型权重路径（必填）
  2. --caption_file    caption 文件，每行 "<文件名><空格或Tab><latex 公式>"
                       （格式与 data/caption.txt 一致）
  3. --content_dir     内容图文件夹；content_dir + 文件名 = 内容图完整路径
  4. --style           风格列表（>=1 项）。元素可以是：
                         - 单张风格图路径   → 固定使用这一张
                         - 风格图文件夹路径 → 每张内容图随机挑一张
  5. --output_dir      输出文件夹列表，与 --style **长度必须对齐**

风格列表与输出列表一一对应：第 i 个风格跑完所有内容图，结果存到第 i 个输出文件夹，
文件名与内容图文件名保持一致。

用法示例：
    python scripts/batch_infer.py \
        --config config/default.yaml \
        --checkpoint checkpoints/exp_002/checkpoint_step_0410000.pt \
        --caption_file data/caption.txt \
        --content_dir /home/user/data3/daxger9/HMER/pdf2img/img/val_dataset/val_dataset-cambria \
        --style  data/style2/18_em_3.png  data/style3 \
        --output_dir eval_output/style_a eval_output/style_b

python scripts/batch_infer.py \
    --config config/default.yaml \
    --checkpoint checkpoints/exp_002/checkpoint_step_0410000.pt \
    --caption_file /home/user/data3/daxger9/HMER/latex_filling/val_dataset/generated_formulas.txt \
    --content_dir /home/user/data3/daxger9/HMER/pdf2img/img/val_dataset/val_dataset-cambria \
    --style  data/style1/18_em_3.png  data/style2/18_em_3.png \
    --output_dir eval_output/exp_002/val_dataset/style1 eval_output/exp_002/val_dataset/style2
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 允许从任意目录直接运行：把仓库根目录加入 sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from tqdm import tqdm

from config.config_loader import load_config, set_config
from inference.inferencer import Inferencer
from main import build_pipeline, set_seed

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


# ── 解析 caption 文件 ────────────────────────────────────────────────

def load_caption_file(path: str) -> List[Tuple[str, str]]:
    """
    解析 caption 文件 → [(文件名, 公式字符串), ...]（保持文件顺序）。

    兼容两种分隔：Tab（优先，与数据文件一致）或空格（第一个 token 是文件名）。
    文件名可能带扩展名也可能不带（data/caption.txt 里是不带扩展名的 key），
    具体文件在 resolve_content_path 里再补全。
    """
    items: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.rstrip("\n").strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                name, formula = line.split("\t", 1)
            else:
                parts = line.split()
                if len(parts) < 2:
                    print(f"[caption] line {lineno}: 只有文件名没有公式，按空 caption 处理: {line!r}")
                    name, formula = parts[0], ""
                else:
                    name, formula = parts[0], " ".join(parts[1:])
            items.append((name.strip(), formula.strip()))
    return items


def resolve_content_path(content_dir: str, name: str) -> Optional[str]:
    """
    content_dir + 文件名 → 内容图完整路径。

    文件名可能缺扩展名（caption.txt 常用 stem 作 key），这里依次尝试：
      原名 → 原名+各扩展名 → 目录下同名 stem 的任意图片。
    找不到返回 None。
    """
    direct = os.path.join(content_dir, name)
    if os.path.isfile(direct):
        return direct

    if not os.path.splitext(name)[1]:
        for ext in IMAGE_EXTS:
            cand = os.path.join(content_dir, name + ext)
            if os.path.isfile(cand):
                return cand
        stem_hits = sorted(
            p for p in Path(content_dir).glob(name + ".*")
            if p.suffix.lower() in IMAGE_EXTS
        )
        if stem_hits:
            return str(stem_hits[0])
    return None


# ── 风格列表 ────────────────────────────────────────────────────────

def list_images_in_dir(d: str) -> List[str]:
    return sorted(
        str(p) for p in Path(d).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def resolve_style_entry(entry: str) -> Tuple[str, List[str]]:
    """
    解析风格列表的一项。

    Returns:
        (kind, candidates)
        kind == "file"：固定单张风格图，candidates = [该图]
        kind == "dir" ：风格图文件夹，candidates = 该文件夹下所有图片（每次随机挑一张）
    """
    if os.path.isdir(entry):
        imgs = list_images_in_dir(entry)
        if not imgs:
            raise ValueError(f"风格文件夹里没有图片: {entry}")
        return "dir", imgs
    if os.path.isfile(entry):
        return "file", [entry]
    raise ValueError(f"风格路径既不是文件也不是文件夹: {entry}")


# ── 主流程 ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="批量推理：一组内容图 × 多组风格，分别输出到多个文件夹",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型权重路径")
    parser.add_argument("--caption_file", type=str, required=True,
                        help="caption 文件，每行 '<文件名> <latex 公式>'")
    parser.add_argument("--content_dir", type=str, required=True,
                        help="内容图文件夹（拼接 caption 里的文件名得到完整路径）")
    parser.add_argument("--style", "--style_path", dest="style", nargs="+", required=True,
                        metavar="STYLE",
                        help="风格列表：每项是一张风格图路径或一个风格图文件夹路径")
    parser.add_argument("--output_dir", "--output_dirs", dest="output_dir", nargs="+", required=True,
                        metavar="OUT_DIR",
                        help="输出文件夹列表，与 --style 长度对齐")
    parser.add_argument("--device", type=str, default=None,
                        help="覆盖 config.mode.device，如 cuda:0")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子（风格图抽样 + 初始噪声）；缺省用 config.mode.seed")
    parser.add_argument("--limit", type=int, default=0,
                        help="只跑前 N 张内容图（0 或负数表示全部），用于快速验证")
    parser.add_argument("--skip_existing", action="store_true",
                        help="输出文件已存在时跳过（便于断点续跑）")
    parser.add_argument("--empty_cache_every", type=int, default=50,
                        help="每生成多少张清一次 CUDA 缓存（0 表示不清）")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 校验列表 ──
    if len(args.style) != len(args.output_dir):
        raise SystemExit(
            f"[error] --style 有 {len(args.style)} 项，--output_dir 有 {len(args.output_dir)} 项，长度必须对齐"
        )
    if len(args.style) < 1:
        raise SystemExit("[error] --style / --output_dir 至少需要一项")

    style_entries = [resolve_style_entry(s) for s in args.style]
    for d in args.output_dir:
        os.makedirs(d, exist_ok=True)

    # ── 内容图 + caption ──
    caption_items = load_caption_file(args.caption_file)
    print(f"[caption] {len(caption_items)} 行，来自 {args.caption_file}")

    samples: List[Tuple[str, str, str]] = []   # (content_path, caption, out_name)
    missing: List[str] = []
    for name, formula in caption_items:
        path = resolve_content_path(args.content_dir, name)
        if path is None:
            missing.append(name)
            continue
        samples.append((path, formula, os.path.basename(path)))

    if missing:
        print(f"[warn] {len(missing)} 张内容图在 {args.content_dir} 下找不到，已跳过"
              f"（前 5 个：{missing[:5]}）")
    if not samples:
        raise SystemExit(f"[error] 没有可用的内容图：{args.content_dir} + {args.caption_file}")

    if args.limit and args.limit > 0:
        samples = samples[: args.limit]
    print(f"[data] 待推理内容图 {len(samples)} 张 × 风格 {len(style_entries)} 组 "
          f"= {len(samples) * len(style_entries)} 次生成")

    # ── 构建管线（只做一次）──
    config = load_config(args.config)
    if args.device:
        config.mode.device = args.device
    set_config(config)
    seed = args.seed if args.seed is not None else config.mode.seed
    set_seed(seed)

    print(f"[model] building pipeline on {config.mode.device} ...")
    pipeline = build_pipeline(config)
    inferencer = Inferencer(config, pipeline)
    inferencer.load_checkpoint(args.checkpoint)
    print(f"[model] ready. checkpoint={args.checkpoint}, seed={seed}")

    rng = random.Random(seed)
    t_start = time.time()
    failures: List[str] = []

    for si, ((kind, candidates), out_dir) in enumerate(zip(style_entries, args.output_dir)):
        kind_desc = f"文件夹（{len(candidates)} 张，随机抽）" if kind == "dir" else "单张固定"
        print(f"\n[style {si + 1}/{len(style_entries)}] {args.style[si]}  [{kind_desc}] -> {out_dir}")

        pbar = tqdm(samples, desc=f"style{si + 1}", ncols=100)
        for idx, (content_path, caption, out_name) in enumerate(pbar):
            out_path = os.path.join(out_dir, out_name)
            if args.skip_existing and os.path.isfile(out_path):
                continue

            style_path = rng.choice(candidates) if kind == "dir" else candidates[0]
            try:
                inferencer.generate(
                    print_image=content_path,
                    style_image=style_path,
                    output_path=out_path,
                    caption=caption,
                )
            except Exception as e:  # 单张失败不中断整批
                msg = f"[fail] {content_path} | style={style_path} | {type(e).__name__}: {e}"
                print(msg)
                failures.append(msg)

            if args.empty_cache_every and (idx + 1) % args.empty_cache_every == 0 \
                    and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 把本次风格的失败清单落到输出文件夹，方便复现
        if failures:
            with open(os.path.join(out_dir, "failures.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(failures) + "\n")

    elapsed = time.time() - t_start
    n_total = len(samples) * len(style_entries)
    print(f"\n[done] {n_total} 次生成，用时 {elapsed / 60:.1f} min "
          f"（{elapsed / max(1, n_total):.2f} s/张）")
    if failures:
        print(f"[done] 失败 {len(failures)} 张，已写入各输出目录的 failures.txt")
    else:
        print("[done] 全部成功")


if __name__ == "__main__":
    main()
