"""
阶段 B 验收脚本。

输出三类指标：
  1. **seen writer 检索 R@1**（阈值 ≥ 0.95）：参与过训练的风格，衡量「同风格紧致」。
  2. **unseen writer 检索 R@1**（阈值 ≥ 0.70，留一法）：**zero-shot 的核心指标**。
     ⚠️ hold-out writer 数 **必须 ≥ 20**，否则该指标统计上不可信，脚本会明确标注
     「仅作定性参考，不得作为验收依据」。
  3. **内容泄露探测**（必须做）：用冻结特征做「公式 id」最近邻分类，
     若 top-1 准确率接近随机基线（≈ 1/公式数）→ 无泄露；显著高于随机 → 存在泄露，
     **不得进入阶段 C**。

另外报告 f_s_pooled 的通道 std（阈值 > 0.01），用于发现特征塌缩。

用法：
    python scripts/eval_style_encoder.py --data_root ./data \
        --ckpt ./style_ckpt/stage_b/style_encoder_final.pt --height 64

    # 手动指定 hold-out（ckpt 里没存时）
    python scripts/eval_style_encoder.py --data_root ./data --ckpt <ckpt> \
        --holdout_writers 20 --holdout_seed 42

    # 同时报告外部数据集 hold-out（2.5 节的外部手写数据路线）
    python scripts/eval_style_encoder.py --data_root ./data --extra_roots /data/IAM \
        --ckpt <ckpt> --external_holdout 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn.functional as F
from PIL import Image

from data.style_corpus import StyleCorpus
from data.transforms import pad_style_batch, style_transform_pil
from models.style_encoder import ConvNeXtStyleEncoder


# ════════════════════════════════════════════════════════════════════

@torch.no_grad()
def encode_paths(
    encoder,
    paths: Sequence[str],
    height: int,
    max_width: Optional[int],
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """批量抽取 f_s_pooled → (N, D)（CPU float32）。"""
    feats: List[torch.Tensor] = []
    for i in range(0, len(paths), batch_size):
        chunk = list(paths[i: i + batch_size])
        tensors = []
        for p in chunk:
            try:
                with Image.open(p) as img:
                    tensors.append(style_transform_pil(
                        img, height=height, max_width=max_width, crop="center"
                    ))
            except Exception:
                # 坏图：全白（前景 mask 全 0 ⇒ null token 兜底）
                tensors.append(torch.ones(3, height, max(1, height)))
        x, mask = pad_style_batch(tensors, pad_value=1.0, pad_side="right")
        _, pooled = encoder(x.to(device), mask.to(device))
        feats.append(pooled.float().cpu())
    return torch.cat(feats, dim=0) if feats else torch.zeros(0, encoder.feature_dim)


def retrieval_r1(feats: torch.Tensor, writer_ids: torch.Tensor) -> Tuple[float, torch.Tensor]:
    """
    最近邻检索 R@1（余弦相似度；排除自身）。

    Returns:
        (整体 R@1, 每个 query 是否命中的 bool tensor)
    """
    f = F.normalize(feats.float(), dim=-1)
    sim = f @ f.transpose(0, 1)                                    # (N, N)
    N = sim.shape[0]
    sim = sim.clone()
    sim.fill_diagonal_(-1e9)
    top1 = sim.argmax(dim=1)                                       # (N,)
    hit = (writer_ids[top1] == writer_ids)                         # (N,)
    return float(hit.float().mean().item()), hit


def content_leakage_probe(
    feats: torch.Tensor,
    formula_ids: Sequence[str],
    writer_ids: torch.Tensor,
    num_probe_formulas: int = 200,
    seed: int = 42,
) -> Dict[str, float]:
    """
    内容泄露探测：冻结特征 → 预测该风格图对应的是哪条公式。

    做法：**最近质心分类器**（不引入额外依赖）。
    为避免 writer 泄漏，同一条公式的拟合图与测试图来自**不同的 writer**。

    Returns:
        {"top1": acc, "random": 1/F, "ratio": acc / random, "n_formulas": F, "n_test": n}
    """
    by_formula: Dict[str, List[int]] = {}
    for idx, f in enumerate(formula_ids):
        by_formula.setdefault(f, []).append(idx)
    # 只保留 writer 数 >= 4 的公式（否则没法做 writer 不重叠的划分）
    usable = {f: idxs for f, idxs in by_formula.items() if len(idxs) >= 4}
    if not usable:
        return {"top1": float("nan"), "random": float("nan"), "ratio": float("nan"),
                "n_formulas": 0, "n_test": 0}

    rng = random.Random(seed)
    names = sorted(usable)
    rng.shuffle(names)
    names = names[: max(1, num_probe_formulas)]

    centroids: List[torch.Tensor] = []
    test_feats: List[torch.Tensor] = []
    test_labels: List[int] = []

    for ci, f in enumerate(names):
        idxs = list(usable[f])
        rng.shuffle(idxs)
        half = len(idxs) // 2
        fit_idx, test_idx = idxs[:half], idxs[half:]
        if not fit_idx or not test_idx:
            continue
        centroids.append(F.normalize(feats[fit_idx].float(), dim=-1).mean(dim=0))
        test_feats.append(feats[test_idx].float())
        test_labels.extend([ci] * len(test_idx))

    if not centroids:
        return {"top1": float("nan"), "random": float("nan"), "ratio": float("nan"),
                "n_formulas": 0, "n_test": 0}

    C = torch.stack(centroids, dim=0)                 # (F, D)
    C = F.normalize(C, dim=-1)
    T = torch.cat(test_feats, dim=0)                  # (n, D)
    T = F.normalize(T, dim=-1)
    labels = torch.tensor(test_labels, dtype=torch.long)

    pred = (T @ C.transpose(0, 1)).argmax(dim=1)      # (n,)
    acc = float((pred == labels).float().mean().item())
    rand = 1.0 / len(centroids)
    return {
        "top1": acc, "random": rand, "ratio": acc / rand,
        "n_formulas": float(len(centroids)), "n_test": float(labels.numel()),
    }


# ════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="阶段 B 验收：writer 检索 R@1 / 内容泄露探测 / 特征塌缩检查",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--extra_roots", type=str, nargs="*", default=[])
    p.add_argument("--ckpt", type=str, required=True, help="阶段 B 产出的 style_encoder 权重")
    p.add_argument("--height", type=int, default=0, help="0 = 从 ckpt 里读")
    p.add_argument("--max_width", type=int, default=0, help="0 = 从 ckpt 里读（缺省 8*H）")
    p.add_argument("--per_writer", type=int, default=60, help="每个 writer 抽多少张图")
    p.add_argument("--holdout_writers", type=int, default=0,
                   help="ckpt 里没存 holdout 名单时手动指定数量")
    p.add_argument("--holdout_seed", type=int, default=42)
    p.add_argument("--num_probe_formulas", type=int, default=200, help="内容泄露探测用多少条公式")
    p.add_argument("--eval_batch", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_json", type=str, default="", help="可选：把指标写成 json")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    meta = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    height = int(args.height or meta.get("height", 64))
    max_width = int(args.max_width or meta.get("max_width", 0)) or 8 * height
    feature_dim = int(meta.get("feature_dim", 768))
    num_query = int(meta.get("num_query", 4))
    fg_threshold = float(meta.get("fg_threshold", 0.0))
    backbone = str(meta.get("backbone", "convnext_tiny"))

    encoder = ConvNeXtStyleEncoder(
        backbone=backbone, pretrained=False, feature_dim=feature_dim,
        height=height, num_query=num_query, fg_threshold=fg_threshold,
        max_width=max_width, device=str(device),
    )
    encoder.load_pretrained(args.ckpt, strict=False)
    encoder.eval()

    # ── hold-out 名单 ──
    holdout = list(ckpt.get("holdout_writers", []) or []) if isinstance(ckpt, dict) else []
    corpus_full = StyleCorpus([args.data_root] + list(args.extra_roots), verbose=True)
    if not holdout and args.holdout_writers > 0:
        holdout, _ = StyleCorpus.split_holdout(
            corpus_full.writer_names, args.holdout_writers, seed=args.holdout_seed
        )
    holdout_set = set(holdout)

    # ── 抽特征 ──
    rng = random.Random(args.seed)
    paths: List[str] = []
    writers: List[str] = []
    formulas: List[str] = []
    for w in corpus_full.writer_names:
        items = corpus_full.images_of_writer(w, limit=args.per_writer)
        rng.shuffle(items)
        for f, p in items:
            paths.append(p)
            writers.append(w)
            formulas.append(f)

    print(f"\n[eval] {len(paths)} 张图 / {len(set(writers))} 个 writer，抽取特征中 ...")
    feats = encode_paths(encoder, paths, height, max_width, device, args.eval_batch)
    writer_ids = torch.tensor(
        [corpus_full.writer_to_id[w] for w in writers], dtype=torch.long
    )
    is_unseen = torch.tensor([1 if w in holdout_set else 0 for w in writers], dtype=torch.bool)

    print("\n" + "=" * 68)
    print("阶段 B 验收报告")
    print("=" * 68)

    # ── 1. 检索 R@1 ──
    r1_all, hit = retrieval_r1(feats, writer_ids)
    n_unseen = int(is_unseen.sum().item())
    n_seen = int((~is_unseen).sum().item())
    r1_seen = float(hit[~is_unseen].float().mean().item()) if n_seen else float("nan")
    r1_unseen = float(hit[is_unseen].float().mean().item()) if n_unseen else float("nan")

    print(f"\n[1] writer 检索 R@1（最近邻，排除自身）")
    print(f"    seen   writer R@1 = {r1_seen:.4f}   (n={n_seen}, 阈值 ≥ 0.95) "
          f"{'✅' if r1_seen >= 0.95 else '❌'}")
    if n_unseen:
        n_holdout_writers = len({w for w, u in zip(writers, is_unseen.tolist()) if u})
        credible = n_holdout_writers >= 20
        print(f"    unseen writer R@1 = {r1_unseen:.4f}   (n={n_unseen}, "
              f"hold-out writer 数={n_holdout_writers}, 阈值 ≥ 0.70) "
              f"{'✅' if r1_unseen >= 0.70 else '❌'}")
        if not credible:
            print(f"    ⚠️  hold-out writer 数 {n_holdout_writers} < 20：该指标在统计上不可信，"
                  f"**只能作定性参考，不得作为验收通过依据**")
            print(f"        （15 种风格做留一法只有 1~2 个类，测不出泛化；见实施方案 2.5 节）")
    else:
        print("    unseen writer R@1 = n/a（没有 hold-out writer；"
              "训练时加 --holdout_writers N 才能测 zero-shot）")

    # ── 2. 特征塌缩 ──
    ch_std = float(feats.float().std(dim=0).mean().item())
    print(f"\n[2] 特征塌缩检查")
    print(f"    f_s_pooled 通道 std = {ch_std:.5f}   (阈值 > 0.01) "
          f"{'✅' if ch_std > 0.01 else '❌（特征塌缩，检查 SupCon 温度 / 难负样本是否生效）'}")

    # ── 3. 内容泄露探测 ──
    probe = content_leakage_probe(
        feats, formulas, writer_ids, num_probe_formulas=args.num_probe_formulas, seed=args.seed
    )
    print(f"\n[3] 内容泄露探测（冻结特征 → 预测公式 id，最近质心分类）")
    if probe["n_formulas"]:
        print(f"    公式数 F={int(probe['n_formulas'])}, 测试样本 {int(probe['n_test'])}")
        print(f"    top-1 = {probe['top1']:.4f}   随机基线 = {probe['random']:.5f}   "
              f"比值 = {probe['ratio']:.2f}×")
        if probe["ratio"] < 3.0:
            print("    ✅ 接近随机基线 ⇒ 基本没有把公式内容编码进风格特征")
        elif probe["ratio"] < 10.0:
            print("    ⚠️  显著高于随机基线 ⇒ 存在内容泄露，建议加强难负样本再进阶段 C")
        else:
            print("    ❌ 严重泄露 ⇒ 风格编码器在「抄参考答案」，不得进入阶段 C")
    else:
        print("    [跳过] 没有足够多「被 ≥4 个 writer 写过」的公式，无法构造探测集")

    # ── 汇总 ──
    print("\n" + "=" * 68)
    verdicts = []
    verdicts.append(("seen R@1 ≥ 0.95", r1_seen >= 0.95))
    if n_unseen and len({w for w, u in zip(writers, is_unseen.tolist()) if u}) >= 20:
        verdicts.append(("unseen R@1 ≥ 0.70", r1_unseen >= 0.70))
    verdicts.append(("f_s std > 0.01", ch_std > 0.01))
    if probe["n_formulas"]:
        verdicts.append(("无内容泄露（< 3× 随机）", probe["ratio"] < 3.0))
    for name, ok in verdicts:
        print(f"  {'✅' if ok else '❌'} {name}")
    print("=" * 68)

    if args.out_json:
        payload = {
            "r1_all": r1_all, "r1_seen": r1_seen, "r1_unseen": r1_unseen,
            "n_seen": n_seen, "n_unseen": n_unseen,
            "n_holdout_writers": len(holdout_set),
            "f_s_channel_std": ch_std,
            "content_leakage": probe,
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[eval] 指标已写入 {args.out_json}")


if __name__ == "__main__":
    main()
