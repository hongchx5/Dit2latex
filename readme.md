DiTtoLatex/
├── 代码构建.prompt        ← 详细代码构建规范
├── requirements.txt       ← 11 个依赖
├── main.py                ← CLI 入口 (train/infer/eval)
├── config/
│   ├── default.yaml       ← 所有超参数（桶集合 / tiling / 缓存配置）
│   └── config_loader.py   ← YAML→dataclass 解析
├── data/
│   ├── buckets.py         ← 动态分桶：选桶 / contain 缩放 + 居中对称填充 / 裁剪还原
│   ├── dataset.py         ← bucket-aware 采样：每张 print 图→固定桶，9 条三元组
│   ├── transforms.py      ← 内容图桶变换 + 风格图 tiling + **风格图整图预处理（缩放到 H）**
│   ├── style_cache.py     ← 风格 tile 特征缓存（键 = 路径 + tiling 参数；仅 clip_tiled 基线）
│   ├── style_corpus.py    ← 风格语料索引 (writer, formula) → path + 矩形网格采样
│   └── style_augment.py   ← 风格图笔迹增广（仿射 / 形态学 / 墨色；不破坏内容）
├── models/
│   ├── vae.py             ← SD VAE / OffLine AvgPool（8× 下采样，任意尺寸）
│   ├── style_encoder.py   ← **ConvNeXtStyleEncoder**（整图 + 前景感知 4-query 聚合）
│   │                        / TiledCLIPStyleEncoder（基线）/ Offline CNN
│   └── pipeline.py        ← 训练/推理共用管线
├── dit/
│   ├── dit.py             ← DiT 完整模型（12 层，动态分辨率）
│   ├── dit_block.py       ← AdaLN→SA→AdaLN→CA→SwiGLU
│   ├── adaln.py           ← AdaLN-Zero（α零初始化）
│   ├── attention.py       ← Self-Attn 2D RoPE / Cross-Attn
│   └── patch_embed.py     ← Patch（动态 grid，无固定位置编码 buffer）
├── diffusion/
│   ├── noise_schedule.py  ← 余弦 θ 调度 + q_sample
│   └── ddim.py            ← DDIM + CFG 采样
├── losses/
│   ├── diffusion_loss.py  ← MSE 噪声预测损失
│   └── perceptual_loss.py← VGG relu3_3 感知损失
├── train/
│   ├── trainer.py         ← 桶循环训练（一桶 N 步再换）+ AMP + val + 多卡 DDP
│   ├── optimizer.py       ← AdamW + warmup + cosine decay
│   └── checkpoint.py     ← 保存/恢复 + EMA
├── inference/
│   └── inferencer.py      ← 风格条件推理（选桶 + tiling + 输出裁剪）
├── eval/
│   ├── fid_lpips.py       ← FID / LPIPS
│   └── style_metrics.py   ← 风格检索 / Gram MSE
└── scripts/
    ├── train.sh
    ├── infer.sh
    ├── precompute_style_cache.py  ← 预计算风格 tile 特征缓存（仅 clip_tiled 基线）
    ├── smoke_test.py              ← 全链路冒烟测试
    ├── stat_style_height.py       ← 统计风格图高度分布 → 推荐 H / max_width
    ├── pretrain_style_dino.py     ← 阶段 A：DINO 自监督预训练
    ├── train_style_encoder.py     ← 阶段 B：SupCon + ArcFace 度量学习微调
    ├── eval_style_encoder.py      ← 阶段 B 验收：R@1 / 内容泄露探测 / 塌缩检查
    └── batch_infer.py             ← 批量推理（内容图 × 多组风格）

关键设计要点汇总：

- 动态分桶：桶集合（物理像素，H×W）= (64,1024) (96,512) (128,512) (128,384) (256,256)；
  每张内容图按宽高比确定性选桶（最小填充），等比缩放使一条边恰好容入桶，
  内容居中、对称白色填充（奇数余量右/下多一列/行）；同 batch 同桶、尺寸统一。
- 位置编码：DiT 内 Self-Attention 使用 2D RoPE（head_dim 前半行坐标、后半列坐标），
  天然支持任意 grid 尺寸；PatchEmbed 不再携带固定 sin-cos buffer。
- 风格图（**新，默认**）：整图等比缩放到固定高度 H（默认 64，**绝不压扁**），
  batch 内按最宽样本右侧 padding（**填充 +1.0 白色背景**，严禁用 0）；
  ConvNeXt-T（**stem stride 4→2，总 stride 16**）+ 归一化 2D 正弦位置编码
  + 前景感知 4-query 聚合（前景 mask 用 **max_pool** 下采样，细笔画才不会被稀释）
  + 末尾追加可学习 **null token**（NaN 安全防护）→ f_s_seq (B,4,768) / f_s_pooled (B,768)。
  backbone 参与反传，**不使用 tile 特征缓存**。
- 风格图（**旧基线，配置开关** `model.style_encoder.type: "clip_tiled"`）：
  等比缩放到 height=224 → 以 stride=168 滑动切 224×224 小块，尾块丢弃；
  逐块过冻结 CLIP → 1D 正弦位置编码（pos=alpha·2π，alpha=i/(T-1)，T=1 时 0.5，加在 K/V 侧）
  → 4 个可学习查询 Cross-Attention 聚合 → 同上两个输出，带 tile 特征缓存。
- 风格 tile 特征缓存：预计算脚本可提前生成（路径+tiling 参数为键）；训练/推理命中直接
  用，未命中临时计算并写回；聚合模块可学习、不缓存。
- 桶循环训练：一个桶连续训练 bucket_steps 步后随机换桶（避免频繁换桶负载抖动）。
- 其余：z_t 与 z_p 通道拼接 → z_t'（8 通道）；输出 8 通道取前 4 通道作 ε̂；
  训练损失 = MSE + 0.1× 感知损失，CFG dropout 10%；offline_test=true 时 VAE/CLIP 替换为简单层。

# 预计算风格 tile 特征缓存（可选，加速训练首轮）

python scripts/precompute_style_cache.py
    --data_root ./data --cache_dir ./style_cache
    --model_name ./pretrained/openai/clip-vit-base-patch32

# 正常模式训练

python main.py train --config config/default.yaml

# 多卡 DDP 训练（config 中 distributed.enabled: true + gpus 列表；启动命令不变）

# distributed:

# enabled: true

# gpus: ["cuda:0", "cuda:1"]

# 说明：每卡一个进程（batch_size 为每卡值）；桶切换由共享种子确定性序列驱动，

# 所有进程同步；checkpoint / TensorBoard / 验证仅由 rank 0 执行。

python main.py train --config config/default.yaml

# Offline 测试模式（修改 yaml 中 mode.offline_test: true）

python main.py train --config config/default.yaml

# 推理（可变分辨率：print 图按 data.max_tokens 的 token 预算等比缩放，保留原始宽高比，
#      输出尺寸随输入宽高比变化；风格参考图任意分辨率均可）

python main.py infer --print_path data/print/img1.png --style_path data/style1/img2.png --output result.png
python main.py infer --checkpoint checkpoints/checkpoint_step_0157500.pt --print_path data/print/95_miguel.png --style_path data/style1/76_miguel.png --output result.png

# 推理 + caption 条件（--caption 传入空格 split 的 latex 公式；需 config 中 model.caption.enabled: true）

python main.py infer --print_path data/print/img1.png --style_path data/style1/img2.png --output result.png --caption "x + 1 = 2"

python main.py infer --checkpoint checkpoints/exp_002/checkpoint_step_0110000.pt --print_path data/print/106_carlos.png --style_path data/test1/18_em_0.bmp --output a1.png --caption "y ^ { 4 } + y + 1 = 0"

python main.py infer --checkpoint checkpoints/exp_002/checkpoint_step_0410000.pt --print_path /home/user/data3/daxger9/HMER/pdf2img/img/val_dataset/val_dataset-cambria/20260921031917-3.png --style_path data/style2/18_em_3.png --output eval_output/val_dataset/20260921031917-3.png --caption "9 7 7 f - 5 3 v l \leq - s - \sigma"

# 冒烟测试（无数据也可运行：全部用假数据验证形状/梯度/采样链路）

python scripts/smoke_test.py

# ════════════════════════════════════════════════════════════════════
#  风格编码器（ConvNeXt 版）三阶段训练流程
# ════════════════════════════════════════════════════════════════════
# 对应分支：feat/convnext-style-encoder（基线 feat/fit-variable-resolution）
# 规格文档：风格编码器改造实施方案.md

# 0. 先统计风格图高度分布，确定 H（默认 64；脚本会给推荐值 + 宽度截断影响）

python scripts/stat_style_height.py --data_root ./data --limit_per_dir 2000

# 1. 阶段 A：DINO 自监督预训练（**zero-shot 泛化的关键**，不需要 writer 标签）
#    单卡 / 多卡（torchrun）都可以；想补多样性可加 --extra_roots /data/IAM /data/CVL

python scripts/pretrain_style_dino.py --data_root ./data --height 64 \
    --epochs 200 --batch_size 32 --out_dir ./style_ckpt/dino

torchrun --nproc_per_node=2 scripts/pretrain_style_dino.py --data_root ./data \
    --epochs 200 --batch_size 32 --out_dir ./style_ckpt/dino

# 2. 阶段 B：SupCon + ArcFace 度量学习微调
#    核心是 writers × formulas 矩形 batch：同行 = 正样本，同列 = **难负样本**
#    （内容完全相同、只差笔迹 ⇒ 压制内容泄露的最强手段）
#    想做留一法评估就加 --holdout_writers 20（名单会写进 ckpt）

python scripts/train_style_encoder.py --data_root ./data \
    --init_ckpt ./style_ckpt/dino/dino_final.pt \
    --height 64 --batch_writers 16 --batch_formulas 16 --epochs 150 \
    --out_dir ./style_ckpt/stage_b

# 3. 阶段 B 验收：writer 检索 R@1 / 内容泄露探测 / 特征塌缩检查

python scripts/eval_style_encoder.py --data_root ./data \
    --ckpt ./style_ckpt/stage_b/style_encoder_final.pt

# 4. 阶段 C：接入 DiT 联合微调（backbone **不冻结**，用更小的 lr）

#    config/default.yaml:
#      model.style_encoder.type: "convnext"
#      model.style_encoder.init_ckpt: "./style_ckpt/stage_b/style_encoder_final.pt"
#      model.style_encoder.freeze_backbone: false
#      model.style_encoder.backbone_lr_scale: 0.5   # backbone lr = 主 lr × 0.5
#      data.style_height: 64                        # 须与 style_encoder.height 一致

python main.py train --config config/default.yaml

# 5. 切回旧基线做效果对照（走 tiling + 冻结 CLIP + tile 特征缓存）

#    config/default.yaml: model.style_encoder.type: "clip_tiled"

## 风格编码器改造 —— 交付说明

1. **ImageNet 预训练权重**：代码默认 `pretrained: true`，走 torchvision
   `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`。stem 权重 shape 为 `(96,3,4,4)`，
   stride 是运行时参数，所以 **stride 4→2 后预训练权重仍可直接加载，无需丢弃**。
   若服务器无法联网下载，脚本会 **自动降级为从零初始化并打印 warning**，
   这种情况下阶段 A（DINO）的 epoch 数需要相应提高（建议 ×2 以上）。
   **请在首次训练时看一眼启动日志确认到底走的是哪条路径。**
2. **H 的取值**：默认 **64**（16 的倍数，特征图高度 64/16 = 4，位置编码与 mask 对齐最干净）。
   最终值请以 `scripts/stat_style_height.py` 的高度中位数为准，改完必须同步
   `model.style_encoder.height` 与 `data.style_height`（两处不一致会导致裁剪宽度算错）。
3. **阶段 A 方案**：完整 DINO（teacher EMA + 多裁剪），不是简化替代方案。
   裁剪做了针对性改造：全局 = 原宽 60%~100% 的随机片段，局部 = 左/中/右/随机
   四个水平片段，**全程保持宽高比、绝不压扁**；teacher 只吃全局裁剪，
   自蒸馏对 = student 局部片段 ↔ teacher 全局图。
4. **各阶段权重路径**：
   - 阶段 A：`./style_ckpt/dino/dino_final.pt`（+ 每 `save_every` epoch 一份）
   - 阶段 B：`./style_ckpt/stage_b/style_encoder_final.pt`
   - 阶段 C：`checkpoints/checkpoint_step_*.pt`（整条 pipeline，风格编码器在内）
5. **实测吞吐**：本机**没有 torch/cuda，未实测**。定性预期如下，请在服务器上补测并回填：
   - 移除 tile 特征缓存后，风格编码器从「查缓存」变成「每个 batch 在线前向」，
     这是**新增的确定性开销**；
   - 但 ConvNeXt-T ≈ 28M 参数、输入仅 `(B,3,64,~512)`，远小于 CLIP ViT-B/32 的 86M
     与 `(T,3,224,224)` 多 tile 前向；
   - 净效应大概率**接近持平或略快**（省掉了每个 tile 一次 ViT 前向），
     但 batch 内 `W_max` 随极端宽高比样本波动会带来显存峰值抖动
     ⇒ 建议按长宽比分桶采样（现有 bucket 机制已天然做到一部分）。
6. **与实施方案的偏离**（共 3 处，均为必要修正）：
   - **stem 改造同时改了 padding**：只把 stride 从 4 改成 2 而 padding 仍为 0 时，
     输出尺寸是 `floor((H-4)/2)+1 = H/2-1`（H=64 → 31 而非 32），
     后续每层都会差一点，最终特征图与前景 mask 对不齐。
     因此同步把 padding 设为 `(k-stride)//2 = 1`，保证输出严格是 `H/2`。
     **不影响预训练权重加载**（padding 与权重 shape 无关）。
   - **DINO 用 1 个全局 + 4 个局部裁剪**，自蒸馏对取「student 局部 ↔ teacher 全局
     （同一张图）」。方案里写的「全局裁剪 1 个」是可行的，但 teacher 只吃全局、
     student 的全局与 teacher 输入完全相同 ⇒ 必须跳过该对，否则自蒸馏退化为恒等映射。
   - **style_mask 与前景 mask 的对齐做了兜底**：卷积取整与 `max_pool` 取整在极端
     尺寸下可能差 1 格，所以加了一个 crop/pad 对齐步骤（`_match_spatial`），
     mask 侧补的是 0（背景），不会误抑制真实内容。

## 不变量（改风格编码器时不要破坏）

- `style_encoder.encode(...)` 必须返回 `(f_s_seq (B,M,D), f_s_pooled (B,D))`，默认 M=4、D=768；
  只要 `feature_dim` 与输出一致，**DiT 侧零改动**（`main.py` 里
  `context_dim=config.model.style_encoder.feature_dim`）。
- `bucket_collate` 返回的元组个数保持不变：本分支 **7 元组**。
  `I_s` 从「路径列表」变成 `(style_tensor, style_mask)` **二元组**，但仍占**一个元素位**。
- CFG dropout 行为不变：`f_s_seq`、`f_s_pooled`、`z_p`、`caption_seq` 用同一个 mask 同时置零。
- 归一化沿用 `[-1, 1]`（白底 +1，墨迹 -1），与 `I_p` / `I_t` 一致。
- 阶段 C 风格编码器参数自动纳入 EMA：`trainer` 的 EMA 作用于 `pipeline.state_dict()`，
  风格编码器在 pipeline 下即可，无需额外处理。
- **不要给 ConvNeXt 风格编码器加特征缓存**：它要参与反传。
  `style_cache.py` / `scripts/precompute_style_cache.py` 保留给 `clip_tiled` 基线用。
