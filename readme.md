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
│   ├── transforms.py      ← 内容图桶变换 + 风格图平铺切分（tiling）
│   └── style_cache.py     ← 风格 tile 特征缓存（键 = 路径 + tiling 参数）
├── models/
│   ├── vae.py             ← SD VAE / OffLine AvgPool（8× 下采样，任意尺寸）
│   ├── style_encoder.py   ← TiledCLIPStyleEncoder（tiling + 4-query 聚合）/ Offline CNN
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
    ├── precompute_style_cache.py  ← 预计算风格 tile 特征缓存
    └── smoke_test.py              ← 全链路冒烟测试

关键设计要点汇总：

- 动态分桶：桶集合（物理像素，H×W）= (64,1024) (96,512) (128,512) (128,384) (256,256)；
  每张内容图按宽高比确定性选桶（最小填充），等比缩放使一条边恰好容入桶，
  内容居中、对称白色填充（奇数余量右/下多一列/行）；同 batch 同桶、尺寸统一。
- 位置编码：DiT 内 Self-Attention 使用 2D RoPE（head_dim 前半行坐标、后半列坐标），
  天然支持任意 grid 尺寸；PatchEmbed 不再携带固定 sin-cos buffer。
- 风格图平铺切分（tiling）：等比缩放到 height=224 → 以 stride=168 滑动切 224×224 小块，
  最后不足 224 的尾块丢弃；逐块过冻结 CLIP → 1D 正弦位置编码（pos=alpha·2π，
  alpha=i/(T-1)，T=1 时 alpha=0.5，加在 K/V 侧）→ 4 个可学习查询 Cross-Attention 聚合
  → f_s_seq (B,4,768) 供 Cross-Attn、concat 后 LN+MLP 4 倍压缩 f_s_pooled (B,768) 供 AdaLN。
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

# 推理（静态分桶：print 图等比缩放 + 白填充到 config.data.train_size，输出固定 train_size 尺寸）

python main.py infer --print_path data/print/img1.png --style_path data/style1/img2.png --output result.png
python main.py infer --checkpoint checkpoints/checkpoint_step_0157500.pt --print_path data/print/95_miguel.png --style_path data/style1/76_miguel.png --output result.png

# 推理 + caption 条件（--caption 传入空格 split 的 latex 公式；需 config 中 model.caption.enabled: true）

python main.py infer --print_path data/print/img1.png --style_path data/style1/img2.png --output result.png --caption "x + 1 = 2"

python main.py infer --checkpoint checkpoints/128x512_0157500.pt --print_path data/print/106_carlos.png --style_path data/test1/18_em_0.bmp --output a1.png --caption "y ^ { 4 } + y + 1 = 0"

python main.py infer --checkpoint checkpoints/128x512_0157500.pt --print_path /home/user/data3/daxger9/HMER/pdf2img/img/temp/a9.png --style_path data/style1/18_em_3.png --output eval_output/a9.png --caption "`f ( x ) = \frac { x } { 2 + x }`"

# 冒烟测试（无数据也可运行：全部用假数据验证形状/梯度/采样链路）

python scripts/smoke_test.py
