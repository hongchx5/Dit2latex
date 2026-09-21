"""
静态分桶 + caption 开关 端到端冒烟验证。

覆盖：
1. 静态分桶：dataset 按 all_size 分类，只加载 train_size 的图
2. caption 开启（use_caption=True）：DiT 前向/反向 + pipeline + DDIM
3. caption 关闭（use_caption=False）：DiT 前向/反向（caption_seq=None）+ pipeline + DDIM

运行：python scripts/smoke_test.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAPTION = os.path.join(PROJ, "data", "caption.txt")
DICT = os.path.join(PROJ, "data", "dictionary.txt")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_image(path, w, h):
    img = Image.new("RGB", (w, h), (255, 255, 255))
    ImageDraw.Draw(img).text((w // 10, h // 3), "x+1", fill=(0, 0, 0))
    img.save(path)


class FakePerceptual(torch.nn.Module):
    def forward(self, x, y):
        return F.mse_loss(x, y)


def _build_pipe(use_caption):
    from models.vae import OfflineVAE
    from models.style_encoder import OfflineStyleEncoder
    from models.caption_encoder import CaptionEncoder
    from models.dit.dit import DiT
    from models.pipeline import DiTtolatexPipeline
    from diffusion.noise_schedule import NoiseSchedule

    H = 192
    heads = 3
    vae = OfflineVAE(latent_dim=4, f=8).to(DEV)
    style_enc = OfflineStyleEncoder(feature_dim=H).to(DEV)
    cap_enc = CaptionEncoder(vocab_size=112, hidden_dim=H, max_len=256, num_layers=1, num_heads=heads).to(DEV) if use_caption else None
    dit = DiT(hidden_dim=H, num_heads=heads, depth=3, patch_size=2, in_channels=8, out_channels=8,
              latent_dim=4, context_dim=H, use_caption=use_caption).to(DEV)
    ns = NoiseSchedule(num_timesteps=100, beta_schedule="cosine", device=DEV)
    pipe = DiTtolatexPipeline(
        vae=vae, style_encoder=style_enc, caption_encoder=cap_enc, dit=dit,
        noise_schedule=ns, perceptual_loss=FakePerceptual(),
        perceptual_loss_weight=0.0, cfg_dropout_rate=0.1, device=str(DEV),
    ).to(DEV)
    return pipe


def test_static_bucket():
    from data.dataset import HandwrittenFormulaDataset, bucket_collate
    from torch.utils.data import DataLoader

    tmp = tempfile.mkdtemp(prefix="ds_")
    try:
        os.makedirs(os.path.join(tmp, "print"))
        os.makedirs(os.path.join(tmp, "style1"))
        # 不同宽高比，分别分类到不同 all_size 桶
        make_image(os.path.join(tmp, "print", "a.png"), 512, 64)    # -> 64x512 类
        make_image(os.path.join(tmp, "print", "b.png"), 512, 128)   # -> 128x512 类
        make_image(os.path.join(tmp, "print", "c.png"), 256, 256)   # -> 256x256 类
        make_image(os.path.join(tmp, "print", "d.png"), 512, 96)    # -> 96x512 类
        for i in range(3):
            make_image(os.path.join(tmp, "style1", f"s{i}.png"), 300, 128)

        all_size = [(64, 512), (96, 512), (128, 512), (256, 256)]
        train_size = (128, 512)

        ds = HandwrittenFormulaDataset(
            data_root=tmp, buckets=all_size, train_size=train_size,
            repeats_per_image=1, styles_per_repeat=1, style_as_tensor=True, vae_f=8,
        )
        # 只加载分类到 128x512 的图：只有 b.png（512x128）
        assert len(ds.print_images) == 1, ds.print_images
        assert ds.print_images[0] == "b.png"
        assert ds.buckets_with_data() == [(128, 512)]
        print(f"[1] static bucket: only {ds.print_images} loaded for train_size={train_size} OK")

        loader = DataLoader(ds, batch_size=1, collate_fn=bucket_collate)
        I_p, I_s, I_t, buckets, cap_ids, cap_mask = next(iter(loader))
        assert I_p.shape[2:] == (128, 512)
        assert buckets[0].tolist() == [128, 512]
        print(f"[1] static bucket collate: I_p {tuple(I_p.shape)} OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_caption_on():
    from diffusion.ddim import ddim_sample
    pipe = _build_pipe(use_caption=True)

    # DiT 前向 + 反向
    B, L = 2, 6
    z = torch.randn(B, 8, 16, 32, device=DEV)  # 128x256 桶 latent
    f_seq = torch.randn(B, 4, 192, device=DEV)
    f_pooled = torch.randn(B, 192, device=DEV)
    cap_seq = torch.randn(B, L, 192, device=DEV)
    cap_mask = torch.zeros(B, L, dtype=torch.bool, device=DEV)
    cap_mask[:, 4:] = True
    t = torch.randint(0, 100, (B,), device=DEV)
    out = pipe.dit(z, f_seq, f_pooled, cap_seq, cap_mask, t)
    assert out.shape == (B, 4, 16, 32)
    out.mean().backward()
    assert pipe.dit.blocks[0].caption_attn.to_kv.weight.grad is not None
    pipe.zero_grad()

    # pipeline 端到端
    I_p = torch.rand(B, 3, 128, 256, device=DEV) * 2 - 1
    I_t = torch.rand(B, 3, 128, 256, device=DEV) * 2 - 1
    I_s = [torch.rand(3, 224, 224, device=DEV) * 2 - 1 for _ in range(B)]
    cap_ids = torch.randint(0, 112, (B, L), device=DEV)
    outs = pipe(I_p, I_s, I_t, cap_ids, cap_mask)
    assert torch.isfinite(outs["loss"])
    outs["loss"].backward()
    assert pipe.caption_encoder.token_embed.weight.grad is not None
    pipe.zero_grad()
    print(f"[2] caption ON: DiT + pipeline forward/backward OK (loss={outs['loss'].item():.4f})")

    # DDIM（注意：cap_mask 需与 z_t 同 batch=1，避免 batch 污染）
    with torch.no_grad():
        z_t = torch.randn(1, 4, 16, 32, device=DEV)
        z_p = torch.randn(1, 4, 16, 32, device=DEV)
        f_seq = torch.randn(1, 4, 192, device=DEV)
        f_pooled = torch.randn(1, 192, device=DEV)
        cap_seq = torch.randn(1, L, 192, device=DEV)
        cap_mask1 = torch.zeros(1, L, dtype=torch.bool, device=DEV)
        z0 = ddim_sample(pipe, pipe.noise_schedule, z_t, z_p, f_pooled, f_seq, cap_seq, cap_mask1,
                         num_steps=5, cfg_scale=2.0)
        assert z0.shape == z_t.shape and torch.isfinite(z0).all()
    print("[2] caption ON: DDIM OK")


def test_caption_off():
    from diffusion.ddim import ddim_sample
    pipe = _build_pipe(use_caption=False)
    assert pipe.caption_encoder is None
    assert pipe.dit.blocks[0].caption_attn is None

    # DiT 前向 + 反向（caption_seq=None）
    B, L = 2, 6
    z = torch.randn(B, 8, 16, 32, device=DEV)
    f_seq = torch.randn(B, 4, 192, device=DEV)
    f_pooled = torch.randn(B, 192, device=DEV)
    t = torch.randint(0, 100, (B,), device=DEV)
    out = pipe.dit(z, f_seq, f_pooled, None, None, t)
    assert out.shape == (B, 4, 16, 32)
    out.mean().backward()
    pipe.zero_grad()

    # pipeline 端到端（caption_ids 任意，但 encoder=None 会忽略）
    I_p = torch.rand(B, 3, 128, 256, device=DEV) * 2 - 1
    I_t = torch.rand(B, 3, 128, 256, device=DEV) * 2 - 1
    I_s = [torch.rand(3, 224, 224, device=DEV) * 2 - 1 for _ in range(B)]
    cap_ids = torch.zeros(B, 1, dtype=torch.long, device=DEV)
    cap_mask = torch.ones(B, 1, dtype=torch.bool, device=DEV)
    outs = pipe(I_p, I_s, I_t, cap_ids, cap_mask)
    assert torch.isfinite(outs["loss"])
    outs["loss"].backward()
    pipe.zero_grad()
    print(f"[3] caption OFF: DiT + pipeline forward/backward OK (loss={outs['loss'].item():.4f})")

    # DDIM（caption_seq=None）
    with torch.no_grad():
        z_t = torch.randn(1, 4, 16, 32, device=DEV)
        z_p = torch.randn(1, 4, 16, 32, device=DEV)
        f_seq = torch.randn(1, 4, 192, device=DEV)
        f_pooled = torch.randn(1, 192, device=DEV)
        z0 = ddim_sample(pipe, pipe.noise_schedule, z_t, z_p, f_pooled, f_seq, None, None,
                         num_steps=5, cfg_scale=2.0)
        assert z0.shape == z_t.shape and torch.isfinite(z0).all()
    print("[3] caption OFF: DDIM OK")


def main():
    test_static_bucket()
    test_caption_on()
    test_caption_off()
    print("\nALL STATIC-BUCKET + CAPTION-SWITCH SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
