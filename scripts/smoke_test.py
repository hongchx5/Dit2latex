"""FiT 可变分辨率 + caption 开关端到端冒烟验证。

覆盖：
1. FiT 动态分桶：dataset 按 patch 预算缩放并按尺寸分桶。
2. caption 开启（use_caption=True）：DiT 前向/反向 + pipeline + DDIM。
3. caption 关闭（use_caption=False）：DiT 前向/反向（caption_seq=None）+ pipeline + DDIM。

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


def test_fit_bucket():
    from data.dataset import HandwrittenFormulaDataset, bucket_collate
    from torch.utils.data import DataLoader

    tmp = tempfile.mkdtemp(prefix="fit_ds_")
    try:
        os.makedirs(os.path.join(tmp, "print"))
        os.makedirs(os.path.join(tmp, "style1"))
        # 多种宽高比
        make_image(os.path.join(tmp, "print", "square.png"), 256, 256)
        make_image(os.path.join(tmp, "print", "wide.png"), 512, 128)
        make_image(os.path.join(tmp, "print", "narrow.png"), 128, 512)
        make_image(os.path.join(tmp, "print", "tiny.png"), 300, 20)   # 应被 min_height 过滤
        for i in range(6):
            make_image(os.path.join(tmp, "style1", f"s{i}.png"), 300, 128)

        ds = HandwrittenFormulaDataset(
            data_root=tmp,
            max_patch=192,
            patch_size=16,
            multiple=8,
            min_height=64,
            batch_size=2,
            drop_last=True,
            seed=42,
            repeats_per_image=1,
            styles_per_repeat=1,
            style_as_tensor=True,
            vae_f=8,
            use_caption=False,
        )
        assert len(ds.print_images) == 3, f"expected 3 kept, got {len(ds.print_images)}"
        assert "tiny.png" not in ds.print_images, "tiny.png should be discarded by min_height"
        print(f"[1] FiT filtering: {len(ds.print_images)} images kept, tiny discarded OK")

        # 验证 build_batches 每个 batch 内尺寸一致
        batches = ds.build_batches(epoch=0)
        for batch in batches:
            sizes = {ds.bucket_of_sample[i] for i in batch}
            assert len(sizes) == 1, f"mixed sizes in batch: {sizes}"
        print(f"[1] FiT batches: {len(batches)} batches, all uniform size OK")

        # 用 DataLoader 走一遍
        from data.dataset import FitDistributedSampler
        sampler = FitDistributedSampler(ds, batch_size=2, num_replicas=1, rank=0, seed=42)
        loader = DataLoader(ds, batch_size=None, sampler=sampler, collate_fn=bucket_collate)
        seen_sizes = set()
        for I_p, I_s, I_t, buckets, _, _ in loader:
            h, w = I_p.shape[2], I_p.shape[3]
            seen_sizes.add((h, w))
            assert I_p.shape == I_t.shape
            assert buckets[:, 0].tolist() == [h] * I_p.shape[0]
            assert buckets[:, 1].tolist() == [w] * I_p.shape[0]
            break  # 只验证一个 batch
        print(f"[1] FiT DataLoader: sampled sizes {seen_sizes} OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_caption_on():
    from diffusion.ddim import ddim_sample
    pipe = _build_pipe(use_caption=True)

    B, L = 2, 6
    # 固定 128x256 -> latent 16x32，与动态分辨率无关
    z = torch.randn(B, 8, 16, 32, device=DEV)
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

    B, L = 2, 6
    z = torch.randn(B, 8, 16, 32, device=DEV)
    f_seq = torch.randn(B, 4, 192, device=DEV)
    f_pooled = torch.randn(B, 192, device=DEV)
    t = torch.randint(0, 100, (B,), device=DEV)
    out = pipe.dit(z, f_seq, f_pooled, None, None, t)
    assert out.shape == (B, 4, 16, 32)
    out.mean().backward()
    pipe.zero_grad()

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
    test_fit_bucket()
    test_caption_on()
    test_caption_off()
    print("\nALL FIT + CAPTION-SWITCH SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
