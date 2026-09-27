"""
可变分辨率（token 预算）+ caption 开关 端到端冒烟验证。

覆盖：
1. token 预算分桶：dataset 按 max_tokens 拟合尺寸 + 宽高比自动分箱，输出 token_mask
2. caption 开启（use_caption=True）：DiT 前向/反向 + pipeline + DDIM
3. caption 关闭（use_caption=False）：DiT 前向/反向（caption_seq=None）+ pipeline + DDIM
4. 可变分辨率：DiT 带 attn_mask 前向/反向 + 带 token_mask 的扩散损失

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


def test_token_budget_buckets():
    from data.dataset import HandwrittenFormulaDataset, bucket_collate
    from torch.utils.data import DataLoader, Subset

    tmp = tempfile.mkdtemp(prefix="ds_")
    try:
        os.makedirs(os.path.join(tmp, "print"))
        os.makedirs(os.path.join(tmp, "style1"))
        # 宽高比跨度很大：1:3 / 1:10 / 1:1 / 1:6
        make_image(os.path.join(tmp, "print", "a.png"), 384, 128)
        make_image(os.path.join(tmp, "print", "b.png"), 1280, 128)
        make_image(os.path.join(tmp, "print", "c.png"), 256, 256)
        make_image(os.path.join(tmp, "print", "d.png"), 768, 128)
        for i in range(3):
            make_image(os.path.join(tmp, "style1", f"s{i}.png"), 300, 128)

        MAX_TOKENS = 256
        ds = HandwrittenFormulaDataset(
            data_root=tmp, max_tokens=MAX_TOKENS, min_grid_h=5, num_aspect_bins=4,
            repeats_per_image=1, styles_per_repeat=1, style_as_tensor=True,
            vae_f=8, patch_size=2,
        )
        # 不再按 train_size 过滤：所有图都参与训练
        assert len(ds.print_images) == 4, ds.print_images

        # 每个 canvas 都必须满足 token 预算
        for ch, cw in ds.buckets_with_data():
            assert (ch // 16) * (cw // 16) <= MAX_TOKENS, (ch, cw)

        # 逐桶取 batch（与训练一致：batch 内同 canvas）
        bucket = ds.buckets_with_data()[0]
        loader = DataLoader(Subset(ds, ds.indices_for_bucket(bucket)),
                            batch_size=2, collate_fn=bucket_collate)
        I_p, I_s, I_t, buckets, cap_ids, cap_mask, token_mask = next(iter(loader))
        assert tuple(I_p.shape[2:]) == bucket
        gh, gw = bucket[0] // 16, bucket[1] // 16
        assert token_mask.shape == (I_p.shape[0], gh * gw), token_mask.shape
        # 每个样本至少有 1 个真实 token（否则 softmax 全 -inf → NaN）
        assert int((~token_mask).sum(dim=1).min()) >= 1
        # 内容区是矩形：mask 为 False 的位置数量 = 内容网格面积
        print(f"[1] token budget: bucket {bucket} ({gh}x{gw}={gh*gw} tokens), "
              f"token_mask {tuple(token_mask.shape)}, "
              f"real tokens/sample {[int(v) for v in (~token_mask).sum(dim=1)]} OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_variable_resolution_forward():
    """可变分辨率：DiT 带 attn_mask 前向/反向 + 带 token_mask 的扩散损失。"""
    from losses.diffusion_loss import simple_diffusion_loss

    pipe = _build_pipe(use_caption=True)

    B = 2
    # latent 16x32 → grid 8x16 = 128 tokens
    z = torch.randn(B, 8, 16, 32, device=DEV)
    grid_h, grid_w, N = 8, 16, 128
    attn_mask = torch.zeros(B, N, dtype=torch.bool, device=DEV)
    attn_mask[:, 100:] = True           # 后 28 个 token 是 padding
    f_seq = torch.randn(B, 4, 192, device=DEV)
    f_pooled = torch.randn(B, 192, device=DEV)
    cap_seq = torch.randn(B, 6, 192, device=DEV)
    cap_mask = torch.zeros(B, 6, dtype=torch.bool, device=DEV)
    t = torch.randint(0, 100, (B,), device=DEV)

    out = pipe.dit(z, f_seq, f_pooled, cap_seq, cap_mask, t, attn_mask=attn_mask)
    assert out.shape == (B, 4, 16, 32)
    assert torch.isfinite(out).all(), "带 attn_mask 的前向出现非有限值"
    # padding token 的输出被清零
    pad_pixels = out.reshape(B, 4, grid_h, 2, grid_w, 2).permute(0, 1, 2, 4, 3, 5)
    pad_pixels = pad_pixels.reshape(B, 4, grid_h * grid_w, 4)
    assert float(pad_pixels[:, :, 100:, :].abs().max()) == 0.0
    out.mean().backward()
    pipe.zero_grad()
    print("[4] variable resolution: DiT forward/backward with attn_mask OK")

    # 扩散损失只在真实 token 上统计
    pred = torch.randn(B, 4, 16, 32, device=DEV)
    tgt = torch.randn(B, 4, 16, 32, device=DEV)
    loss = simple_diffusion_loss(pred, tgt, token_mask=attn_mask)
    keep = (~attn_mask).view(B, 1, grid_h, grid_w).float()
    keep = keep.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3).expand(-1, 4, -1, -1)
    ref = ((pred - tgt) ** 2 * keep).sum() / keep.sum()
    assert torch.allclose(loss, ref, atol=1e-6), (loss.item(), ref.item())
    # 无 mask 时退化为普通 MSE
    assert torch.allclose(simple_diffusion_loss(pred, tgt),
                          torch.nn.functional.mse_loss(pred, tgt))
    print(f"[4] variable resolution: masked diffusion loss OK (loss={loss.item():.4f})")


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
    test_token_budget_buckets()
    test_caption_on()
    test_caption_off()
    test_variable_resolution_forward()
    print("\nALL VARIABLE-RESOLUTION + CAPTION-SWITCH SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
