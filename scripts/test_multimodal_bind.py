#!/usr/bin/env python3
"""Smoke tests for CTM-Bind multimodal contrastive model.

Verifies model creation, forward pass, backward pass, and finite loss
for every modality pair — runs locally without cluster or real data.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import CTMLLMConfig
from model.model_ctm_llm import CTMBlock, RMSNorm


def tiny_ctm_config(**overrides):
    cfg = dict(
        vocab_size=6400,
        hidden_size=128,
        d_model=64,
        d_input=32,
        heads=4,
        num_hidden_layers=1,
        iterations=2,
        memory_length=4,
        memory_hidden_dims=1,
        deep_nlms=True,
        n_synch_out=64,
        n_synch_action=64,
        synapse_depth=1,
        self_cond=True,
        cross_layer_state=False,
    )
    cfg.update(overrides)
    return CTMLLMConfig(**cfg)


class PatchEncoder(nn.Module):
    def __init__(self, ctm_cfg, in_channels, image_size, patch_size):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            in_channels, ctm_cfg.hidden_size,
            kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, ctm_cfg.hidden_size))
        self.blocks = nn.ModuleList(
            [CTMBlock(i, ctm_cfg) for i in range(ctm_cfg.num_hidden_layers)])
        self.norm = RMSNorm(ctm_cfg.hidden_size)

    def forward(self, x):
        h = self.patch_embed(x)
        h = h.flatten(2).transpose(1, 2)
        h = h + self.pos_embed
        for block in self.blocks:
            out = block(h)
            h = out.hidden if hasattr(out, 'hidden') else out[0]
        return self.norm(h)


class FlatEncoder(nn.Module):
    def __init__(self, ctm_cfg, in_features, num_tokens):
        super().__init__()
        self.proj = nn.Linear(in_features, ctm_cfg.hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, ctm_cfg.hidden_size))
        self.blocks = nn.ModuleList(
            [CTMBlock(i, ctm_cfg) for i in range(ctm_cfg.num_hidden_layers)])
        self.norm = RMSNorm(ctm_cfg.hidden_size)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1).transpose(1, 2)
        h = self.proj(x) + self.pos_embed
        for block in self.blocks:
            out = block(h)
            h = out.hidden if hasattr(out, 'hidden') else out[0]
        return self.norm(h)


EMBED_DIM = 64
IMAGE_SIZE = 64
PATCH_SIZE = 16
TEMPERATURE = 0.07


def _contrastive_loss(q, k, temperature=TEMPERATURE):
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    logits = torch.matmul(q, k.T) / temperature
    labels = torch.arange(q.shape[0], device=q.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2


def _make_pair(in_ch_a, in_ch_b, num_patches_a, num_patches_b):
    ctm_cfg = tiny_ctm_config()
    enc_a = PatchEncoder(ctm_cfg, in_ch_a, IMAGE_SIZE, PATCH_SIZE)
    enc_b = PatchEncoder(ctm_cfg, in_ch_b, IMAGE_SIZE, PATCH_SIZE)
    proj_a = nn.Linear(ctm_cfg.hidden_size, EMBED_DIM)
    proj_b = nn.Linear(ctm_cfg.hidden_size, EMBED_DIM)
    return enc_a, enc_b, proj_a, proj_b, ctm_cfg


def test_image_text():
    B = 4
    ctm_cfg = tiny_ctm_config()
    enc_img = PatchEncoder(ctm_cfg, 3, IMAGE_SIZE, PATCH_SIZE)
    enc_txt = FlatEncoder(ctm_cfg, in_features=ctm_cfg.hidden_size, num_tokens=16)
    proj_img = nn.Linear(ctm_cfg.hidden_size, EMBED_DIM)
    proj_txt = nn.Linear(ctm_cfg.hidden_size, EMBED_DIM)
    images = torch.randn(B, 3, IMAGE_SIZE, IMAGE_SIZE)
    text_seqs = torch.randn(B, 16, ctm_cfg.hidden_size)
    h_img = enc_img(images)
    h_txt = enc_txt(text_seqs)
    q = proj_img(h_img[:, -1])
    k = proj_txt(h_txt[:, -1])
    loss = _contrastive_loss(q, k)
    assert torch.isfinite(loss), f"image-text loss not finite: {loss.item()}"
    loss.backward()
    print(f"image_text: loss={loss.item():.4f}  OK")


def test_image_thermal():
    B = 4
    enc_rgb, enc_th, proj_rgb, proj_th, cfg = _make_pair(3, 1, 16, 16)
    rgb = torch.randn(B, 3, IMAGE_SIZE, IMAGE_SIZE)
    thermal = torch.randn(B, 1, IMAGE_SIZE, IMAGE_SIZE)
    h_rgb = enc_rgb(rgb)
    h_th = enc_th(thermal)
    loss = _contrastive_loss(proj_rgb(h_rgb[:, -1]), proj_th(h_th[:, -1]))
    assert torch.isfinite(loss)
    loss.backward()
    print(f"image_thermal: loss={loss.item():.4f}  OK")


def test_image_depth():
    B = 4
    enc_rgb, enc_dep, proj_rgb, proj_dep, cfg = _make_pair(3, 1, 16, 16)
    rgb = torch.randn(B, 3, IMAGE_SIZE, IMAGE_SIZE)
    depth = torch.randn(B, 1, IMAGE_SIZE, IMAGE_SIZE)
    h_rgb = enc_rgb(rgb)
    h_dep = enc_dep(depth)
    loss = _contrastive_loss(proj_rgb(h_rgb[:, -1]), proj_dep(h_dep[:, -1]))
    assert torch.isfinite(loss)
    loss.backward()
    print(f"image_depth: loss={loss.item():.4f}  OK")


def test_video_text():
    num_frames = 2
    B = 2
    num_spatial = (IMAGE_SIZE // PATCH_SIZE) ** 2
    vid_tokens = num_frames * num_spatial

    ctm_cfg = tiny_ctm_config()
    enc_vid = PatchEncoder(ctm_cfg, 3, IMAGE_SIZE, PATCH_SIZE)
    enc_vid.pos_embed = nn.Parameter(torch.zeros(1, vid_tokens, ctm_cfg.hidden_size))
    enc_txt = FlatEncoder(ctm_cfg, in_features=ctm_cfg.hidden_size, num_tokens=16)
    proj_vid = nn.Linear(ctm_cfg.hidden_size, EMBED_DIM)
    proj_txt = nn.Linear(ctm_cfg.hidden_size, EMBED_DIM)

    frames_flat = torch.randn(B * num_frames, 3, IMAGE_SIZE, IMAGE_SIZE)
    h_vid = enc_vid.patch_embed(frames_flat)
    h_vid = h_vid.flatten(2).transpose(1, 2)
    h_vid = h_vid.reshape(B, vid_tokens, -1) + enc_vid.pos_embed
    for block in enc_vid.blocks:
        out = block(h_vid)
        h_vid = out.hidden if hasattr(out, 'hidden') else out[0]
    h_vid = enc_vid.norm(h_vid)

    text_seqs = torch.randn(B, 16, ctm_cfg.hidden_size)
    h_txt = enc_txt(text_seqs)

    loss = _contrastive_loss(proj_vid(h_vid[:, -1]), proj_txt(h_txt[:, -1]))
    assert torch.isfinite(loss)
    loss.backward()
    print(f"video_text: loss={loss.item():.4f}  OK")


def test_video_imu():
    num_frames = 2
    B = 2
    num_spatial = (IMAGE_SIZE // PATCH_SIZE) ** 2
    vid_tokens = num_frames * num_spatial
    imu_kernel = 4
    imu_seq_len = 32
    imu_tokens = imu_seq_len // imu_kernel

    ctm_cfg = tiny_ctm_config()
    enc_vid = PatchEncoder(ctm_cfg, 3, IMAGE_SIZE, PATCH_SIZE)
    enc_vid.pos_embed = nn.Parameter(torch.zeros(1, vid_tokens, ctm_cfg.hidden_size))

    imu_conv = nn.Conv1d(6, ctm_cfg.hidden_size, kernel_size=imu_kernel, stride=imu_kernel)
    imu_pos = nn.Parameter(torch.zeros(1, imu_tokens, ctm_cfg.hidden_size))
    enc_imu = FlatEncoder(ctm_cfg, in_features=ctm_cfg.hidden_size, num_tokens=imu_tokens)

    proj_vid = nn.Linear(ctm_cfg.hidden_size, EMBED_DIM)
    proj_imu = nn.Linear(ctm_cfg.hidden_size, EMBED_DIM)

    frames_flat = torch.randn(B * num_frames, 3, IMAGE_SIZE, IMAGE_SIZE)
    h_vid = enc_vid.patch_embed(frames_flat).flatten(2).transpose(1, 2)
    h_vid = h_vid.reshape(B, vid_tokens, -1) + enc_vid.pos_embed
    for block in enc_vid.blocks:
        out = block(h_vid)
        h_vid = out.hidden if hasattr(out, 'hidden') else out[0]
    h_vid = enc_vid.norm(h_vid)

    imu_raw = torch.randn(B, 6, imu_seq_len)
    imu_feat = (imu_conv(imu_raw).transpose(1, 2) + imu_pos)
    h_imu = enc_imu(imu_feat)

    loss = _contrastive_loss(proj_vid(h_vid[:, -1]), proj_imu(h_imu[:, -1]))
    assert torch.isfinite(loss)
    loss.backward()
    print(f"video_imu: loss={loss.item():.4f}  OK")


def test_pooling_modes():
    B = 2
    ctm_cfg = tiny_ctm_config()
    enc = PatchEncoder(ctm_cfg, 3, IMAGE_SIZE, PATCH_SIZE)
    proj = nn.Linear(ctm_cfg.hidden_size, EMBED_DIM)
    images = torch.randn(B, 3, IMAGE_SIZE, IMAGE_SIZE)
    h = enc(images)
    for mode in ("last_tick", "mean"):
        pooled = h[:, -1] if mode == "last_tick" else h.mean(dim=1)
        assert pooled.shape == (B, ctm_cfg.hidden_size), f"pool {mode} shape mismatch"
        out = proj(pooled)
        assert out.shape == (B, EMBED_DIM)
    print("pooling_modes: last_tick + mean  OK")


def test_freeze_encoder():
    B = 2
    ctm_cfg = tiny_ctm_config()
    enc = PatchEncoder(ctm_cfg, 3, IMAGE_SIZE, PATCH_SIZE)
    proj = nn.Linear(ctm_cfg.hidden_size, EMBED_DIM)

    for p in enc.parameters():
        p.requires_grad = False

    frozen = sum(p.numel() for p in enc.parameters() if not p.requires_grad)
    total = sum(p.numel() for p in enc.parameters())
    assert frozen == total, f"freeze incomplete: {frozen}/{total}"

    images = torch.randn(B, 3, IMAGE_SIZE, IMAGE_SIZE)
    h = enc(images)
    out = proj(h[:, -1])
    loss = out.sum()
    loss.backward()

    for p in proj.parameters():
        assert p.grad is not None, "proj param has no grad"
    for p in enc.parameters():
        assert p.grad is None, "frozen param should have no grad"

    print(f"freeze_encoder: {frozen} params frozen, proj grad OK")


def test_gradient_accumulation():
    B = 4
    accum_steps = 4
    ctm_cfg = tiny_ctm_config()
    enc_img = PatchEncoder(ctm_cfg, 3, IMAGE_SIZE, PATCH_SIZE)
    enc_txt = PatchEncoder(ctm_cfg, 1, IMAGE_SIZE, PATCH_SIZE)
    proj_img = nn.Linear(ctm_cfg.hidden_size, EMBED_DIM)
    proj_txt = nn.Linear(ctm_cfg.hidden_size, EMBED_DIM)

    all_params = list(enc_img.parameters()) + list(enc_txt.parameters()) + \
                 list(proj_img.parameters()) + list(proj_txt.parameters())
    optimizer = torch.optim.Adam(all_params, lr=1e-3)

    for _ in range(accum_steps):
        images = torch.randn(B, 3, IMAGE_SIZE, IMAGE_SIZE)
        text_seqs = torch.randn(B, 1, IMAGE_SIZE, IMAGE_SIZE)
        h_img = enc_img(images)
        h_txt = enc_txt(text_seqs)
        q = proj_img(h_img[:, -1])
        k = proj_txt(h_txt[:, -1])
        loss = _contrastive_loss(q, k) / accum_steps
        loss.backward()

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    print(f"gradient_accumulation: {accum_steps} steps  OK")


def main():
    tests = [
        test_pooling_modes,
        test_freeze_encoder,
        test_image_text,
        test_image_thermal,
        test_image_depth,
        test_video_text,
        test_video_imu,
        test_gradient_accumulation,
    ]
    passed, failed = 0, 0
    for test in tests:
        name = test.__name__
        try:
            test()
            passed += 1
        except Exception as e:
            import traceback
            print(f"FAIL {name}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
