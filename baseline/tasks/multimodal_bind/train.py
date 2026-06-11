#!/usr/bin/env python3
"""CTM-Bind: multimodal contrastive training with CTM encoders.

Supports modality pairs:
  image_text, image_thermal, image_depth, video_text, video_imu

Usage:
    python -m baseline.tasks.multimodal_bind.train \
        --modality_pair image_text --dataset flickr30k \
        --hidden_size 512 --d_model 256 --iterations 10 \
        --max_steps 500 --batch_size 4 --device 0
"""

import argparse
import gc
import json
import os
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from model.config import CTMLLMConfig
from model.model_ctm_llm import CTMBlock
from model.building_blocks import RMSNorm
from baseline.utils.housekeeping import set_seed


# ── Model components ──────────────────────────────────────────────────

class PatchEncoder(nn.Module):
    def __init__(self, ctm_cfg, in_channels, image_size, patch_size, num_hidden_layers=None):
        super().__init__()
        num_layers = num_hidden_layers or ctm_cfg.num_hidden_layers
        self.patch_embed = nn.Conv2d(
            in_channels, ctm_cfg.hidden_size,
            kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, ctm_cfg.hidden_size))
        self.blocks = nn.ModuleList(
            [CTMBlock(i, ctm_cfg) for i in range(num_layers)])
        self.norm = RMSNorm(ctm_cfg.hidden_size)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        h = self.patch_embed(x)
        h = h.flatten(2).transpose(1, 2)
        h = h + self.pos_embed
        for block in self.blocks:
            out = block(h)
            h = out.hidden if hasattr(out, 'hidden') else out[0]
        return self.norm(h)


class FlatEncoder(nn.Module):
    def __init__(self, ctm_cfg, in_features, num_tokens, num_hidden_layers=None):
        super().__init__()
        num_layers = num_hidden_layers or ctm_cfg.num_hidden_layers
        self.proj = nn.Linear(in_features, ctm_cfg.hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, ctm_cfg.hidden_size))
        self.blocks = nn.ModuleList(
            [CTMBlock(i, ctm_cfg) for i in range(num_layers)])
        self.norm = RMSNorm(ctm_cfg.hidden_size)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1).transpose(1, 2)
        h = self.proj(x) + self.pos_embed
        for block in self.blocks:
            out = block(h)
            h = out.hidden if hasattr(out, 'hidden') else out[0]
        return self.norm(h)


class CTMBindModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        ctm_cfg = CTMLLMConfig(
            hidden_size=args.hidden_size,
            d_model=args.d_model,
            d_input=args.d_input,
            heads=args.heads,
            num_hidden_layers=args.num_hidden_layers,
            iterations=args.iterations,
            memory_length=args.memory_length,
            memory_hidden_dims=args.memory_hidden_dims,
            deep_nlms=bool(args.deep_nlms),
            n_synch_out=args.n_synch_out,
            n_synch_action=args.n_synch_action,
            synapse_depth=args.synapse_depth,
            self_cond=bool(args.self_cond),
            cross_layer_state=bool(args.cross_layer_state),
            neuron_select_type=getattr(args, 'neuron_select_type', 'random-pairing'),
            cell_sparsity_mode=args.cell_sparsity_mode,
            cell_topk=args.cell_topk,
            cell_sparsity_rescale=bool(args.cell_sparsity_rescale),
            tick_loss_mode=args.tick_loss_mode,
            dropout=args.dropout,
            vocab_size=6400,
        )
        self.ctm_cfg = ctm_cfg
        self.modality_pair = args.modality_pair
        self.embed_dim = args.embed_dim
        self.pool_mode = args.pool_mode
        self.temperature = args.temperature
        self.image_size = args.image_size
        self.patch_size = args.patch_size

        pair = args.modality_pair
        enc_a, enc_b, proj_a, proj_b = self._build_pair(pair, ctm_cfg, args)
        self.enc_a = enc_a
        self.enc_b = enc_b
        self.proj_a = proj_a
        self.proj_b = proj_b

    def _build_pair(self, pair, ctm_cfg, args):
        hs = ctm_cfg.hidden_size
        ed = args.embed_dim
        img_sz = args.image_size
        ps = args.patch_size

        if pair == "image_text":
            enc_a = PatchEncoder(ctm_cfg, 3, img_sz, ps)
            enc_b = FlatEncoder(ctm_cfg, hs, args.max_seq_len)
            return enc_a, enc_b, nn.Linear(hs, ed), nn.Linear(hs, ed)

        elif pair in ("image_thermal", "image_depth"):
            enc_a = PatchEncoder(ctm_cfg, 3, img_sz, ps)
            enc_b = PatchEncoder(ctm_cfg, 1, img_sz, ps)
            return enc_a, enc_b, nn.Linear(hs, ed), nn.Linear(hs, ed)

        elif pair == "video_text":
            nf = getattr(args, 'num_video_frames', 4)
            num_spatial = (img_sz // ps) ** 2
            vid_tokens = nf * num_spatial
            enc_a = PatchEncoder(ctm_cfg, 3, img_sz, ps)
            enc_a.pos_embed = nn.Parameter(torch.zeros(1, vid_tokens, hs))
            nn.init.trunc_normal_(enc_a.pos_embed, std=0.02)
            enc_b = FlatEncoder(ctm_cfg, hs, args.max_seq_len)
            return enc_a, enc_b, nn.Linear(hs, ed), nn.Linear(hs, ed)

        elif pair == "video_imu":
            nf = getattr(args, 'num_video_frames', 4)
            num_spatial = (img_sz // ps) ** 2
            vid_tokens = nf * num_spatial
            enc_a = PatchEncoder(ctm_cfg, 3, img_sz, ps)
            enc_a.pos_embed = nn.Parameter(torch.zeros(1, vid_tokens, hs))
            nn.init.trunc_normal_(enc_a.pos_embed, std=0.02)
            imu_kernel = 4
            imu_seq_len = getattr(args, 'imu_seq_len', 250)
            imu_tokens = imu_seq_len // imu_kernel
            self.imu_conv = nn.Conv1d(6, hs, kernel_size=imu_kernel, stride=imu_kernel)
            self.imu_pos = nn.Parameter(torch.zeros(1, imu_tokens, hs))
            nn.init.trunc_normal_(self.imu_pos, std=0.02)
            enc_b = FlatEncoder(ctm_cfg, hs, imu_tokens)
            return enc_a, enc_b, nn.Linear(hs, ed), nn.Linear(hs, ed)

        else:
            raise ValueError(f"Unknown modality_pair: {pair}")

    def pool(self, h):
        if self.pool_mode == "last_tick":
            return h[:, -1]
        elif self.pool_mode == "mean":
            return h.mean(dim=1)
        else:
            return h[:, -1]

    def encode_a(self, x):
        return self.enc_a(x)

    def encode_b(self, x):
        if self.modality_pair == "video_imu" and x.dim() == 3 and x.shape[1] == 6:
            x = self.imu_conv(x).transpose(1, 2) + self.imu_pos
        return self.enc_b(x)

    def forward(self, x_a, x_b):
        h_a = self.encode_a(x_a)
        h_b = self.encode_b(x_b)
        q = self.proj_a(self.pool(h_a))
        k = self.proj_b(self.pool(h_b))
        return q, k

    def contrastive_loss(self, q, k):
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        logits = torch.matmul(q, k.T) / self.temperature
        labels = torch.arange(q.shape[0], device=q.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2


# ── Datasets ──────────────────────────────────────────────────────────

class Flickr30kDataset(Dataset):
    def __init__(self, data_root, split="train", image_size=128, max_text_len=32):
        import pandas as pd
        from PIL import Image
        from torchvision import transforms
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        csv_path = os.path.join(data_root, "flickr30k", "flickr_annotations_30k.csv")
        img_dir = os.path.join(data_root, "flickr30k", "flickr30k-images")
        if not os.path.exists(csv_path):
            for candidate in [
                os.path.join(data_root, "flickr_annotations_30k.csv"),
                "dataset/flickr30k/flickr_annotations_30k.csv",
            ]:
                if os.path.exists(candidate):
                    csv_path = candidate
                    break
        if not os.path.exists(img_dir):
            for candidate in [
                os.path.join(data_root, "flickr30k-images"),
                "dataset/flickr30k/flickr30k-images",
            ]:
                if os.path.exists(candidate):
                    img_dir = candidate
                    break

        self.df = pd.read_csv(csv_path)
        if "split" in self.df.columns:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.img_dir = img_dir
        self.max_text_len = max_text_len
        self.hidden_size_placeholder = 512

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import json as _json
        from PIL import Image
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(img)
        except Exception:
            img_tensor = torch.zeros(3, self.image_size, self.image_size)
        captions = row.get("raw", row.get("caption", ""))
        if isinstance(captions, str):
            try:
                captions = _json.loads(captions)
            except Exception:
                captions = [captions]
        if isinstance(captions, list):
            caption = captions[0] if captions else ""
        else:
            caption = str(captions)
        text_tensor = torch.zeros(self.max_text_len, self.hidden_size_placeholder)
        return img_tensor, text_tensor, caption


class TartanRGBTDataset(Dataset):
    PAIR_DIRS = {"thermal": "thermal_right_rect_8", "depth": "stereo_depth"}

    def __init__(self, data_root, modality="thermal", image_size=128, max_samples=None):
        from torchvision import transforms
        self.image_size = image_size
        self.transform_rgb = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.transform_other = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.modality = modality
        self.image_pairs = []
        base = os.path.join(data_root, "TartanRGBT") if os.path.exists(os.path.join(data_root, "TartanRGBT")) else data_root
        other_name = self.PAIR_DIRS.get(modality, "thermal_right_rect_8")
        for day_dir in sorted(os.listdir(base)):
            day_path = os.path.join(base, day_dir)
            if not os.path.isdir(day_path) or day_dir.startswith("."):
                continue
            for traj_dir in sorted(os.listdir(day_path)):
                traj_path = os.path.join(day_path, traj_dir)
                if not os.path.isdir(traj_path):
                    continue
                self._collect_pairs(traj_path, "rgb_in_thermal", other_name)
        if max_samples:
            self.image_pairs = self.image_pairs[:max_samples]

    @staticmethod
    def _list_images(path, name):
        dir_path = os.path.join(path, name)
        zip_path = dir_path + ".zip"
        import zipfile
        entries = []
        if os.path.isdir(dir_path):
            for f in sorted(os.listdir(dir_path)):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    entries.append(("dir", os.path.join(dir_path, f)))
        if os.path.isfile(zip_path):
            try:
                with zipfile.ZipFile(zip_path) as z:
                    names = sorted(z.namelist())
            except Exception:
                names = []
            for n in names:
                if os.path.basename(n).lower().endswith((".png", ".jpg", ".jpeg")):
                    entries.append(("zip", zip_path, n))
        return entries

    @staticmethod
    def _frame_key(name):
        base = os.path.basename(name)
        no_ext = os.path.splitext(base)[0]
        digits = ""
        for ch in no_ext:
            if ch.isdigit() or (not digits and ch == "0"):
                if ch.isdigit():
                    digits += ch
            elif digits:
                break
        return digits.zfill(8) if digits else no_ext

    def _collect_pairs(self, traj_path, rgb_name, other_name):
        rgb_entries = self._list_images(traj_path, rgb_name)
        other_entries = self._list_images(traj_path, other_name)
        if not rgb_entries or not other_entries:
            return
        def _key(entry):
            return self._frame_key(entry[-1])
        rgb_map = {_key(e): e for e in rgb_entries}
        other_map = {_key(e): e for e in other_entries}
        common = sorted(set(rgb_map) & set(other_map))
        for key in common:
            self.image_pairs.append((rgb_map[key], other_map[key]))

    def __len__(self):
        return len(self.image_pairs)

    @staticmethod
    def _open_image(entry, mode="RGB"):
        from PIL import Image
        if entry[0] == "dir":
            return Image.open(entry[1]).convert(mode)
        import io, zipfile
        with zipfile.ZipFile(entry[1]) as z:
            return Image.open(io.BytesIO(z.read(entry[2]))).convert(mode)

    def __getitem__(self, idx):
        rgb_entry, other_entry = self.image_pairs[idx]
        try:
            rgb_tensor = self.transform_rgb(self._open_image(rgb_entry, "RGB"))
        except Exception:
            rgb_tensor = torch.zeros(3, self.image_size, self.image_size)
        try:
            other_tensor = self.transform_other(self._open_image(other_entry, "L"))
        except Exception:
            other_tensor = torch.zeros(1, self.image_size, self.image_size)
        return rgb_tensor, other_tensor


class MSRVideoTextDataset(Dataset):
    def __init__(self, data_root, split="train", image_size=128, num_frames=4, max_text_len=32):
        import json as _json
        from torchvision import transforms
        self.image_size = image_size
        self.num_frames = num_frames
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.max_text_len = max_text_len
        self.hidden_size_placeholder = 512
        base = os.path.join(data_root, "MSR-VTT") if os.path.exists(os.path.join(data_root, "MSR-VTT")) else data_root
        json_name = "msrvtt_train_9k.json" if split == "train" else "msrvtt_test_1k.json"
        json_path = os.path.join(base, json_name)
        with open(json_path) as f:
            data = _json.load(f)
        self.samples = []
        video_dir = os.path.join(base, "videos")
        if not os.path.isdir(video_dir):
            video_dir = os.path.join(base, "MSRVTT_Videos")
            if not os.path.isdir(video_dir):
                video_dir = base
        for item in data:
            vid_id = item.get("video_id", "")
            caption = ""
            if "caption" in item:
                c = item["caption"]
                caption = c[0] if isinstance(c, list) else str(c)
            elif "sentence" in item:
                caption = str(item["sentence"])
            vid_path = os.path.join(video_dir, f"{vid_id}.mp4")
            if not os.path.exists(vid_path):
                for ext in [".mp4", ".avi", ".mov"]:
                    alt = os.path.join(video_dir, f"{vid_id}{ext}")
                    if os.path.exists(alt):
                        vid_path = alt
                        break
            self.samples.append((vid_path, caption))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid_path, caption = self.samples[idx]
        frames = self._load_video_frames(vid_path)
        text_tensor = torch.zeros(self.max_text_len, self.hidden_size_placeholder)
        return frames, text_tensor, caption

    def _load_video_frames(self, path):
        from PIL import Image
        C, H, W = 3, self.image_size, self.image_size
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return torch.zeros(self.num_frames, C, H, W)
            indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
            frames = []
            for fi in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame)
                    frames.append(self.transform(pil_img))
                else:
                    frames.append(torch.zeros(C, H, W))
            cap.release()
        except Exception:
            frames = [torch.zeros(C, H, W) for _ in range(self.num_frames)]
        return torch.stack(frames)


class MCAPHousingDataset(Dataset):
    def __init__(self, data_root, image_size=128, imu_seq_len=250, num_video_frames=4):
        from torchvision import transforms
        self.image_size = image_size
        self.num_video_frames = num_video_frames
        self.imu_seq_len = imu_seq_len
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.samples = []
        base = os.path.join(data_root, "MCAP-Housing") if os.path.exists(os.path.join(data_root, "MCAP-Housing")) else data_root
        for fname in sorted(os.listdir(base)):
            if fname.endswith(".mcap"):
                self.samples.append(os.path.join(base, fname))

    def __len__(self):
        return max(len(self.samples), 1)

    def __getitem__(self, idx):
        mcap_path = self.samples[idx % len(self.samples)]
        frames, imu = self._load_mcap(mcap_path)
        return frames, imu

    def _load_mcap(self, path):
        C, H, W = 3, self.image_size, self.image_size
        frames = torch.zeros(self.num_video_frames, C, H, W)
        imu = torch.zeros(6, self.imu_seq_len)
        try:
            from mcap.reader import make_reader
            from mcap_ros2.decoder import DecoderFactory
            from PIL import Image as PILImage
            import io
            rgb_frames = []
            imu_data = []
            with open(path, "rb") as f:
                reader = make_reader(f, decoder_factories=[DecoderFactory()])
                for schema, channel, message, decoded in reader.iter_decoded_messages():
                    if channel.topic == "/camera/rgb/compressed" and len(rgb_frames) < self.num_video_frames:
                        try:
                            img = PILImage.open(io.BytesIO(decoded.data)).convert("RGB")
                            rgb_frames.append(self.transform(img))
                        except Exception:
                            pass
                    elif channel.topic == "/imu":
                        if len(imu_data) < self.imu_seq_len:
                            acc = [decoded.linear_acceleration.x, decoded.linear_acceleration.y, decoded.linear_acceleration.z]
                            gyro = [decoded.angular_velocity.x, decoded.angular_velocity.y, decoded.angular_velocity.z]
                            imu_data.append(acc + gyro)
            if rgb_frames:
                indices = np.linspace(0, len(rgb_frames) - 1, self.num_video_frames, dtype=int)
                frames = torch.stack([rgb_frames[i] for i in indices])
            if imu_data:
                arr = np.array(imu_data[:self.imu_seq_len]).T
                imu = torch.from_numpy(arr).float()
                if imu.shape[1] < self.imu_seq_len:
                    pad = torch.zeros(6, self.imu_seq_len - imu.shape[1])
                    imu = torch.cat([imu, pad], dim=1)
                imu = imu[:, :self.imu_seq_len]
        except Exception:
            pass
        return frames, imu


class SyntheticPairDataset(Dataset):
    def __init__(self, modality_pair, image_size=128, patch_size=32, length=100,
                 hidden_size=512, max_seq_len=32, num_video_frames=4, imu_seq_len=250):
        self.modality_pair = modality_pair
        self.image_size = image_size
        self.patch_size = patch_size
        self.length = length
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.num_video_frames = num_video_frames
        self.imu_seq_len = imu_seq_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        pair = self.modality_pair
        img_sz = self.image_size
        if pair == "image_text":
            img = torch.randn(3, img_sz, img_sz)
            txt = torch.randn(self.max_seq_len, self.hidden_size)
            return img, txt
        elif pair in ("image_thermal", "image_depth"):
            rgb = torch.randn(3, img_sz, img_sz)
            other = torch.randn(1, img_sz, img_sz)
            return rgb, other
        elif pair == "video_text":
            nf = self.num_video_frames
            vid = torch.randn(nf, 3, img_sz, img_sz)
            txt = torch.randn(self.max_seq_len, self.hidden_size)
            return vid, txt
        elif pair == "video_imu":
            nf = self.num_video_frames
            vid = torch.randn(nf, 3, img_sz, img_sz)
            imu = torch.randn(6, self.imu_seq_len)
            return vid, imu
        else:
            raise ValueError(f"Unknown pair: {pair}")


# ── Data loading helpers ──────────────────────────────────────────────

def _resolve_data_root(data_root, dataset_name):
    candidates = [data_root]
    if data_root and not os.path.isabs(data_root):
        candidates.append(os.path.abspath(data_root))
    for root in candidates:
        if root and os.path.isdir(root):
            return root
    return data_root


def get_dataset(args):
    pair = args.modality_pair
    dataset_name = args.dataset
    data_root = getattr(args, 'data_root', 'dataset')
    data_root = _resolve_data_root(data_root, dataset_name)

    if dataset_name == "flickr30k":
        ds = Flickr30kDataset(
            data_root, image_size=args.image_size,
            max_text_len=getattr(args, 'max_seq_len', 32))
    elif dataset_name == "tartanrgbt":
        modality = "thermal" if pair == "image_thermal" else "depth"
        ds = TartanRGBTDataset(
            data_root, modality=modality, image_size=args.image_size)
    elif dataset_name == "msrvtt":
        ds = MSRVideoTextDataset(
            data_root, image_size=args.image_size,
            num_frames=getattr(args, 'num_video_frames', 4),
            max_text_len=getattr(args, 'max_seq_len', 32))
    elif dataset_name == "mcap_housing":
        ds = MCAPHousingDataset(
            data_root, image_size=args.image_size,
            imu_seq_len=getattr(args, 'imu_seq_len', 250),
            num_video_frames=getattr(args, 'num_video_frames', 4))
    else:
        print(f"[WARN] Unknown dataset '{dataset_name}', using synthetic data")
        ds = SyntheticPairDataset(
            pair, image_size=args.image_size,
            patch_size=args.patch_size, length=1000,
            hidden_size=args.hidden_size,
            max_seq_len=getattr(args, 'max_seq_len', 32),
            num_video_frames=getattr(args, 'num_video_frames', 4),
            imu_seq_len=getattr(args, 'imu_seq_len', 250))
    return ds


def collate_fn(batch, modality_pair):
    if modality_pair in ("image_text", "video_text"):
        x_a = torch.stack([b[0] for b in batch])
        x_b = torch.stack([b[1] for b in batch])
        return x_a, x_b
    elif modality_pair in ("image_thermal", "image_depth"):
        x_a = torch.stack([b[0] for b in batch])
        x_b = torch.stack([b[1] for b in batch])
        return x_a, x_b
    elif modality_pair == "video_imu":
        x_a = torch.stack([b[0] for b in batch])
        x_b = torch.stack([b[1] for b in batch])
        return x_a, x_b
    else:
        x_a = torch.stack([b[0] for b in batch])
        x_b = torch.stack([b[1] for b in batch])
        return x_a, x_b


# ── Video encoder helper ──────────────────────────────────────────────

def encode_video(model, frames):
    B, nf, C, H, W = frames.shape
    frames_flat = frames.reshape(B * nf, C, H, W)
    h = model.enc_a.patch_embed(frames_flat)
    h = h.flatten(2).transpose(1, 2)
    num_spatial = h.shape[1]
    h = h.reshape(B, nf * num_spatial, -1) + model.enc_a.pos_embed
    for block in model.enc_a.blocks:
        out = block(h)
        h = out.hidden if hasattr(out, 'hidden') else out[0]
    h = model.enc_a.norm(h)
    return h


# ── Training ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="CTM-Bind multimodal contrastive training")

    parser.add_argument('--train_module', type=str, default='')
    parser.add_argument('--model_type', type=str, default='ctm_bind')
    parser.add_argument('--modality_pair', type=str, default='image_text',
                        choices=['image_text', 'image_thermal', 'image_depth',
                                 'video_text', 'video_imu'])
    parser.add_argument('--dataset', type=str, default='flickr30k')
    parser.add_argument('--data_root', type=str, default='dataset')

    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--d_input', type=int, default=128)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--num_hidden_layers', type=int, default=6)
    parser.add_argument('--n_synch_out', type=int, default=256)
    parser.add_argument('--n_synch_action', type=int, default=256)
    parser.add_argument('--synapse_depth', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--memory_length', type=int, default=8)
    parser.add_argument('--memory_hidden_dims', type=int, default=2)
    parser.add_argument('--deep_nlms', type=int, default=1)
    parser.add_argument('--self_cond', type=int, default=1)
    parser.add_argument('--cross_layer_state', type=int, default=1)
    parser.add_argument('--neuron_select_type', type=str, default='random-pairing')
    parser.add_argument('--cell_sparsity_mode', type=str, default='none')
    parser.add_argument('--cell_topk', type=int, default=256)
    parser.add_argument('--cell_sparsity_rescale', type=int, default=1)
    parser.add_argument('--tick_loss_mode', type=str, default='last')
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--pool_mode', type=str, default='last_tick',
                        choices=['last_tick', 'mean'])
    parser.add_argument('--max_seq_len', type=int, default=196)
    parser.add_argument('--num_video_frames', type=int, default=4)
    parser.add_argument('--imu_seq_len', type=int, default=250)

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--use_scheduler', type=int, default=1)
    parser.add_argument('--scheduler_type', type=str, default='cosine')
    parser.add_argument('--save_interval', type=int, default=200)
    parser.add_argument('--log_interval', type=int, default=10)

    parser.add_argument('--dtype', type=str, default='bfloat16')
    parser.add_argument('--device', type=int, nargs='+', default=[0])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_dir', type=str, default='logs/ctm_bind')
    parser.add_argument('--experiment_name', type=str, default='')
    parser.add_argument('--swanlab_name', type=str, default='')
    parser.add_argument('--save_weight', type=str, default='')
    parser.add_argument('--freeze_modality', type=str, default=None)

    args = parser.parse_args()
    return args


def freeze_encoder(encoder):
    for p in encoder.parameters():
        p.requires_grad = False


def train(args):
    set_seed(args.seed)

    if args.device[0] >= 0:
        device = f'cuda:{args.device[0]}'
    else:
        device = 'cpu'
    torch_dtype = getattr(torch, args.dtype, torch.bfloat16)

    print(f"CTM-Bind: pair={args.modality_pair} dataset={args.dataset}")
    print(f"  hidden_size={args.hidden_size} d_model={args.d_model} iterations={args.iterations}")
    print(f"  embed_dim={args.embed_dim} temperature={args.temperature} pool={args.pool_mode}")
    print(f"  device={device} dtype={args.dtype}")

    os.makedirs(args.log_dir, exist_ok=True)
    with open(os.path.join(args.log_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)

    model = CTMBindModel(args)
    if torch_dtype == torch.bfloat16:
        model = model.to(torch_dtype)
    model = model.to(device)

    if args.freeze_modality == "image" or args.freeze_modality == "video":
        freeze_encoder(model.enc_a)
        print(f"  Frozen encoder A ({args.freeze_modality})")
    elif args.freeze_modality == "text":
        freeze_encoder(model.enc_b)
        print(f"  Frozen encoder B (text)")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  total_params={total_params:,} trainable={trainable_params:,}")

    dataset = get_dataset(args)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, drop_last=True,
        collate_fn=lambda batch: collate_fn(batch, args.modality_pair))

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)

    if args.use_scheduler and args.scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_steps, eta_min=args.lr * 0.01)
    else:
        scheduler = None

    global_step = 0
    running_loss = 0.0
    is_video = args.modality_pair in ("video_text", "video_imu")
    is_video_imu = args.modality_pair == "video_imu"

    model.train()
    pbar = tqdm(total=args.max_steps, desc="training")
    start_time = time.time()

    for epoch in range(args.epochs):
        for batch in loader:
            if global_step >= args.max_steps:
                break

            x_a, x_b = batch
            x_a = x_a.to(device=device, dtype=torch_dtype)
            x_b = x_b.to(device=device, dtype=torch_dtype)

            ctx = torch.autocast(device_type='cuda', dtype=torch_dtype) if torch_dtype == torch.bfloat16 and device != 'cpu' else nullcontext()

            with ctx:
                if is_video:
                    h_a = encode_video(model, x_a)
                    h_b = model.encode_b(x_b)
                else:
                    h_a = model.encode_a(x_a)
                    h_b = model.encode_b(x_b)
                q = model.proj_a(model.pool(h_a))
                k = model.proj_b(model.pool(h_b))
                loss = model.contrastive_loss(q, k) / args.accumulation_steps

            loss.backward()

            if (global_step + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler:
                    scheduler.step()

            global_step += 1
            running_loss += loss.item() * args.accumulation_steps
            pbar.update(1)
            pbar.set_postfix(loss=f"{running_loss / global_step:.4f}", step=global_step)

            if global_step % args.log_interval == 0:
                avg_loss = running_loss / global_step
                elapsed = time.time() - start_time
                print(f"  step={global_step} loss={avg_loss:.4f} "
                      f"lr={optimizer.param_groups[0]['lr']:.2e} "
                      f"time={elapsed:.1f}s")

            if args.save_interval > 0 and global_step % args.save_interval == 0:
                ckpt_path = os.path.join(args.log_dir, f"ckpt_{global_step}.pt")
                torch.save(model.state_dict(), ckpt_path)

            if global_step >= args.max_steps:
                break

    pbar.close()

    final_loss = running_loss / max(global_step, 1)
    total_time = time.time() - start_time
    print(f"\nDone: steps={global_step} loss={final_loss:.4f} time={total_time:.1f}s")

    if args.save_weight:
        save_path = os.path.join(args.log_dir, f"{args.save_weight}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path}")

    metrics = {
        "experiment_name": args.experiment_name,
        "loss": final_loss,
        "contrastive_loss": final_loss,
        "global_step": global_step,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "elapsed_seconds": total_time,
        "modality_pair": args.modality_pair,
        "dataset": args.dataset,
    }
    metrics_dir = os.environ.get("CTM_METRICS_DIR", "runs/metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    if args.experiment_name:
        mpath = os.path.join(metrics_dir, f"{args.experiment_name}.json")
        with open(mpath, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics: {mpath}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
