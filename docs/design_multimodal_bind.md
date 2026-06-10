# CTM-Bind: CTM 自监督多模态绑定训练设计

## 1. 核心思想

**用 CTM 替代 ViT 作为通用编码器，以 ImageBind 的 image-anchored contrastive learning 方式将所有模态绑定到同一嵌入空间。**

CTM 的独特优势：
- **Tick-based 迭代推理**：不同模态天然需要不同的"思考深度"（IMU 少 tick、图像多 tick）
- **神经元级状态**：每个神经元维护独立 trace，天然适合异构模态融合
- **Synchronization 机制**：pairwise 神经元交互 + 指数衰减，替代 LayerNorm 做跨模态对齐
- **已有的 DINOProjectionHead**：现成的对比学习头，直接复用
- **Async Clock Bands**：不同模态以不同频率更新，完美匹配多模态异步特性

---

## 2. 模态-数据集映射

| 模态 | 数据集 | 数据格式 | CTM 输入方式 |
|------|--------|----------|-------------|
| **Image** | Flickr30k | ZIP 内 JPEG | Patch → Linear → hidden_size |
| **Text** | Flickr30k / MSR-VTT | CSV/JSON 文本 | Token Embedding（已有） |
| **Video** | MSR-VTT | ZIP 内 MP4 | 采样 N 帧 → 各帧 Patch → 帧间位置编码 |
| **Thermal** | TartanRGBT | ZIP 内 PNG (1ch) | 单通道 Patch → Linear → hidden_size |
| **Depth** | TartanRGBT / MCAP | ZIP 内 PNG / 16UC1 | 单通道 Patch → Linear → hidden_size |
| **IMU** | MCAP-Housing | ROS2 IMU msg | 6轴时序 → 1D Conv → hidden_size |
| **PointCloud** | MCAP-Housing | PointCloud2 | 点采样 → PointNet-like → hidden_size |
| **Pose** | MCAP / TartanRGBT | 6DoF transform | 7维向量 → Linear → hidden_size |

### Image-Anchored 绑定关系（跟 ImageBind 一致）

```
Image ←→ Text      (Flickr30k, 已有配对数据)
Image ←→ Thermal   (TartanRGBT, RGB-Thermal 同帧对齐)
Image ←→ Depth     (TartanRGBT + MCAP, RGB-Depth 对齐)
Video ←→ Text      (MSR-VTT, 视频字幕配对)
Video ←→ IMU       (MCAP-Housing, 时间对齐的 RGB+IMU)
```

通过 Image 锚点，所有模态 emergent 对齐：
- Thermal ← Image → Text → **Thermal-Text 零样本**
- Depth ← Image → Text → **Depth-Text 零样本**
- IMU ← Video(Image) → Text → **IMU-Text 零样本**

---

## 3. 架构设计

### 3.1 总体结构

```
┌─────────────────────────────────────────────────────────┐
│                    CTM-Bind Model                        │
│                                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │  Image   │ │  Text    │ │ Thermal  │ │   IMU    │   │
│  │ Encoder  │ │ Encoder  │ │ Encoder  │ │ Encoder  │   │
│  │ (CTM)    │ │ (CTM)    │ │ (CTM)    │ │ (CTM)    │   │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘   │
│       │            │            │            │          │
│       ▼            ▼            ▼            ▼          │
│  ┌─────────────────────────────────────────────────┐    │
│  │        Shared DINOProjectionHead (线性投影)      │    │
│  │        → L2 Normalize → d_dim 嵌入空间           │    │
│  └────────────────────────┬────────────────────────┘    │
│                           │                              │
│                    InfoNCE Loss                           │
│                (以 Image 为 anchor)                       │
└─────────────────────────────────────────────────────────┘
```

### 3.2 模态编码器统一接口

所有编码器共享 CTM 核心计算（tick loop + NLM + synchronization），区别仅在输入嵌入层：

```python
class ModalityEncoder(nn.Module):
    """通用 CTM 编码器，可处理任意模态"""
    def __init__(self, config, modality_embed: nn.Module):
        self.modality_embed = modality_embed   # 模态特定的输入投影
        self.blocks = nn.ModuleList([CTMBlock(i, config) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size)

    def forward(self, x, mask=None):
        h = self.modality_embed(x)          # (B, T, hidden_size)
        for block in self.blocks:
            h, *_ = block(h, mask=mask)
        return self.norm(h)                  # (B, T, hidden_size)
```

### 3.3 各模态的输入嵌入层

#### Image Patch Embedding
```python
class ImagePatchEmbed(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, hidden_size=768, image_size=224):
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(zeros(1, self.num_patches, hidden_size))

    def forward(self, x):  # (B, 3, 224, 224)
        x = self.proj(x)                       # (B, hidden, H', W')
        x = x.flatten(2).transpose(1, 2)       # (B, num_patches, hidden_size)
        return x + self.pos_embed
```

#### Thermal / Depth Embedding
```python
class SingleChannelPatchEmbed(nn.Module):
    def __init__(self, patch_size=16, hidden_size=768, image_size=224):
        self.proj = nn.Conv2d(1, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(zeros(1, self.num_patches, hidden_size))

    def forward(self, x):  # (B, 1, H, W)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x + self.pos_embed
```

#### Video Embedding (时空联合)
```python
class VideoPatchEmbed(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, hidden_size=768, num_frames=8, image_size=224):
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.num_spatial = (image_size // patch_size) ** 2
        self.temporal_embed = nn.Parameter(zeros(1, num_frames, hidden_size))
        self.spatial_embed = nn.Parameter(zeros(1, self.num_spatial, hidden_size))

    def forward(self, frames):  # (B, num_frames, 3, H, W)
        B, F, C, H, W = frames.shape
        frames = frames.reshape(B * F, C, H, W)
        tokens = self.proj(frames).flatten(2).transpose(1, 2)  # (B*F, num_spatial, hidden)
        tokens = tokens.reshape(B, F, -1, hidden)
        tokens = tokens + self.spatial_embed[None, :, :] + self.temporal_embed[:, :, None, :]
        return tokens.reshape(B, F * self.num_spatial, hidden)  # (B, F*S, hidden)
```

#### IMU Embedding (1D 时序)
```python
class IMUEmbed(nn.Module):
    def __init__(self, in_channels=6, hidden_size=768, kernel_size=8, seq_len=2000):
        self.conv = nn.Conv1d(in_channels, hidden_size, kernel_size=kernel_size, stride=kernel_size)
        self.num_tokens = seq_len // kernel_size  # 250 tokens
        self.pos_embed = nn.Parameter(zeros(1, self.num_tokens, hidden_size))

    def forward(self, x):  # (B, 6, 2000) — accel_xyz + gyro_xyz
        x = self.conv(x).transpose(1, 2)  # (B, 250, hidden)
        return x + self.pos_embed
```

### 3.4 CTM 编码器的关键适配

#### Tick 数量按模态调整

CTM 的 tick 机制允许不同模态使用不同的推理深度：

```python
modality_ticks = {
    'image':    30,   # 图像复杂度高，充分推理
    'text':     20,   # 文本已有序列结构，中等 tick
    'thermal':  15,   # 热成像信息密度较低
    'depth':    15,   # 深度图相对简单
    'video':    20,   # 视频帧间有时间冗余
    'imu':      10,   # IMU 信号简单，少量 tick 即可
}
```

实现方式：在 `CTMBlock.forward()` 中根据传入的 `max_ticks` 参数提前终止迭代。

#### Async Clock Band 用于多模态融合（进阶）

当需要将多种模态在同一 CTM 实例内融合时，使用 Async Clock Bands：

```
Band 0 (period=1):  RGB  —— 每个 tick 都更新
Band 1 (period=2):  Thermal / Depth —— 每 2 个 tick 更新
Band 2 (period=4):  IMU / Pose —— 每 4 个 tick 更新
```

这样 RGB 的快速变化和 IMU 的缓慢漂移在同一 tick loop 内自然解耦。

---

## 4. 训练方案

### 4.1 阶段一：Image-Text 对齐（基础锚点）

**数据**: Flickr30k (31k 图像 × 5 captions)
**目标**: 训练 Image Encoder + Text Encoder 的基础对比能力

```
Batch: [B images, B captions]
Image → CTMImageEncoder → mean_pool → DINOProj → L2 → q_i
Text  → CTMTextEncoder  → mean_pool → DINOProj → L2 → k_i
Loss  = InfoNCE(q_i, k_i, τ=0.07)  # 对称损失
```

**关键决策**：
- Image 和 Text 使用**独立的 CTM 编码器**（不共享权重），各自有独立的神经元状态
- DINOProjectionHead 也**独立**（每个模态一个），但投影到相同的维度
- 不冻结任何编码器（CTM 不像 ViT 有预训练权重，从头训练）
- 因为 Flickr30k 只有 31k 图像，需要大量 augmentation + **过采样 50x**（跟 ImageBind 对小数据集的做法一致）

### 4.2 阶段二：逐步绑定其他模态

每个新模态只需要与 Image 配对训练：

| 步骤 | 绑定 | 数据 | 冻结 |
|------|------|------|------|
| 2a | Image ←→ Thermal | TartanRGBT (RGB-Thermal 对齐帧) | 冻结 Image Encoder |
| 2b | Image ←→ Depth | TartanRGBT + MCAP (RGB-Depth 对齐帧) | 冻结 Image Encoder |
| 2c | Video ←→ Text | MSR-VTT (视频-字幕) | 冻结 Text Encoder |
| 2d | Video ←→ IMU | MCAP-Housing (时间对齐的 RGB+IMU) | 冻结 Video Encoder |

**每步只训练新模态的编码器**，Image/Text Encoder 的权重保持不变（作为锚点）。

### 4.3 阶段三（可选）：联合微调

解冻所有编码器，混合所有数据集，加入：
- **跨模态对比**：Thermal-Text, Depth-Text, IMU-Text（emergent 能力巩固）
- **Intra-modality 重建**：每个模态的重建 loss 保持表示质量
- **Tick diversity loss**：鼓励不同 tick 输出不同层面的特征

### 4.4 InfoNCE Loss 实现

```python
def info_nce_loss(q, k, temperature=0.07):
    """
    q: (B, D) — anchor 模态 (e.g., image)
    k: (B, D) — target 模态 (e.g., text)
    对称对比损失
    """
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    logits = torch.matmul(q, k.T) / temperature   # (B, B)
    labels = torch.arange(q.shape[0], device=q.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2
```

---

## 5. 数据加载

### 5.1 Flickr30k (Image-Text)

```python
class Flickr30kDataset(Dataset):
    def __init__(self, zip_path, csv_path, split='train', image_size=224):
        self.zip_path = zip_path
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['split'] == split]
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with ZipFile(self.zip_path) as zf:
            img_data = zf.read(f"flickr30k-images/{row['filename']}")
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        img = self.transform(img)
        captions = ast.literal_eval(row['raw'])
        cap = random.choice(captions)
        return img, cap
```

### 5.2 TartanRGBT (Image-Thermal / Image-Depth)

```python
class TartanRGBTDataset(Dataset):
    """从同一轨迹同一帧取 RGB + Thermal + Depth"""
    def __init__(self, base_dir, traj_filter=None, image_size=224):
        # 扫描所有轨迹，找同时有 zed_left + thermal_left_8 + stereo_depth 的轨迹
        self.pairs = []  # [(traj_dir, frame_idx), ...]
        for traj_dir in sorted(glob(os.path.join(base_dir, 'day*/*'))):
            rgb_zip = os.path.join(traj_dir, 'zed_left_rect.zip')
            thermal_zip = os.path.join(traj_dir, 'thermal_left_rect_8.zip')
            depth_zip = os.path.join(traj_dir, 'stereo_depth.zip')
            if os.path.exists(rgb_zip) and os.path.exists(thermal_zip):
                n_frames = len(ZipFile(rgb_zip).namelist())
                for i in range(n_frames):
                    self.pairs.append((traj_dir, i))

    def __getitem__(self, idx):
        traj_dir, frame_idx = self.pairs[idx]
        fname = f"{frame_idx:08d}.png"
        with ZipFile(os.path.join(traj_dir, 'zed_left_rect.zip')) as zf:
            rgb = self._load_image(zf, f"zed_left_rect/{fname}")
        with ZipFile(os.path.join(traj_dir, 'thermal_left_rect_8.zip')) as zf:
            thermal = self._load_thermal(zf, f"thermal_left_rect_8/{fname}")
        return rgb, thermal
```

### 5.3 MCAP-Housing (Video-IMU)

```python
class MCAPIMUDataset(Dataset):
    """从 MCAP 文件中抽取时间对齐的 RGB 帧 + IMU 片段"""
    def __init__(self, mcap_path, clip_len_sec=2.0, stride_sec=1.0, fps=10):
        self.mcap_path = mcap_path
        self.clips = []  # [(start_ns, end_ns), ...]
        # 预扫描 MCAP 生成 clip 列表
        self._scan_clips(clip_len_sec, stride_sec, fps)

    def __getitem__(self, idx):
        start_ns, end_ns = self.clips[idx]
        frames, imu_data = [], []
        with open(self.mcap_path, 'rb') as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for msg in reader.iter_decoded_messages():
                if start_ns <= msg.log_time_ns <= end_ns:
                    if msg.channel.topic == '/camera/rgb/compressed':
                        frames.append(self._decode_rgb(msg))
                    elif msg.channel.topic == '/imu':
                        imu_data.append(self._decode_imu(msg))
        return torch.stack(frames), torch.tensor(imu_data)
```

---

## 6. CTM 特有的训练技巧

### 6.1 Tick-Level 特征聚合

CTM 输出多个 tick 的隐藏状态。如何得到单一嵌入用于对比学习：

**方案 A：Last Tick**（最简单）
```python
embedding = tick_outputs[-1].mean(dim=1)  # 最后一个 tick 的 mean pool
```

**方案 B：Weighted Mean**（推荐）
```python
weights = F.softmax(self.tick_weights, dim=0)  # learnable (num_ticks,)
embedding = sum(w * out.mean(1) for w, out in zip(weights, tick_outputs))
```

**方案 C：Sync Output**（利用 synchronization 机制）
```python
# synchronization 输出本身就是神经元交互的紧凑表示
embedding = sync_output.mean(dim=1)
```

### 6.2 Neuron Sparsity 用于模态特化

对不同的模态编码器应用不同级别的 cell sparsity：

```python
# 图像编码器：低 sparsity，保留更多神经元（视觉信息丰富）
image_encoder.cell_topk = d_model  # 不做稀疏

# IMU 编码器：高 sparsity，只用关键神经元（信号简单）
imu_encoder.cell_topk = d_model // 4  # 只激活 25% 神经元
```

### 6.3 训练效率

CTM 的 30 tick × 12 层计算量巨大。建议：

1. **小模型起步**：`hidden_size=512, d_model=256, d_input=128, layers=6, ticks=15`
2. **Patch 大小增大**：32×32 patch（而非 16×16），减少 token 数量 4x
3. **图像分辨率降低**：128×128 起步，验证流程后再提升
4. **NLM recursive fast path**：开启 `residual_compute_mode`，避免每 tick 重算 NLM
5. **梯度累积**：模拟大 batch size（ImageBind 推荐 1024+）
6. **混合精度**：bf16 训练

---

## 7. 实现路线图

### Phase 1：最小可行原型 (3-5 天)

1. **`model/modality_embed.py`**：实现 ImagePatchEmbed + SingleChannelPatchEmbed
2. **`model/ctm_bind.py`**：CTMBindModel 核心类
   - ModalityEncoder（包装 CTMBlock + modality_embed）
   - 对比学习 loss（InfoNCE）
   - 池化策略（mean / weighted / last_tick）
3. **`dataset/flickr_dataset.py`**：Flickr30k Dataset
4. **`trainer/train_bind.py`**：训练循环（单模态对训练）
5. **验证**：在 Flickr30k image-text 检索上跑 baseline（R@1, R@5, R@10）

### Phase 2：多模态扩展 (3-5 天)

6. **`dataset/tartanrgbt_dataset.py`**：RGB-Thermal-Depth 对齐数据集
7. **`dataset/mcap_dataset.py`**：MCAP RGB-IMU 数据集
8. **`dataset/msrvtt_dataset.py`**：MSR-VTT Video-Text 数据集
9. 阶段式训练：冻结 image encoder → 训练 thermal/depth/IMU encoder
10. **验证**：零样本跨模态检索（Thermal→Text, Depth→Text）

### Phase 3：CTM 增强 (可选)

11. Async Clock Bands 多模态融合
12. Neuron sparsity 模态特化
13. Tick diversity 跨模态 loss
14. 联合微调所有模态

---

## 8. 与 ImageBind 的关键区别

| 方面 | ImageBind | CTM-Bind |
|------|-----------|----------|
| 骨干网络 | ViT (预训练 CLIP) | CTM (从头训练) |
| 特征提取 | 单次前向 | 多 tick 迭代推理 |
| 模态差异处理 | 不同大小的 ViT | 同一架构不同 tick 数 |
| 归一化 | LayerNorm | Synchronization (神经元对) |
| 对比头 | 线性投影 | DINOProjectionHead (MLP+WN) |
| 锚点 | 冻结的 CLIP | 训练的 CTM (阶段一学习) |
| 预训练需求 | 依赖 CLIP | 无外部预训练依赖 |
| 计算瓶颈 | Attention | NLM + Synapse per tick |

---

## 9. 预期风险与对策

| 风险 | 影响 | 对策 |
|------|------|------|
| CTM 计算量过大，训练太慢 | 无法完成实验 | 小模型 + 大 patch + 低分辨率 + recursive NLM |
| 31k 图像数据量不足，欠拟合 | 对比学习效果差 | 50x 过采样 + 强 augmentation + 温度调参 |
| 多模态数据不对齐 | 噪声标签 | 严格使用已对齐的数据对（RGB-Thermal 同帧） |
| CTM 从头训练收敛慢 | 阶段一耗时长 | 先用更小的 CTM (3 层 5 tick) 跑通流程 |
| 异构模态嵌入空间难对齐 | emergent 能力弱 | 增大 batch size（梯度累积）、增加训练 epoch |
