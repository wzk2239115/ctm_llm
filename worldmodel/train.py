"""Training loop for a JEPA world model.

Given a filled :class:`~worldmodel.data.ReplayBuffer`, sample action-conditioned
frame chunks and minimise the JEPA latent-prediction loss. An optional EMA
target encoder (``ema_decay > 0``) is updated each step.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from worldmodel.data import ChunkDataset, collate


def train_world_model(
    model,
    buffer,
    horizon: int,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    frameskip: int = 1,
    device: str | torch.device = 'cpu',
    log_every: int = 10,
    seed: int | None = None,
) -> list[dict]:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    device = torch.device(device)
    model = model.to(device).train()
    dataset = ChunkDataset(buffer, horizon=horizon, frameskip=frameskip)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        collate_fn=collate, num_workers=0,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict] = []
    step = 0
    last_entry = None
    for ep in range(epochs):
        t0 = time.time()
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, metrics = model.jepa_loss(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            model._update_ema()
            entry = {'epoch': ep, 'step': step,
                     **{k: float(v) if not torch.is_tensor(v) else float(v.item()) for k, v in metrics.items()}}
            last_entry = entry
            if step % log_every == 0:
                history.append(entry)
            step += 1
    # Always expose the final training step's metrics, even when log_every is huge
    # (callers read hist[-1] for final loss/dyn_err; without this they'd get step 0
    # — the pre-training initial values — which made epochs look like a no-op).
    if last_entry is not None and (not history or history[-1]['step'] != last_entry['step']):
        history.append(last_entry)
    return history
