import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import CTMLLMConfig


class ObjectiveDenoiseHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        mid = max(1, hidden_size // 2)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mid, bias=False),
            nn.SiLU(),
            nn.Linear(mid, hidden_size, bias=False),
        )

    def forward(self, x):
        return self.net(x)


def objective_enabled(config: CTMLLMConfig) -> bool:
    mode = getattr(config, 'objective_mode', 'none')
    if mode in (None, 'none'):
        return False
    return (
        float(getattr(config, 'objective_denoise_weight', 0.0)) > 0
        or float(getattr(config, 'objective_ce_weight', 0.0)) > 0
        or mode != 'causal_ce'
    )


def _sample_time(batch_size, config: CTMLLMConfig, device, dtype):
    if config.objective_time_schedule == 'logit_normal':
        normal = torch.randn(batch_size, 1, 1, device=device, dtype=dtype)
        return torch.sigmoid(normal * 0.8 + 0.8)
    return torch.rand(batch_size, 1, 1, device=device, dtype=dtype)


def select_objective_latent(input_ids, hidden, embed_tokens, config: CTMLLMConfig):
    space = config.objective_latent_space
    if space == 'token_embed':
        latent = embed_tokens(input_ids)
    elif space == 'frozen_encoder':
        latent = hidden.detach()
    else:
        latent = hidden
    return latent


def compute_objective_loss(
    *,
    config: CTMLLMConfig,
    input_ids,
    labels,
    hidden,
    lm_head,
    denoise_head: ObjectiveDenoiseHead,
    embed_tokens,
    base_ce_loss_fn,
):
    mode = config.objective_mode
    ce_weight = float(config.objective_ce_weight)
    denoise_weight = float(config.objective_denoise_weight)
    zero = hidden.new_zeros(())
    metrics = {'objective_ce': 0.0, 'objective_denoise': 0.0}

    if not objective_enabled(config):
        return zero, metrics

    total = zero
    if ce_weight > 0 and mode in ('causal_ce', 'hybrid_flow_ce'):
        logits = lm_head(hidden)
        ce = base_ce_loss_fn(logits, labels)
        total = total + ce_weight * ce
        metrics['objective_ce'] = float(ce.detach().float().item())

    if denoise_weight > 0 and mode in ('latent_denoise', 'hybrid_flow_ce'):
        latent = select_objective_latent(input_ids, hidden, embed_tokens, config)
        if config.objective_latent_space == 'frozen_encoder':
            latent = latent.detach()

        noise_scale = float(config.objective_decoder_noise_scale)
        cond_drop = float(config.objective_cond_drop_prob)
        if cond_drop > 0 and latent.requires_grad:
            drop = (torch.rand(latent.size(0), 1, 1, device=latent.device) < cond_drop)
            latent = torch.where(drop, torch.zeros_like(latent), latent)

        noise = torch.randn_like(latent) * noise_scale
        t = _sample_time(latent.size(0), config, latent.device, latent.dtype)
        noisy = t * latent + (1.0 - t) * noise

        if float(config.objective_self_cond_prob) > 0:
            with torch.no_grad():
                self_cond = denoise_head(noisy)
            noisy = noisy + float(config.objective_self_cond_prob) * self_cond.detach()

        pred = denoise_head(noisy)
        target = latent - noise
        denoise_loss = F.mse_loss(pred, target)
        total = total + denoise_weight * denoise_loss
        metrics['objective_denoise'] = float(denoise_loss.detach().float().item())

    return total, metrics
