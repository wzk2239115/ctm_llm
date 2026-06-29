"""Plain JEPA world-model builder (CNN / MLP encoder)."""

from __future__ import annotations

import torch.nn as nn

from worldmodel.wm.encoders import CNNEncoder, MLPEncoder
from worldmodel.wm.predictors import MLPPredictor
from worldmodel.wm.world_model import WorldModel


def build_jepa_wm(
    obs_key: str,
    obs_shape: tuple[int, ...],
    action_dim: int,
    latent_dim: int = 64,
    hidden: int = 256,
    cost_mode: str = 'last',
    ema_decay: float = 0.0,
    var_weight: float = 0.0,
) -> WorldModel:
    if obs_key == 'state':
        encoder: nn.Module = MLPEncoder(obs_dim=int(obs_shape[0]), latent_dim=latent_dim)
    elif obs_key == 'pixels':
        encoder = CNNEncoder(latent_dim=latent_dim, channels=int(obs_shape[0]))
    else:
        raise KeyError(f"Unsupported obs_key '{obs_key}'")
    predictor = MLPPredictor(latent_dim=latent_dim, action_dim=action_dim, hidden=hidden)
    return WorldModel(
        encoder=encoder,
        predictor=predictor,
        obs_key=obs_key,
        action_dim=action_dim,
        cost_mode=cost_mode,
        ema_decay=ema_decay,
        var_weight=var_weight,
    )
