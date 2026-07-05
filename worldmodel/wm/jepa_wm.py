"""Plain JEPA world-model builder (CNN / MLP encoder)."""

from __future__ import annotations

import numpy as np
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
    goal_shape: tuple[int, ...] | None = None,
) -> WorldModel:
    if obs_key == 'state':
        encoder: nn.Module = MLPEncoder(obs_dim=int(obs_shape[0]), latent_dim=latent_dim)
    elif obs_key == 'pixels':
        encoder = CNNEncoder(latent_dim=latent_dim, channels=int(obs_shape[0]))
    else:
        raise KeyError(f"Unsupported obs_key '{obs_key}'")
    predictor = MLPPredictor(latent_dim=latent_dim, action_dim=action_dim, hidden=hidden)

    # If goal has a different dimensionality than obs, build a separate
    # goal encoder so get_cost can encode goals of a different shape.
    goal_encoder = None
    if goal_shape is not None:
        goal_dim = int(np.prod(goal_shape))
        obs_dim = int(np.prod(obs_shape))
        if goal_dim != obs_dim:
            if obs_key == 'state':
                goal_encoder = MLPEncoder(obs_dim=goal_dim, latent_dim=latent_dim)
            else:
                goal_encoder = MLPEncoder(obs_dim=goal_dim, latent_dim=latent_dim)

    return WorldModel(
        encoder=encoder,
        predictor=predictor,
        obs_key=obs_key,
        action_dim=action_dim,
        cost_mode=cost_mode,
        ema_decay=ema_decay,
        var_weight=var_weight,
        goal_encoder=goal_encoder,
    )
