"""Action-conditioned latent predictors.

Given the current latent ``z`` and an action ``a``, predict the next latent.
For a Markov (history=1) world model this is a single feed-forward module; the
:class:`WorldModel` rolls it out autoregressively over the planning horizon.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPPredictor(nn.Module):
    """``concat(z, act_emb) -> z_next`` with a small residual MLP."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden: int = 256,
        act_emb: int = 32,
        n_layers: int = 2,
    ):
        super().__init__()
        self.act_embed = nn.Sequential(
            nn.Linear(action_dim, act_emb), nn.GELU(),
            nn.Linear(act_emb, act_emb),
        )
        layers = []
        d_in = latent_dim + act_emb
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d_in, hidden), nn.GELU()]
            d_in = hidden
        layers += [nn.Linear(d_in, latent_dim)]
        self.net = nn.Sequential(*layers)
        self.residual = latent_dim

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        a = self.act_embed(action.float())
        h = torch.cat([z, a], dim=-1)
        # Residual on the latent stabilises short-horizon dynamics.
        return z + self.net(h)
