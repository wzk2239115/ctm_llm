"""Encoders mapping an observation to a latent vector.

Three interchangeable encoders so the world model can be studied under
identical dynamics/solver while varying only the perception front-end:

* :class:`MLPEncoder`     — for vector (state) observations.
* :class:`CNNEncoder`     — small ConvNet for image observations (JEPA baseline).
* :class:`CTMEncoder`     — wraps :class:`baseline.models.ctm.ContinuousThoughtMachine`,
  exposing its final-tick synchronisation representation as the latent. This is
  the CTM-as-world-model-encoder that we benchmark against the plain CNN.

Every encoder is a plain ``nn.Module``: ``(B, *obs_shape) -> (B, latent_dim)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from baseline.models.ctm import ContinuousThoughtMachine


class MLPEncoder(nn.Module):
    """Two-layer MLP encoder for vector observations."""

    def __init__(self, obs_dim: int, latent_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class CNNEncoder(nn.Module):
    """Small ConvNet encoder for image observations ``(B, C, H, W)``."""

    def __init__(self, latent_dim: int, channels: int = 3, hidden: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden * 2, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(hidden * 2, hidden * 4, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden * 4, latent_dim), nn.LayerNorm(latent_dim),
        )
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return self.proj(self.conv(x))


class CTMEncoder(nn.Module):
    """Use a Continuous Thought Machine as the encoder.

    Runs the CTM for ``iterations`` internal ticks over the input image and
    returns the **last-tick synchronisation representation** as the latent.
    The latent dimension is ``ctm.synch_representation_size_out``.

    By default CTM's heavy idea-config attributes (topk, halting, etc.) are
    left at their inert defaults so the encoder behaves as a vanilla CTM.
    """

    def __init__(self, ctm: ContinuousThoughtMachine):
        super().__init__()
        self.ctm = ctm
        self.latent_dim = int(ctm.synch_representation_size_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CTM.forward returns (predictions, certainties, synchronisation_out)
        # (plus an optional extras dict); the latent is synchronisation_out.
        out = self.ctm(x.float())
        return out[2]

    @torch.no_grad()
    def ema_init(self, other: 'CTMEncoder', decay: float) -> None:
        for p, op in zip(self.parameters(), other.parameters()):
            p.data.mul_(decay).add_(op.data, alpha=1 - decay)
