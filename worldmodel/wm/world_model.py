"""The JEPA-style world model — the heart of the framework.

A :class:`WorldModel` is a :class:`~worldmodel.protocols.Costable`: given the
current observation and a goal, it can score any batch of candidate action
sequences via latent roll-out, so a sampling solver (CEM) can plan with it.

It is **encoder-agnostic**: pass any encoder whose latent dim matches the
predictor's. Wiring a CTM encoder vs a plain CNN encoder — keeping everything
else fixed — is exactly the CTM-vs-JEPA comparison we want to run.

Training follows a JEPA objective: predict future latents from the current
latent + actions, regressing against the (stop-gradiented) encoder latents of
the real future frames. No pixel reconstruction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        obs_key: str,
        action_dim: int,
        goal_key: str = 'goal',
        cost_mode: str = 'last',
        ema_decay: float = 0.0,
        var_weight: float = 0.0,
        var_gamma: float = 1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.obs_key = obs_key
        self.goal_key = goal_key
        self.action_dim = int(action_dim)
        self.latent_dim = int(getattr(encoder, 'latent_dim'))
        self.cost_mode = cost_mode  # 'last' | 'mean'
        self.ema_decay = float(ema_decay)
        self.var_weight = float(var_weight)   # VICReg-style anti-collapse
        self.var_gamma = float(var_gamma)
        if self.ema_decay > 0:
            import copy
            self.target_encoder = copy.deepcopy(encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad_(False)
        else:
            self.target_encoder = None

    # -- encoding --
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    @torch.no_grad()
    def _encode_target(self, x: torch.Tensor) -> torch.Tensor:
        if self.target_encoder is not None:
            return self.target_encoder(x)
        return self.encoder(x).detach()

    @torch.no_grad()
    def _update_ema(self) -> None:
        if self.target_encoder is None:
            return
        for p, op in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            p.data.mul_(self.ema_decay).add_(op.data, alpha=1 - self.ema_decay)

    # -- latent roll-out --
    def rollout(self, init_latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Roll the predictor forward.

        Args:
            init_latent: ``(..., D)``.
            actions:     ``(..., H, A)``.
        Returns:
            Predicted latents ``(..., H, D)``.

        If the predictor exposes its own ``rollout`` (e.g. the streaming CTM,
        which carries persistent state across ticks), delegate to it so the
        recurrence stays unbroken across the horizon.
        """
        if hasattr(self.predictor, 'rollout'):
            return self.predictor.rollout(init_latent, actions)
        preds = []
        z = init_latent
        H = actions.shape[-2]
        for t in range(H):
            z = self.predictor(z, actions[..., t, :])
            preds.append(z)
        return torch.stack(preds, dim=-2)

    # -- Costable: used by the solver at plan time --
    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        obs = info_dict[self.obs_key]
        goal = info_dict[self.goal_key]
        nE, nS = obs.shape[0], obs.shape[1]
        D = self.latent_dim

        # Flatten (nE, nS, ...) -> (nE*nS, ...) for the encoder.
        obs_flat = obs.reshape(nE * nS, *obs.shape[2:])
        init_latent = self.encode(obs_flat).reshape(nE, nS, D)

        # action_candidates: (nE, nS, H, A)
        pred_latents = self.rollout(init_latent, action_candidates)  # (nE, nS, H, D)

        goal_flat = goal.reshape(nE * nS, *goal.shape[2:])
        goal_latent = self._encode_target(goal_flat).reshape(nE, nS, D)

        if self.cost_mode == 'mean':
            diff = pred_latents - goal_latent.unsqueeze(-2)
        else:  # 'last'
            diff = pred_latents[..., -1, :] - goal_latent
        cost = diff.pow(2).sum(dim=-1)  # (nE, nS)
        return cost

    # -- JEPA training loss --
    def jepa_loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        frames = batch[self.obs_key]  # (B, H+1, *obs_shape)
        actions = batch['action']     # (B, H, A)
        B, Hp1 = frames.shape[0], frames.shape[1]
        H = Hp1 - 1
        D = self.latent_dim

        flat = frames.reshape(B * Hp1, *frames.shape[2:])
        latents = self.encode(flat).reshape(B, Hp1, D)
        init_latent = latents[:, 0]
        target = self._encode_target(
            frames[:, 1:].reshape(B * H, *frames.shape[2:])
        ).reshape(B, H, D)

        pred = self.rollout(init_latent, actions)  # (B, H, D)
        loss = F.mse_loss(pred, target)

        # VICReg variance term: keep every latent dim alive (anti-collapse).
        var_loss = torch.zeros((), device=latents.device)
        if self.var_weight > 0:
            std = latents.reshape(-1, D).std(dim=0)
            var_loss = F.relu(self.var_gamma - std).mean()
            loss = loss + self.var_weight * var_loss

        with torch.no_grad():
            pred_det = pred.detach()
            tgt_det = target.detach()
            dynamics_err = (pred_det - tgt_det).pow(2).mean().item()
            latent_var = latents.var(dim=0).mean().item()
        metrics = {
            'loss': loss.detach(),
            'dynamics_err': dynamics_err,
            'latent_var': latent_var,
            'var_loss': var_loss.detach(),
        }
        return loss, metrics
