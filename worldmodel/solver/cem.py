"""Cross-Entropy Method solver.

Samples ``num_samples`` action sequences per env from a Gaussian, scores them
with the world model's ``get_cost``, keeps the ``topk`` lowest-cost elites,
and refits mean/std. Returns the final mean as the plan.

Adapted (rewritten) from the CEM in stable-worldmodel (MIT). Simplifications:
no per-env micro-batching, no callback hooks, no action-block frameskip —
``action_block=1`` only. Action candidates are clipped to the Box bounds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from worldmodel.solver.base import _box_bounds
from worldmodel.spaces import Box


class CEMSolver:
    def __init__(
        self,
        model,
        num_samples: int = 128,
        n_steps: int = 8,
        topk: int = 16,
        var_scale: float = 1.0,
        var_min: float = 0.0001,
        alpha: float = 0.0,
        device: str | torch.device = 'cpu',
        seed: int = 1234,
    ):
        self.model = model
        self.num_samples = int(num_samples)
        self.n_steps = int(n_steps)
        self.topk = int(topk)
        self.var_scale = float(var_scale)
        self.var_min = float(var_min)
        self.alpha = float(alpha)  # momentum: mean <- (1-a)*new + a*old
        self.device = torch.device(device)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        try:
            self._dtype = next(model.parameters()).dtype
        except (AttributeError, StopIteration):
            self._dtype = torch.float32
        self._configured = False

    # -- configuration --
    def configure(
        self, *, action_space: Box, n_envs: int, config: Any
    ) -> None:
        if not isinstance(action_space, Box):
            raise TypeError('CEMSolver requires a Box action space.')
        self.action_space = action_space
        self._n_envs = int(n_envs)
        self._config = config
        self._action_dim = int(np.prod(action_space.shape))
        self._low, self._high = _box_bounds(
            action_space, self.device, self._dtype
        )
        self._configured = True

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def horizon(self) -> int:
        return self._config.horizon

    def __call__(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        return self.solve(info_dict, init_action)

    # -- core --
    @torch.inference_mode()
    def solve(
        self, info_dict: dict, init_action: torch.Tensor | None = None
    ) -> dict:
        if not self._configured:
            raise RuntimeError('CEMSolver.configure(...) was not called.')

        total_envs = len(next(iter(info_dict.values())))
        H = self.horizon
        D = self._action_dim
        dev = self.device

        # Initial mean (optionally warm-started) and variance.
        if init_action is not None:
            mean = init_action.to(dev, dtype=self._dtype)
            if mean.shape[1] < H:
                pad = torch.zeros(total_envs, H - mean.shape[1], D, device=dev, dtype=self._dtype)
                mean = torch.cat([mean, pad], dim=1)
            elif mean.shape[1] > H:
                mean = mean[:, :H]
        else:
            mean = torch.zeros(total_envs, H, D, device=dev, dtype=self._dtype)
        var = torch.full((total_envs, H, D), self.var_scale, device=dev, dtype=self._dtype)

        # Expand the info dict once to (n_envs, num_samples, ...).
        expanded: dict = {}
        for k, v in info_dict.items():
            if torch.is_tensor(v):
                vv = v.to(device=dev, dtype=self._dtype if v.is_floating_point() else v.dtype)
                expanded[k] = vv.unsqueeze(1).expand(total_envs, self.num_samples, *vv.shape[1:])
            else:
                arr = np.asarray(v)
                expanded[k] = torch.as_tensor(
                    np.repeat(arr[:, None, ...], self.num_samples, axis=1),
                    device=dev,
                    dtype=self._dtype,
                )

        for step in range(self.n_steps):
            eps = torch.randn(
                total_envs, self.num_samples, H, D,
                generator=self.gen, device=dev, dtype=self._dtype,
            )
            candidates = eps * var.unsqueeze(1) + mean.unsqueeze(1)
            candidates = candidates.clamp(self._low, self._high)
            candidates[:, 0] = mean  # elite-injection: keep current mean

            costs = self.model.get_cost(expanded, candidates)
            if costs.shape != (total_envs, self.num_samples):
                raise ValueError(
                    f'get_cost returned shape {tuple(costs.shape)}, '
                    f'expected ({total_envs}, {self.num_samples}).'
                )

            k = min(self.topk, self.num_samples)
            topk_costs, topk_idx = torch.topk(costs, k=k, dim=1, largest=False)
            b_idx = torch.arange(total_envs, device=dev).unsqueeze(1).expand(-1, k)
            elites = candidates[b_idx, topk_idx]  # (n_envs, k, H, D)

            new_mean = elites.mean(dim=1)
            new_var = elites.var(dim=1).clamp_min(self.var_min)
            if self.alpha > 0:
                mean = (1 - self.alpha) * new_mean + self.alpha * mean
            else:
                mean = new_mean
            var = new_var

        return {
            'actions': mean.detach().cpu(),
            'costs': topk_costs.mean(dim=1).detach().cpu(),
        }
