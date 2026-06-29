"""Policies: random, and model-based MPC via a solver.

:func:`WorldModelPolicy` is the model-based planner: every env step it asks the
solver (e.g. CEM) for an action sequence that minimises the world model's cost
to the goal, and executes the first action. (Re-planning every step is the
simplest correct MPC; action reuse / receding-horizon buffering is left as a
future optimisation.)

Keeping the planner this thin means the *same* policy/solver can drive any
:class:`~worldmodel.protocols.Costable` world model — CTM or JEPA — for a clean
head-to-head comparison.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from worldmodel.solver import Solver


@dataclass(frozen=True)
class PlanConfig:
    """MPC planning configuration.

    Attributes:
        horizon: how many steps ahead the solver plans.
        receding_horizon: env steps to execute before re-planning (1 = replan
            every step). Currently only ``1`` is exercised by WorldModelPolicy.
    """

    horizon: int
    receding_horizon: int = 1

    @property
    def plan_len(self) -> int:
        return self.horizon


class BasePolicy:
    def __init__(self, seed: int | None = None):
        self.env = None
        self.seed = seed

    def set_env(self, env) -> None:
        self.env = env

    def get_action(self, info_dict: dict) -> np.ndarray:
        raise NotImplementedError


class RandomPolicy(BasePolicy):
    """Samples an action uniformly from the action space, per env."""

    def get_action(self, info_dict: dict) -> np.ndarray:
        n = self.env.num_envs
        return np.stack([self.env.action_space.sample() for _ in range(n)])


class WorldModelPolicy(BasePolicy):
    """Plans an action with a solver + world model, every step."""

    def __init__(
        self,
        solver: Solver,
        config: PlanConfig,
        transform: dict[str, Callable] | None = None,
    ):
        super().__init__()
        self.solver = solver
        self.cfg = config
        self.transform = transform or {}

    def set_env(self, env) -> None:
        self.env = env
        self.solver.configure(
            action_space=env.action_space,
            n_envs=env.num_envs,
            config=self.cfg,
        )

    def _to_tensor(self, info_dict: dict) -> dict:
        out = {}
        dev = self.solver.device
        for k, v in info_dict.items():
            t = torch.as_tensor(np.asarray(v))
            if t.dtype == torch.float64:
                t = t.float()
            out[k] = t.to(dev)
        return out

    def get_action(self, info_dict: dict) -> np.ndarray:
        info = self._to_tensor(info_dict)
        out = self.solver(info)
        actions = out['actions']  # (num_envs, horizon, action_dim)
        first = actions[:, 0, :].numpy().astype(np.float32)
        return first
