"""2D point-reach environments (state and image variants)."""

from __future__ import annotations

import numpy as np
import torch

from worldmodel.spaces import Box


class _PointReachBase:
    """Shared 2D point dynamics + goal-conditioned success logic.

    Agent position ``p`` lives in ``[0, 1]^2``. Action in ``[-1, 1]^2`` is a
    velocity command: ``p <- clip(p + max_speed * dt * a)``. An episode
    succeeds (terminated) when the agent is within ``threshold`` of the goal.
    """

    max_speed = 1.0
    dt = 0.2
    threshold = 0.08
    max_steps = 50

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self._agent = np.zeros(2, dtype=np.float32)
        self._goal = np.zeros(2, dtype=np.float32)
        self._step = 0
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,))
        self.reset(seed=seed)

    # -- subclass hooks --
    def _obs(self) -> dict:
        raise NotImplementedError

    def _goal_obs(self) -> dict:
        raise NotImplementedError

    @property
    def observation_space(self) -> Box:
        raise NotImplementedError

    @property
    def goal_space(self) -> Box:
        raise NotImplementedError

    # -- core API --
    def reset(self, seed: int | None = None, goal: np.ndarray | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._agent = self.rng.uniform(0.1, 0.9, 2).astype(np.float32)
        if goal is not None:
            self._goal = np.asarray(goal, dtype=np.float32)
        else:
            # Resample goal until it is far enough to be non-trivial.
            for _ in range(32):
                g = self.rng.uniform(0.1, 0.9, 2).astype(np.float32)
                if float(np.linalg.norm(g - self._agent)) > 0.3:
                    break
            self._goal = g
        self._step = 0
        return self._obs()

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._agent = np.clip(
            self._agent + self.max_speed * self.dt * action, 0.0, 1.0
        ).astype(np.float32)
        self._step += 1
        dist = float(np.linalg.norm(self._agent - self._goal))
        terminated = dist < self.threshold
        truncated = self._step >= self.max_steps
        reward = float(terminated)
        info = {'distance': dist}
        return self._obs(), reward, terminated, truncated, info

    # -- convenience for collect/eval --
    def set_state(self, agent: np.ndarray, goal: np.ndarray):
        self._agent = np.asarray(agent, dtype=np.float32).copy()
        self._goal = np.asarray(goal, dtype=np.float32).copy()
        self._step = 0


class PointStateReach(_PointReachBase):
    """Vector observation: ``{'state': agent_xy(2), 'goal': goal_xy(2)}``."""

    def _obs(self) -> dict:
        return {'state': self._agent.copy(), 'goal': self._goal.copy()}

    def _goal_obs(self) -> dict:
        return {'state': self._goal.copy()}

    @property
    def observation_space(self) -> Box:
        return Box(low=0.0, high=1.0, shape=(2,))

    @property
    def goal_space(self) -> Box:
        return Box(low=0.0, high=1.0, shape=(2,))


class PointImageReach(_PointReachBase):
    """Image observation: a Gaussian dot rendered at the agent / goal position.

    ``image_size`` square RGB frame; ``sigma`` controls the dot spread. The
    current observation renders the agent dot; ``'goal'`` renders the goal dot,
    so both share the encoder's input distribution.
    """

    def __init__(
        self,
        image_size: int = 32,
        sigma: float = 2.0,
        seed: int | None = None,
    ):
        self.image_size = int(image_size)
        self.sigma = float(sigma)
        coords = np.arange(image_size, dtype=np.float32) / image_size
        gx, gy = np.meshgrid(coords, coords, indexing='xy')  # (H, W), row=y
        self._grid = np.stack([gx, gy], axis=-1)  # (H, W, 2)
        super().__init__(seed=seed)

    def _render(self, pos: np.ndarray) -> np.ndarray:
        # Gaussian blob centred at pos; output (C=3, H, W) float32 in [0, 1].
        d2 = np.sum((self._grid - pos) ** 2, axis=-1)  # (H, W)
        blob = np.exp(-0.5 * d2 / (self.sigma**2)).astype(np.float32)
        img = np.stack([blob, blob, blob], axis=0)  # (3, H, W)
        return img

    def _obs(self) -> dict:
        return {
            'pixels': self._render(self._agent),
            'goal': self._render(self._goal),
        }

    def _goal_obs(self) -> dict:
        return {'pixels': self._render(self._goal)}

    @property
    def observation_space(self) -> Box:
        return Box(
            low=0.0, high=1.0, shape=(3, self.image_size, self.image_size)
        )

    @property
    def goal_space(self) -> Box:
        return self.observation_space


def make_env(name: str, **kwargs):
    """Registry helper: ``make_env('point-state')`` / ``'point-image'``."""
    name = name.lower().replace('_', '-')
    if name in ('point-state', 'pointstate', 'state'):
        return PointStateReach(**kwargs)
    if name in ('point-image', 'pointimage', 'image'):
        return PointImageReach(**kwargs)
    raise KeyError(f"Unknown env '{name}'. Use 'point-state' or 'point-image'.")
