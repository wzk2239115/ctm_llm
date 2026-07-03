"""Goal-conditioned MountainCar (gym MountainCar dynamics), reimplemented in
pure numpy (no gymnasium dependency).

Follows the worldmodel Env protocol (same style as bench.Pendulum/CartPole)::

    env = MountainCar()
    obs = env.reset(seed=0)            # {'state': ..., 'goal': ...}
    obs, r, term, trunc, info = env.step(action)

A 1D car sits in a sinusoidal valley; the engine is too weak to drive straight
up the right hill, so the agent must build momentum by swinging back and forth.
State = (position, velocity); continuous action in [-1,1] is the applied force
direction (the natural continuous extension of gym's {-1,0,+1} torque set).

Goal = a target position on the right hill; success = position >= goal_pos.
Reward is sparse: 1.0 on success, 0.0 otherwise (matches bench.py).

* **Full obs** — (position, velocity). Markov; a 1-frame predictor suffices.
* **Partial obs** — velocity is hidden. The next position depends on the unseen
  velocity, so a Markov predictor is fundamentally limited; a recurrent
  predictor that carries thought across ticks can infer velocity from the
  position history. This is where stream-ctm is expected to win.
"""

from __future__ import annotations

import numpy as np

from worldmodel.spaces import Box


class MountainCar:
    """Goal-conditioned mountain car (gym MountainCar transition dynamics).

    Dynamics (per gym MountainCar-v0, force=0.001, gravity=0.0025)::

        velocity += action * force - gravity * cos(3 * position)
        position += velocity

    position clipped to [-1.2, 0.6], velocity clipped to [-0.07, 0.07]; an
    inelastic wall collision zeros a leftward velocity at the left bound.

    partial obs hides velocity -> recurrence must infer it from the position
    history to predict the next state.
    """

    min_position = -1.2
    max_position = 0.6
    max_speed = 0.07
    force = 0.001
    gravity = 0.0025
    max_steps = 200

    def __init__(self, partial=False, seed=None):
        self.partial = partial
        self.rng = np.random.default_rng(seed)
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,))
        self._s = np.zeros(2, dtype=np.float32)  # position, velocity
        self._goal_pos = 0.5
        self._step = 0
        self.reset(seed=seed)

    @property
    def observation_space(self):
        if self.partial:
            return Box(low=self.min_position, high=self.max_position, shape=(1,))
        return Box(low=[self.min_position, -self.max_speed],
                   high=[self.max_position, self.max_speed], shape=(2,))

    @property
    def goal_space(self):
        return self.observation_space

    def reset(self, seed=None, goal=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        position = float(self.rng.uniform(-0.6, -0.4))
        self._s = np.array([position, 0.0], dtype=np.float32)
        self._goal_pos = float(self.rng.uniform(0.45, 0.55))
        if goal is not None:
            g = np.asarray(goal, dtype=np.float32).reshape(-1)
            self._goal_pos = float(g[0])
        self._step = 0
        return self._obs()

    def _goal_state(self):
        return np.array([self._goal_pos, 0.0], dtype=np.float32)

    def step(self, action):
        a = float(np.clip(action[0], -1.0, 1.0))
        position, velocity = self._s
        velocity = velocity + (a * self.force + np.cos(3.0 * position) * (-self.gravity))
        velocity = float(np.clip(velocity, -self.max_speed, self.max_speed))
        position = position + velocity
        position = float(np.clip(position, self.min_position, self.max_position))
        if position == self.min_position and velocity < 0:
            velocity = 0.0
        self._s = np.array([position, velocity], dtype=np.float32)
        self._step += 1
        dist = max(0.0, self._goal_pos - position)
        terminated = position >= self._goal_pos
        truncated = self._step >= self.max_steps
        return (self._obs(), float(terminated), bool(terminated), truncated,
                {'distance': dist, 'position': position, 'velocity': velocity,
                 'goal_position': self._goal_pos})

    def _state_obs(self):
        if self.partial:
            return np.array([self._s[0]], dtype=np.float32)
        return self._s.copy()

    def _goal_obs(self):
        g = self._goal_state()
        if self.partial:
            return np.array([g[0]], dtype=np.float32)
        return g

    def _obs(self):
        return {'state': self._state_obs(), 'goal': self._goal_obs()}
