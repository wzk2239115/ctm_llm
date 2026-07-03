"""3-link planar swimmer (goal-conditioned), reimplemented by reference.

Mirrors the DeepMind Control ``swimmer`` "locomote to a target" task with a
hand-rolled approximate fluid-drag dynamics instead of MuJoCo / a full
Lagrangian. Zero-dependency (pure numpy).

Observation / goal / action follow the worldmodel Env protocol (see bench.py)::

    obs = env.reset(seed)            # {'state': ..., 'goal': ...}
    obs, r, term, trunc, info = env.step(action)

The swimmer is a chain of ``n_links`` joints in the plane. Joint 1 sets the
body heading; oscillating joints 2 and 3 produce forward thrust along the
heading, damped by viscous drag. The task is to bring the head (x, y) within
``threshold`` of a target position.

Two modalities (same design as bench.py):
  * **Full obs**  — state = [x, y, a1..a3, da1..da3] (8,). Markov-predictable.
  * **Partial**   — angular velocities are HIDDEN -> state = [x, y, a1..a3] (5).
    The next head position depends on the (unobserved) joint velocities, so a
    1-frame Markov predictor is fundamentally limited; a recurrent predictor
    (stream-ctm) can infer them from history. This is where stream-ctm wins.
"""

from __future__ import annotations

import numpy as np

from worldmodel.spaces import Box


class Swimmer:
    """3-link planar swimmer that locomotes its head to a target xy.

    Simplified fluid-drag dynamics: each action component is a joint
    angular-velocity command; joint-2/3 oscillation drives forward thrust along
    the body heading (joint 1), with multiplicative viscous drag. Bounded
    velocity + clipped arena keep the dynamics stable (never diverges).

    partial obs hides the joint angular velocities (POMDP).
    """

    n_links = 3
    dt = 0.1
    drag = 0.9            # viscous damping factor on forward speed
    prop_gain = 0.08      # joint oscillation -> forward thrust coupling
    max_speed = 0.35      # cap forward speed (stability)
    arena = 3.0           # head x,y lives in [-arena, arena]
    angle_limit = np.pi   # joint-angle half-range (radians)
    max_steps = 200
    threshold = 0.20      # head-to-goal success radius

    def __init__(self, partial=False, seed=None):
        self.partial = partial
        self.rng = np.random.default_rng(seed)
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.n_links,))
        self._pos = np.zeros(2, dtype=np.float32)                          # head x, y
        self._angles = np.zeros(self.n_links, dtype=np.float32)            # a1, a2, a3
        self._avel = np.zeros(self.n_links, dtype=np.float32)              # angular vel
        self._vel = 0.0                                                    # forward speed (scalar)
        self._goal = np.zeros(2, dtype=np.float32)
        self._step = 0
        self.reset(seed=seed)

    @property
    def observation_space(self):
        # x,y in [-arena,arena]; angles in [-pi,pi]; avel (if shown) in [-1,1].
        if self.partial:
            low = np.array([-self.arena, -self.arena, -np.pi, -np.pi, -np.pi], dtype=np.float32)
            high = np.array([self.arena, self.arena, np.pi, np.pi, np.pi], dtype=np.float32)
        else:
            low = np.array(
                [-self.arena, -self.arena, -np.pi, -np.pi, -np.pi, -1.0, -1.0, -1.0],
                dtype=np.float32,
            )
            high = np.array(
                [self.arena, self.arena, np.pi, np.pi, np.pi, 1.0, 1.0, 1.0],
                dtype=np.float32,
            )
        return Box(low=low, high=high)

    @property
    def goal_space(self):
        return Box(low=-self.arena, high=self.arena, shape=(2,))

    def reset(self, seed=None, goal=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._pos = self.rng.uniform(-1.0, 1.0, 2).astype(np.float32)
        self._angles = self.rng.uniform(-0.5, 0.5, self.n_links).astype(np.float32)
        self._avel = np.zeros(self.n_links, dtype=np.float32)
        self._vel = 0.0
        if goal is not None:
            self._goal = np.asarray(goal, dtype=np.float32)
        else:
            # resample a goal far enough away to be non-trivial
            g = self._goal
            for _ in range(32):
                g = self.rng.uniform(-self.arena * 0.8, self.arena * 0.8, 2).astype(np.float32)
                if float(np.linalg.norm(g - self._pos)) > 1.5:
                    break
            self._goal = g
        self._step = 0
        return self._obs()

    def _state_obs(self):
        if self.partial:
            return np.concatenate([self._pos, self._angles]).astype(np.float32)
        return np.concatenate([self._pos, self._angles, self._avel]).astype(np.float32)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._avel = a.copy()
        # integrate joint angles
        self._angles = self._angles + a * self.dt
        # joint 1 (heading) wraps to [-pi, pi]; joints 2,3 clipped
        self._angles[0] = (self._angles[0] + np.pi) % (2 * np.pi) - np.pi
        self._angles[1:] = np.clip(self._angles[1:], -self.angle_limit, self.angle_limit)
        # forward thrust from oscillation of joints 2 and 3, damped by drag
        prop = float(np.sin(self._angles[1]) * a[1] + np.sin(self._angles[2]) * a[2])
        self._vel = (self._vel + self.prop_gain * prop) * self.drag
        self._vel = float(np.clip(self._vel, -self.max_speed, self.max_speed))
        heading = float(self._angles[0])
        self._pos = self._pos + self._vel * np.array(
            [np.cos(heading), np.sin(heading)], dtype=np.float32
        ) * self.dt
        self._pos = np.clip(self._pos, -self.arena, self.arena).astype(np.float32)
        self._step += 1
        dist = float(np.linalg.norm(self._pos - self._goal))
        terminated = dist < self.threshold
        truncated = self._step >= self.max_steps
        return self._obs(), float(terminated), bool(terminated), truncated, {'distance': dist}

    def _obs(self):
        return {'state': self._state_obs(), 'goal': self._goal.copy()}
